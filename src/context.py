"""Conversation context: what the answering model sees, and how a follow-up becomes a standalone query."""
from typing import Callable, List

from ollama_client import query_model

REWRITE_MODEL = "llama3.2:3b"  # cheapest served model; the rewrite is a small, easy task
MAX_EXCHANGES = 3              # how many previous user/assistant pairs are sent
MAX_CHARS_PER_MESSAGE = 1500   # long earlier answers are cut so the context stays small
MAX_REWRITE_CHARS = 500

REWRITE_PROMPT = """Rewrite the user's latest message as one standalone question or request that can be understood without the conversation. Keep every specific detail (topic, programming language, constraints, tone). Do NOT answer it. Output only the rewritten message.

Conversation:
{conversation}

Latest message: {query}

Standalone message:"""


def _clip(text: str) -> str:
    text = text or ""
    return text if len(text) <= MAX_CHARS_PER_MESSAGE else text[:MAX_CHARS_PER_MESSAGE] + " ..."


def recent_history(history: List[dict]) -> List[dict]:
    return history[-MAX_EXCHANGES:]


def build_chat_messages(history: List[dict], query: str) -> List[dict]:
    """Messages for Ollama's chat endpoint: recent exchanges, then the new user message."""
    messages = []
    for item in recent_history(history):
        messages.append({"role": "user", "content": _clip(item["query"])})
        messages.append({"role": "assistant", "content": _clip(item["response"])})
    messages.append({"role": "user", "content": query})
    return messages


def build_rewrite_prompt(history: List[dict], query: str) -> str:
    conversation = "\n".join(
        f"User: {_clip(item['query'])}\nAssistant: {_clip(item['response'])}"
        for item in recent_history(history)
    )
    return REWRITE_PROMPT.format(conversation=conversation, query=query)


def clean_rewrite(raw: str, fallback: str) -> str:
    """Models often wrap the answer in quotes or add a label; keep the first non-empty line."""
    for line in (raw or "").strip().splitlines():
        line = line.strip().strip('"\'`').strip()
        if line.lower().startswith("standalone message:"):
            line = line.split(":", 1)[1].strip()
        if line:
            return line[:MAX_REWRITE_CHARS]
    return fallback


def rewrite_query(history: List[dict], query: str, generate: Callable[[str, str], str] = query_model,
                  strict: bool = False) -> str:
    """Standalone version of a follow-up.

    Serving (strict=False): any failure falls back to the original query so the user still gets an answer.
    Experiments (strict=True): errors propagate, otherwise a dead Ollama would silently make
    'rewrite' identical to 'none'.
    """
    try:
        return clean_rewrite(generate(REWRITE_MODEL, build_rewrite_prompt(history, query)), query)
    except Exception:
        if strict:
            raise
        return query
