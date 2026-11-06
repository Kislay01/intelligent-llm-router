import os
import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = f"{OLLAMA_HOST}/api/generate"
OLLAMA_CHAT_URL = f"{OLLAMA_HOST}/api/chat"


def query_model(model: str, prompt: str, options: dict | None = None, timeout: int = 120) -> str:
    """Send a prompt to a local Ollama model and return its text response.

    `options` is Ollama's sampling options (e.g. {"temperature": 0}); omitted by default.
    """
    payload = {"model": model, "prompt": prompt, "stream": False}
    if options:
        payload["options"] = options
    response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()["response"]


def query_chat(model: str, messages: list) -> str:
    """Send a chat history ([{"role": "user"|"assistant", "content": str}, ...]) to an Ollama model."""
    response = requests.post(
        OLLAMA_CHAT_URL,
        json={"model": model, "messages": messages, "stream": False},
        timeout=120
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


if __name__ == "__main__":
    result = query_model("llama3.2:3b", "What is 2 + 2?")
    print(result)