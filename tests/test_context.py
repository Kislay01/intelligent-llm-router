import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import context
from context import build_chat_messages, build_rewrite_prompt, clean_rewrite, rewrite_query


def _history(n):
    return [{"query": f"q{i}", "response": f"a{i}", "routed_model": "llama3.2:3b"} for i in range(n)]


def test_chat_messages_end_with_the_new_query_and_alternate_roles():
    msgs = build_chat_messages(_history(2), "explain more")
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant", "user"]
    assert msgs[-1] == {"role": "user", "content": "explain more"}


def test_only_recent_exchanges_are_sent():
    msgs = build_chat_messages(_history(10), "x")
    assert len(msgs) == context.MAX_EXCHANGES * 2 + 1
    assert msgs[0]["content"] == "q7"


def test_long_earlier_answers_are_clipped():
    history = [{"query": "q", "response": "z" * 10_000, "routed_model": "m"}]
    msgs = build_chat_messages(history, "x")
    assert len(msgs[1]["content"]) < context.MAX_CHARS_PER_MESSAGE + 10


def test_rewrite_prompt_contains_history_and_query():
    prompt = build_rewrite_prompt(_history(1), "now in Java")
    assert "q0" in prompt and "a0" in prompt and "now in Java" in prompt


@pytest.mark.parametrize("raw,expected", [
    ('"Explain quicksort more simply"', "Explain quicksort more simply"),
    ("Standalone message: Explain quicksort", "Explain quicksort"),
    ("\n\nExplain quicksort\nExtra chatter", "Explain quicksort"),
    ("   ", "original"),
])
def test_clean_rewrite(raw, expected):
    assert clean_rewrite(raw, "original") == expected


def test_rewrite_falls_back_to_original_on_error_when_serving():
    def broken(model, prompt):
        raise RuntimeError("ollama down")
    assert rewrite_query(_history(1), "more", generate=broken) == "more"


def test_rewrite_raises_in_strict_mode():
    def broken(model, prompt):
        raise RuntimeError("ollama down")
    with pytest.raises(RuntimeError):
        rewrite_query(_history(1), "more", generate=broken, strict=True)
