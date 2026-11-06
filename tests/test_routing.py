import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from routing import resolve_route

HISTORY = [{"query": "Explain quicksort", "response": "...", "routed_model": "qwen2.5-coder:7b"}]


def _predict(table):
    """predict(text) from a dict text -> (model, confidence)."""
    def predict(text):
        model, conf = table.get(text, ("llama3.2:3b", 0.1))
        return model, conf, {model: conf}
    return predict


def _rewrite(history, query):
    return "Explain quicksort more simply"


def test_none_mode_ignores_history_and_uses_global_fallback():
    d = resolve_route("none", "explain more", HISTORY, _predict({}), _rewrite)
    assert d.routing_query == "explain more"
    assert d.fallback_triggered and d.model == "llama3.1:8b"


def test_sticky_mode_keeps_previous_model_when_unsure():
    d = resolve_route("sticky", "explain more", HISTORY, _predict({}), _rewrite)
    assert d.model == "qwen2.5-coder:7b" and d.fallback_triggered


def test_sticky_mode_trusts_a_confident_prediction():
    d = resolve_route("sticky", "capital of France?", HISTORY,
                      _predict({"capital of France?": ("llama3.2:3b", 0.9)}), _rewrite)
    assert d.model == "llama3.2:3b" and not d.fallback_triggered


def test_sticky_without_history_uses_global_fallback():
    d = resolve_route("sticky", "hi", [], _predict({}), _rewrite)
    assert d.model == "llama3.1:8b"


def test_rewrite_mode_routes_the_rewritten_text():
    table = {"Explain quicksort more simply": ("qwen2.5-coder:7b", 0.8)}
    d = resolve_route("rewrite", "explain more", HISTORY, _predict(table), _rewrite)
    assert d.routing_query == "Explain quicksort more simply"
    assert d.model == "qwen2.5-coder:7b" and not d.fallback_triggered


def test_rewrite_mode_does_not_rewrite_the_first_message():
    calls = []
    def rewrite(history, query):
        calls.append(query)
        return "changed"
    d = resolve_route("rewrite", "first message", [], _predict({}), rewrite)
    assert d.routing_query == "first message" and calls == []


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        resolve_route("magic", "x", [], _predict({}), _rewrite)
