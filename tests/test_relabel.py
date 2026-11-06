import os
import sys

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from relabel import cost_aware_label, get_label


def test_cheapest_within_tolerance_wins():
    scores = {"llama3.2:3b": 9, "llama3.1:8b": 10, "mistral:7b": 8, "qwen2.5-coder:7b": 5}
    # 3b (0.06) is within 1 of best (10) and cheaper than 8b (0.18)
    assert cost_aware_label(scores, tolerance=1) == "llama3.2:3b"


def test_tolerance_zero_requires_best_score():
    scores = {"llama3.2:3b": 9, "llama3.1:8b": 10, "mistral:7b": 8, "qwen2.5-coder:7b": 5}
    assert cost_aware_label(scores, tolerance=0) == "llama3.1:8b"


def test_price_tie_goes_to_higher_score():
    scores = {"llama3.2:3b": 9, "mistral:7b": 10, "llama3.1:8b": 10, "qwen2.5-coder:7b": 3}
    assert cost_aware_label(scores, tolerance=1) == "mistral:7b"


def test_unserved_model_is_never_a_label():
    scores = {"codellama:7b": 10, "llama3.1:8b": 8, "llama3.2:3b": 7, "mistral:7b": 7, "qwen2.5-coder:7b": 6}
    assert cost_aware_label(scores, tolerance=0) == "llama3.1:8b"


def test_no_served_scores_raises():
    with pytest.raises(ValueError):
        cost_aware_label({"codellama:7b": 9})


def test_none_tolerance_returns_judge_winner():
    row = {"winner": "mistral:7b", "scores": {"llama3.2:3b": 10, "mistral:7b": 10}}
    assert get_label(row, None) == "mistral:7b"


def test_judge_labels_never_return_unserved_model():
    row = {"winner": "codellama:7b",
           "scores": {"codellama:7b": 10, "llama3.1:8b": 9, "llama3.2:3b": 7,
                      "mistral:7b": 7, "qwen2.5-coder:7b": 5}}
    assert get_label(row, None) == "llama3.1:8b"
