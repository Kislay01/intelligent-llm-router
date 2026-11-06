import json
import os
import sys
from unittest.mock import patch

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import backfill_qwen as bq


def _old_row():
    return {
        "id": 1, "category": "simple", "subtopic": "geo", "query": "Capital of Japan?",
        "responses": {"llama3.2:3b": "Tokyo", "llama3.1:8b": "Tokyo.", "codellama:7b": "Tokyo",
                      "mistral:7b": "Tokyo, a big city"},
        "scores": {"llama3.2:3b": 10, "llama3.1:8b": 10, "codellama:7b": 9, "mistral:7b": 10},
        "winner": "mistral:7b",
    }


def test_needs_backfill():
    row = _old_row()
    assert bq.needs_backfill(row)
    row["scores"]["qwen2.5-coder:7b"] = 8
    row["responses"]["qwen2.5-coder:7b"] = "Tokyo"
    assert not bq.needs_backfill(row)


def test_rebuild_row_replaces_pool_and_keeps_legacy():
    judgment = {"scores": {"llama3.2:3b": 9, "llama3.1:8b": 10, "qwen2.5-coder:7b": 7, "mistral:7b": 8},
                "winner": "wrong-on-purpose"}
    new = bq.rebuild_row(_old_row(), "Tokyo!", judgment)
    assert set(new["scores"]) == set(bq.OLLAMA_MODELS)
    assert "codellama:7b" not in new["responses"]
    assert new["responses"]["qwen2.5-coder:7b"] == "Tokyo!"
    assert new["winner"] == "llama3.1:8b"  # derived from scores, not the judge's field
    assert new["legacy_winner"] == "mistral:7b"
    assert new["legacy_scores"]["codellama:7b"] == 9
    assert not bq.needs_backfill(new)


@pytest.mark.parametrize("scores", [
    {"llama3.2:3b": 9, "llama3.1:8b": 10, "qwen2.5-coder:7b": 7},                       # missing model
    {"llama3.2:3b": 9, "llama3.1:8b": 11, "qwen2.5-coder:7b": 7, "mistral:7b": 8},       # out of range
    {"llama3.2:3b": 9, "llama3.1:8b": 10, "qwen2.5-coder:7b": 7, "codellama:7b": 8},    # wrong pool
])
def test_validate_scores_rejects_bad_judgments(scores):
    with pytest.raises(ValueError):
        bq.validate_scores(scores)


def test_save_rows_roundtrip(tmp_path):
    path = str(tmp_path / "r.jsonl")
    bq.save_rows([_old_row()], path)
    assert bq.load_rows(path)[0]["id"] == 1
    assert not os.path.exists(path + ".tmp")


@patch("backfill_qwen.time.sleep")
@patch("backfill_qwen.ask_gemini")
def test_judge_retries_then_succeeds(mock_gemini, _sleep):
    ok = json.dumps({"scores": {m: 8 for m in bq.OLLAMA_MODELS}, "winner": "x"})
    mock_gemini.side_effect = [RuntimeError("503"), ok]
    assert bq.judge_with_retry("p")["scores"]["mistral:7b"] == 8
    assert mock_gemini.call_count == 2
