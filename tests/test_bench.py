import os
import sys

import pytest

BENCH = os.path.join(os.path.dirname(__file__), "..", "bench")
sys.path.append(BENCH)
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from extract import build_program, extract_code
from generate import pending
from load_benchmarks import normalize_humaneval, normalize_mbpp
from report import build_outcomes, headroom, wilson
from runner import run_program, selftest

HE_PROMPT = 'def helper(x):\n    return x * 2\n\n\ndef add_twice(a, b):\n    """Add a and b, then double."""\n'
HE_TASK = {
    "id": "humaneval/0", "source": "humaneval", "prompt": HE_PROMPT, "entry_point": "add_twice",
    "tests": "def check(candidate):\n    assert candidate(1, 2) == 6\n    assert candidate(0, 0) == 0\n",
}
MBPP_TASK = {"id": "mbpp/1", "source": "mbpp", "setup": "import math", "entry_point": None,
             "tests": "assert area(2) == math.pi * 4"}


def _reply(code):
    return f"Sure!\n```python\n{code}\n```\nHope that helps."


# ---- extraction and program assembly ----
def test_extract_prefers_block_defining_entry_point():
    reply = "```python\nprint('demo')\n```\n```python\ndef f(x):\n    return x\n```"
    assert "def f" in extract_code(reply, "f")


def test_extract_handles_unterminated_fence_and_raw_code():
    assert "def f" in extract_code("```python\ndef f():\n    pass", "f")
    assert "def f" in extract_code("def f():\n    pass", "f")


def test_extract_drops_main_demo():
    code = extract_code(_reply("def f():\n    return 1\n\nif __name__ == '__main__':\n    print(input())"), "f")
    assert "__main__" not in code and "def f" in code


def test_humaneval_full_function_reuses_prompt_helpers():
    program = build_program(HE_TASK, _reply("def add_twice(a, b):\n    return helper(a + b)"))
    assert run_program(program, "local")["status"] == "pass"


def test_humaneval_body_only_completion_works():
    program = build_program(HE_TASK, "```python\n    return helper(a + b)\n```")
    assert run_program(program, "local")["status"] == "pass"


def test_humaneval_wrong_answer_fails():
    program = build_program(HE_TASK, _reply("def add_twice(a, b):\n    return a + b"))
    assert run_program(program, "local")["status"] == "fail"


def test_mbpp_program_uses_setup_imports():
    ok = build_program(MBPP_TASK, _reply("def area(r):\n    return math.pi * r * r"))
    bad = build_program(MBPP_TASK, _reply("def area(r):\n    return 3 * r * r"))
    assert run_program(ok, "local")["status"] == "pass"
    assert run_program(bad, "local")["status"] == "fail"


# ---- runner ----
def test_runner_distinguishes_pass_fail_timeout():
    assert run_program("assert True", "local")["status"] == "pass"
    assert run_program("assert False", "local")["status"] == "fail"
    assert run_program("while True:\n    pass", "local", timeout=1)["status"] == "timeout"


def test_local_selftest_passes():
    assert selftest("local") is True


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        run_program("pass", "magic")


# ---- loaders ----
def test_normalize_humaneval_and_missing_columns():
    rows = [{"task_id": "HumanEval/0", "prompt": "def f():\n", "test": "t", "entry_point": "f", "canonical_solution": ""}]
    tasks = normalize_humaneval(rows)
    assert tasks[0]["id"] == "humaneval/0" and "def f" in tasks[0]["question"]
    with pytest.raises(KeyError):
        normalize_humaneval([{"task_id": "x"}])


def test_normalize_mbpp_is_deterministic_and_shows_tests():
    rows = [{"task_id": i, "prompt": f"p{i}", "test_list": [f"assert f({i})"], "test_imports": []} for i in range(1, 50)]
    a = normalize_mbpp(rows, 10, seed=42)
    b = normalize_mbpp(list(reversed(rows)), 10, seed=42)
    assert [t["id"] for t in a] == [t["id"] for t in b] and len(a) == 10
    assert "assert f(" in a[0]["question"]


# ---- generation resume and reporting ----
def test_pending_skips_done_pairs():
    tasks = [{"id": "a"}, {"id": "b"}]
    assert [t["id"] for t in pending(tasks, "m", {("a", "m")})] == ["b"]


def test_wilson_interval_is_sane():
    p, lo, hi = wilson(50, 100)
    assert p == 0.5 and lo < 0.5 < hi and 0 <= lo and hi <= 1
    assert wilson(0, 0) == (0.0, 0.0, 0.0)


def _outcomes(passing_by_model):
    tasks = [{"id": f"t{i}", "source": "mbpp", "question": "q" * 40} for i in range(len(passing_by_model["llama3.1:8b"]))]
    models = list(passing_by_model)
    gens = [{"task_id": t["id"], "model": m, "response": "x" * 100} for t in tasks for m in models]
    res = [{"task_id": t["id"], "model": m, "passed": passing_by_model[m][i], "status": "pass" if passing_by_model[m][i] else "fail"}
           for i, t in enumerate(tasks) for m in models]
    return tasks, gens, res, models


def test_headroom_counts_agreement_and_oracle():
    data = {
        "llama3.1:8b":      [True, True, False, False],
        "llama3.2:3b":      [True, False, True, False],
        "mistral:7b":       [True, False, False, False],
        "qwen2.5-coder:7b": [True, False, False, False],
    }
    tasks, gens, res, models = _outcomes(data)
    rows = build_outcomes(tasks, gens, res, models)
    h = headroom(rows, models)
    assert (h["all_models_pass"], h["no_model_passes"], h["mixed"]) == (1, 1, 2)
    assert h["baseline_fails_but_a_cheaper_model_passes"] == 1
    assert h["baseline_passes_but_every_cheaper_model_fails"] == 1
    assert h["oracle_pass_rate"][0] == 0.75
    assert h["oracle_savings_vs_baseline_pct"] > 0


def test_build_outcomes_skips_tasks_missing_a_model():
    data = {m: [True, True] for m in ["llama3.1:8b", "llama3.2:3b", "mistral:7b", "qwen2.5-coder:7b"]}
    tasks, gens, res, models = _outcomes(data)
    gens = [g for g in gens if not (g["task_id"] == "t1" and g["model"] == "mistral:7b")]
    assert [r["id"] for r in build_outcomes(tasks, gens, res, models)] == ["t0"]


def test_first_run_image_pull_noise_is_not_a_sandbox_error(monkeypatch):
    import subprocess
    import runner

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 0, stdout="", stderr="Unable to find image 'python:3.12-slim' locally\nStatus: Downloaded newer image")
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    assert runner.run_program("pass", "docker")["status"] == "pass"


def test_docker_daemon_down_raises_sandbox_error(monkeypatch):
    import subprocess
    import runner

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 125, stdout="", stderr="Cannot connect to the Docker daemon")
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    with pytest.raises(runner.SandboxError):
        runner.run_program("pass", "docker")


def test_error_type_classifies_stderr_tail():
    from inspect_failures import error_type
    assert error_type("Traceback...\nAssertionError", "fail") == "AssertionError"
    assert error_type("File x\n  ^\nSyntaxError: invalid syntax", "fail") == "SyntaxError"
    assert error_type("", "fail") == "NoMessage"
    assert error_type("anything", "timeout") == "Timeout"


def test_jsonl_survives_a_cut_off_last_line(tmp_path):
    from common import append_jsonl, read_jsonl
    path = str(tmp_path / "g.jsonl")
    append_jsonl(path, {"task_id": "a", "model": "m"})
    with open(path, "a", encoding="utf-8") as f:
        f.write('{"task_id": "b", "mod')            # simulated power loss mid-write
    assert [r["task_id"] for r in read_jsonl(path)] == ["a"]
    append_jsonl(path, {"task_id": "c", "model": "m"})  # must not glue onto the broken line
    assert [r["task_id"] for r in read_jsonl(path)] == ["a", "c"]


# ---- judge vs ground truth ----
def test_pair_accuracy_and_top_pick():
    from judge_vs_truth import pair_accuracy, top_pick_pass
    scores = {"a": 9, "b": 5, "c": 5, "d": 2}
    passed = {"a": True, "b": False, "c": True, "d": False}
    # pairs: (a,b)=1, (a,d)=1, (c,b)=0.5 (tie), (c,d)=1  -> 3.5/4
    assert pair_accuracy(scores, passed) == pytest.approx(0.875)
    assert top_pick_pass(scores, passed) == 1.0
    assert top_pick_pass({"a": 7, "b": 7}, {"a": True, "b": False}) == 0.5


def test_mixed_tasks_filters_unanimous():
    from judge_vs_truth import mixed_tasks
    def row(*flags):
        return {"id": "x", "outcomes": {f"m{i}": {"passed": p} for i, p in enumerate(flags)}}
    assert len(mixed_tasks([row(True, True), row(False, False), row(True, False)])) == 1


def test_analyze_detects_a_judge_that_loves_the_wrong_model():
    from judge_vs_truth import analyze
    models = ["llama3.2:3b", "llama3.1:8b", "qwen2.5-coder:7b", "mistral:7b"]
    # The judge always prefers the 8b model; only qwen is actually right.
    judged = [{"task_id": str(i),
               "scores": {"llama3.2:3b": 5, "llama3.1:8b": 9, "qwen2.5-coder:7b": 6, "mistral:7b": 4},
               "passed": {"llama3.2:3b": False, "llama3.1:8b": False, "qwen2.5-coder:7b": True, "mistral:7b": False}}
              for i in range(10)]
    r = analyze(judged)
    assert r["judge_top_pick_passes"][0] == 0.0
    assert r["judge_top_pick_share"]["llama3.1:8b"] == 1.0
    assert r["always_model_passes"]["qwen2.5-coder:7b"] == 1.0
    assert r["mean_judge_score_passing"] < r["mean_judge_score_failing"] + 1  # sanity: numbers computed


# ---- exact-match benchmarks ----
def test_extract_number_variants():
    from exact import extract_number
    assert extract_number("so 6 * 12 = 72.\nFinal answer: 72") == 72
    assert extract_number("Final answer: $1,234.50.") == 1234.5
    assert extract_number("Final answer: 18 dollars") == 18
    assert extract_number("it costs 5 then 7 total, so 12") == 12      # no final line: last number
    assert extract_number("Final answer: 3\nFinal answer: 4") == 4      # last final line wins
    assert extract_number("no digits here") is None


def test_extract_letter_ignores_stray_capital_a():
    from exact import extract_letter
    assert extract_letter("A common mistake is X.\nFinal answer: C") == "C"
    assert extract_letter("Final answer: (b)".replace("(b)", "(B)")) == "B"
    assert extract_letter("Therefore the answer is D.") == "D"
    assert extract_letter("A is wrong because... I think it is hard") is None


def test_grade_exact_gsm8k_and_mmlu():
    from exact import grade_exact
    g = {"source": "gsm8k", "answer": "72"}
    assert grade_exact(g, "Final answer: 72")["status"] == "pass"
    assert grade_exact(g, "Final answer: 71")["status"] == "fail"
    m = {"source": "mmlu", "answer": "C"}
    assert grade_exact(m, "Final answer: C")["status"] == "pass"
    assert grade_exact(m, "Final answer: A")["status"] == "fail"
    assert grade_exact(m, "I am not sure")["status"] == "fail"


def test_normalize_gsm8k_and_mmlu():
    from load_benchmarks import normalize_gsm8k, normalize_mmlu
    g = normalize_gsm8k([{"question": f"q{i}", "answer": f"work\n#### {i},000"} for i in range(20)], 5, 42)
    assert len(g) == 5 and g[0]["kind"] == "exact" and "," not in g[0]["answer"]
    m = normalize_mmlu([{"question": "q", "subject": "s", "choices": ["w", "x", "y", "z"], "answer": 2}] * 10, 3, 42)
    assert m[0]["answer"] == "C" and "C. y" in m[0]["question"]


def test_per_source_report_finds_category_specialists():
    from report import per_source
    models = ["llama3.2:3b", "llama3.1:8b", "qwen2.5-coder:7b", "mistral:7b"]
    def row(src, winner):
        return {"source": src, "outcomes": {m: {"passed": m == winner, "cost": 1.0} for m in models}}
    rows = [row("humaneval", "qwen2.5-coder:7b")] * 6 + [row("gsm8k", "llama3.1:8b")] * 6
    ps = per_source(rows, models)
    assert ps["best_model_per_source"] == {"humaneval": "qwen2.5-coder:7b", "gsm8k": "llama3.1:8b"}
    assert ps["category_aware_pass_rate"][0] == 1.0 and ps["best_single_pass_rate"][0] == 0.5
