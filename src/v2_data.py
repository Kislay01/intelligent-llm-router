"""v2 data layer: pools the 464 objective benchmark tasks with the judge-labelled queries.

Each query gets, for every served model, a binary "passed" outcome:
  - objective tasks: unit-test / exact-match result (ground truth)
  - judge-labelled queries: judge score >= a threshold calibrated against ground truth (noisier, so lower weight)
"""
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

sys.path.append(os.path.dirname(__file__))

from evaluation import outcome_matrices
from pricing import SERVED_MODELS

BENCH_DIR = "data/bench"
TASKS_FILE = f"{BENCH_DIR}/tasks.json"
OUTCOMES_FILE = f"{BENCH_DIR}/outcomes.jsonl"
JUDGE_RUNS_FILE = f"{BENCH_DIR}/judge_runs.jsonl"
LABELED_FILE = "data/labeled_results.jsonl"
EMBEDDING_CACHE = "data/embeddings_v2.npz"
JUDGED = "judged"

_MBPP_TESTS_MARKER = "\n\nYour code must pass these tests:"


def read_jsonl(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def raw_query(task: dict) -> str:
    """The text a user would type: the task without our benchmark instruction wrapper.

    Models were prompted with the wrapper, but the router only ever sees raw user text at serve time,
    so it must be trained on raw text. (The unit tests are also dropped for MBPP.)
    """
    if task["source"] == "humaneval":
        return task["prompt"].strip()
    _instruction, _, body = task["question"].partition("\n\n")
    if task["source"] == "mbpp":
        body = body.split(_MBPP_TESTS_MARKER)[0]
    return body.strip()


@dataclass
class Calibration:
    threshold: int           # judge score >= threshold counts as "passed"
    balanced_accuracy: float
    accuracy: float
    n_pairs: int
    table: dict              # threshold -> balanced accuracy


def calibrate_threshold(judge_runs: Sequence[dict], outcomes: Sequence[dict]) -> Calibration:
    """Pick the judge-score cut-off that best reproduces ground-truth pass/fail.

    Caveat: judge_runs only cover tasks where the models disagreed, so this is calibrated on the
    harder-to-judge subset.
    """
    truth = {o["id"]: o["outcomes"] for o in outcomes}
    pairs = [(s, bool(truth[r["task_id"]][m]["passed"]))
             for r in judge_runs if r["task_id"] in truth
             for m, s in r["scores"].items() if m in truth[r["task_id"]]]
    if not pairs:
        raise ValueError("no judge scores overlap with graded outcomes")
    scores = np.array([s for s, _ in pairs])
    passed = np.array([p for _, p in pairs])
    table = {}
    for t in range(1, 11):
        pred = scores >= t
        tpr = pred[passed].mean() if passed.any() else 0.0
        tnr = (~pred)[~passed].mean() if (~passed).any() else 0.0
        table[t] = float((tpr + tnr) / 2)
    best = max(table, key=lambda t: (table[t], -t))
    accuracy = float(((scores >= best) == passed).mean())
    return Calibration(best, table[best], accuracy, len(pairs), table)


@dataclass
class Pool:
    ids: List[str]
    texts: List[str]
    source: np.ndarray       # humaneval / mbpp / gsm8k / mmlu / judged
    category: np.ndarray     # stratification key (benchmark name, or the judged query's category)
    passed: np.ndarray       # (n, n_models) float 0/1
    cost: np.ndarray         # (n, n_models) USD
    judge: np.ndarray        # (n, n_models) judge score, NaN for objective tasks
    weight: np.ndarray       # (n,) training weight
    X: Optional[np.ndarray] = None

    @property
    def objective(self) -> np.ndarray:
        return np.where(self.source != JUDGED)[0]

    @property
    def judged(self) -> np.ndarray:
        return np.where(self.source == JUDGED)[0]


def cached_embed(texts: Sequence[str], path: str = EMBEDDING_CACHE,
                 embed_fn: Optional[Callable[[List[str]], np.ndarray]] = None) -> np.ndarray:
    """Unit-length embeddings, computed once per distinct text and cached on disk by text hash."""
    keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
    store = {}
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as z:
            store = dict(zip(z["keys"].tolist(), z["vectors"]))
    missing = sorted({k: t for k, t in zip(keys, texts) if k not in store}.items())
    if missing:
        if embed_fn is None:
            from embed import get_embedder
            model = get_embedder()
            embed_fn = lambda batch: model.encode(batch, batch_size=64, normalize_embeddings=True)  # noqa: E731
        vectors = np.asarray(embed_fn([t for _, t in missing]), dtype=np.float32)
        store.update({k: v for (k, _), v in zip(missing, vectors)})
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez(path, keys=np.array(list(store)), vectors=np.stack(list(store.values())))
    return np.stack([store[k] for k in keys])


def build_pool(judged_weight: float = 0.5, threshold: Optional[int] = None,
               embed_fn: Optional[Callable] = None, cache: str = EMBEDDING_CACHE,
               paths: Optional[dict] = None) -> tuple:
    """Returns (Pool, Calibration, n_judged_dropped). Judged rows missing any served model's score are dropped."""
    p = {"tasks": TASKS_FILE, "outcomes": OUTCOMES_FILE, "runs": JUDGE_RUNS_FILE, "labeled": LABELED_FILE}
    p.update(paths or {})
    with open(p["tasks"], encoding="utf-8") as f:
        tasks = {t["id"]: t for t in json.load(f)}
    outcomes = read_jsonl(p["outcomes"])
    calibration = calibrate_threshold(read_jsonl(p["runs"]), outcomes)
    cut = calibration.threshold if threshold is None else threshold

    ids, texts, source, category = [], [], [], []
    passed, cost, judge, weight = [], [], [], []
    for o in outcomes:
        ids.append(o["id"])
        texts.append(raw_query(tasks[o["id"]]))
        source.append(o["source"])
        category.append(o["source"])
        passed.append([float(o["outcomes"][m]["passed"]) for m in SERVED_MODELS])
        cost.append([o["outcomes"][m]["cost"] for m in SERVED_MODELS])
        judge.append([np.nan] * len(SERVED_MODELS))
        weight.append(1.0)

    rows = read_jsonl(p["labeled"])
    quality, row_cost = outcome_matrices(rows)
    complete = ~np.isnan(quality).any(axis=1)
    for i in np.where(complete)[0]:
        ids.append(f"judged/{rows[i]['id']}")
        texts.append(rows[i]["query"])
        source.append(JUDGED)
        category.append(rows[i]["category"])
        passed.append((quality[i] >= cut).astype(float).tolist())
        cost.append(row_cost[i].tolist())
        judge.append(quality[i].tolist())
        weight.append(judged_weight)

    pool = Pool(ids, texts, np.array(source), np.array(category), np.array(passed), np.array(cost),
                np.array(judge), np.array(weight))
    pool.X = cached_embed(texts, cache, embed_fn)
    return pool, calibration, int((~complete).sum())
