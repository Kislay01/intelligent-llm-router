"""Does routing have headroom on objective labels? Reads graded results, prints the answer.

Writes data/bench/outcomes.jsonl (input for the next modelling step) and data/bench/headroom.json.
Usage: python bench/report.py
"""
import json
import math
import os
import sys

sys.path.append(os.path.dirname(__file__))

from common import (GENERATIONS_FILE, HEADROOM_FILE, MODELS, OUTCOMES_FILE, RESULTS_FILE,
                    load_tasks, read_jsonl)
from pricing import estimate_cost

BASELINE = "llama3.1:8b"


def wilson(passes: int, n: int, z: float = 1.96):
    """95% Wilson interval for a pass rate."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = passes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (p, centre - half, centre + half)


def build_outcomes(tasks, generations, results, models=MODELS):
    """Tasks that every model answered AND that were graded for every model."""
    gens = {(g["task_id"], g["model"]): g for g in generations}
    res = {(r["task_id"], r["model"]): r for r in results}
    rows = []
    for t in tasks:
        if not all((t["id"], m) in gens and (t["id"], m) in res for m in models):
            continue
        rows.append({
            "id": t["id"], "source": t["source"], "question": t["question"],
            "outcomes": {m: {"passed": bool(res[(t["id"], m)]["passed"]),
                             "status": res[(t["id"], m)]["status"],
                             "cost": estimate_cost(t["question"], gens[(t["id"], m)]["response"], m)}
                         for m in models},
        })
    return rows


def headroom(rows, models=MODELS, baseline=BASELINE):
    n = len(rows)
    passed = {m: sum(r["outcomes"][m]["passed"] for r in rows) for m in models}
    cost = {m: sum(r["outcomes"][m]["cost"] for r in rows) for m in models}
    all_pass = sum(all(r["outcomes"][m]["passed"] for m in models) for r in rows)
    none_pass = sum(not any(r["outcomes"][m]["passed"] for m in models) for r in rows)

    # Cost-aware oracle: the cheapest model that passes (if none passes, it cannot help).
    oracle_pass, oracle_cost = 0, 0.0
    for r in rows:
        ok = [m for m in models if r["outcomes"][m]["passed"]]
        pick = min(ok, key=lambda m: r["outcomes"][m]["cost"]) if ok else min(models, key=lambda m: r["outcomes"][m]["cost"])
        oracle_pass += bool(ok)
        oracle_cost += r["outcomes"][pick]["cost"]

    cheaper = [m for m in models if m != baseline]
    return {
        "n_tasks": n,
        "pass_rate": {m: wilson(passed[m], n) for m in models},
        "total_cost": cost,
        "all_models_pass": all_pass, "no_model_passes": none_pass, "mixed": n - all_pass - none_pass,
        "baseline_fails_but_a_cheaper_model_passes":
            sum(not r["outcomes"][baseline]["passed"] and any(r["outcomes"][m]["passed"] for m in cheaper) for r in rows),
        "baseline_passes_but_every_cheaper_model_fails":
            sum(r["outcomes"][baseline]["passed"] and not any(r["outcomes"][m]["passed"] for m in cheaper) for r in rows),
        "oracle_pass_rate": wilson(oracle_pass, n),
        "oracle_cost": oracle_cost,
        "oracle_savings_vs_baseline_pct": (1 - oracle_cost / cost[baseline]) * 100 if cost[baseline] else 0.0,
    }


def per_source(rows, models=MODELS):
    """Pass rate of each model within each benchmark, and what choosing the best model per benchmark would give."""
    sources = sorted({r["source"] for r in rows})
    table = {}
    for src in sources:
        sub = [r for r in rows if r["source"] == src]
        table[src] = {"n": len(sub),
                      "pass": {m: wilson(sum(r["outcomes"][m]["passed"] for r in sub), len(sub)) for m in models}}
    best_single = max(models, key=lambda m: sum(r["outcomes"][m]["passed"] for r in rows))
    per_source_best = {src: max(models, key=lambda m: table[src]["pass"][m][0]) for src in sources}
    by_source_passes = sum(r["outcomes"][per_source_best[r["source"]]]["passed"] for r in rows)
    return {"table": table, "best_single_model": best_single, "best_model_per_source": per_source_best,
            "best_single_pass_rate": wilson(sum(r["outcomes"][best_single]["passed"] for r in rows), len(rows)),
            "category_aware_pass_rate": wilson(by_source_passes, len(rows))}


def main():
    tasks = load_tasks()
    rows = build_outcomes(tasks, read_jsonl(GENERATIONS_FILE), read_jsonl(RESULTS_FILE))
    if not rows:
        sys.exit("No task has been answered and graded by all four models yet.")
    with open(OUTCOMES_FILE, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    h = headroom(rows)
    with open(HEADROOM_FILE, "w", encoding="utf-8") as f:
        json.dump(h, f, indent=1)

    print(f"\n{h['n_tasks']} tasks answered and graded by all {len(MODELS)} models "
          f"({len(tasks)} in the benchmark)\n")
    print(f"{'Model':<20}{'Pass rate (95% CI)':<28}{'Total cost ($)'}")
    for m in MODELS:
        p, lo, hi = h["pass_rate"][m]
        print(f"{m:<20}{p*100:5.1f}% [{lo*100:.1f}, {hi*100:.1f}]{'':<7}{h['total_cost'][m]:.6f}")
    p, lo, hi = h["oracle_pass_rate"]
    print(f"{'Oracle (cheapest passing)':<20}{p*100:5.1f}% [{lo*100:.1f}, {hi*100:.1f}]{'':<7}{h['oracle_cost']:.6f}")
    print(f"\nWhere routing can matter (per task):")
    print(f"  every model passes:   {h['all_models_pass']:>4}   (any model will do: pick the cheapest)")
    print(f"  no model passes:      {h['no_model_passes']:>4}   (nothing to gain)")
    print(f"  models disagree:      {h['mixed']:>4}   <- the only tasks where the choice changes correctness")
    print(f"  {BASELINE} fails but a cheaper one passes: {h['baseline_fails_but_a_cheaper_model_passes']}")
    print(f"  {BASELINE} passes but every cheaper one fails: {h['baseline_passes_but_every_cheaper_model_fails']}")
    print(f"\nCost-aware oracle saves {h['oracle_savings_vs_baseline_pct']:.1f}% vs always-{BASELINE}, "
          f"at pass rate {h['oracle_pass_rate'][0]*100:.1f}% vs {h['pass_rate'][BASELINE][0]*100:.1f}%.")
    ps = per_source(rows)
    if len(ps["table"]) > 1:
        print(f"\nPass rate by benchmark:")
        print(f"{'Benchmark':<12}{'n':>5}  " + "".join(f"{m:<20}" for m in MODELS))
        for src, t in ps["table"].items():
            print(f"{src:<12}{t['n']:>5}  " + "".join(f"{t['pass'][m][0]*100:5.1f}%{'':<14}" for m in MODELS))
        b, c = ps["best_single_pass_rate"], ps["category_aware_pass_rate"]
        print(f"\nBest single model overall ({ps['best_single_model']}): {b[0]*100:.1f}% [{b[1]*100:.1f}, {b[2]*100:.1f}]")
        print(f"Best model per benchmark {ps['best_model_per_source']}: {c[0]*100:.1f}% [{c[1]*100:.1f}, {c[2]*100:.1f}]")
        h["per_source"] = ps
        with open(HEADROOM_FILE, "w", encoding="utf-8") as f:
            json.dump(h, f, indent=1)
    print(f"\nSaved {OUTCOMES_FILE} and {HEADROOM_FILE}")


if __name__ == "__main__":
    main()
