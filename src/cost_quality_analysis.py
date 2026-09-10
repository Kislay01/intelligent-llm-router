import json
import os
import pickle
import random
import sys

sys.path.append(os.path.dirname(__file__))

from embed import embed_query
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

LABELED_FILE = "data/labeled_results.jsonl"
MODEL_PATH = "models/classifier.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
OUTPUT_CHART = "data/cost_quality_chart.png"

# ---------------------------------------------------------------------------
# Real pricing sourced from Together AI (per 1M tokens, converted to per 1K).
# NOTE: codellama:7b is not listed on Together AI's current catalog (older/
# deprecated model) - using a placeholder matching the other 7B-tier models.
# Since codellama wins only 2/174 queries, this has minimal effect on the
# random-routing baseline, but confirm/replace if you find a real source.
# ---------------------------------------------------------------------------
MODEL_PRICING_PER_1M_TOKENS = {
    "llama3.2:3b": 0.06,
    "llama3.1:8b": 0.18,
    "mistral:7b": 0.06,
    "qwen2.5-coder:7b": 0.07,
    "codellama:7b": 0.06,  # placeholder - not listed on Together AI, verify separately
}
MODEL_PRICING_PER_1K_TOKENS = {
    model: price / 1000 for model, price in MODEL_PRICING_PER_1M_TOKENS.items()
}

BASELINE_MODEL = "llama3.1:8b"  # the empirically strongest all-rounder from your data


def estimate_tokens(text: str) -> float:
    """Rough heuristic: ~4 characters per token for English text."""
    return len(text) / 4


def estimate_cost(query_text: str, response_text: str, model: str) -> float:
    price_per_1k = MODEL_PRICING_PER_1K_TOKENS[model]
    total_tokens = estimate_tokens(query_text) + estimate_tokens(response_text)
    return (total_tokens / 1000) * price_per_1k


def load_labeled_data():
    rows = []
    with open(LABELED_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    rows = load_labeled_data()

    with open(MODEL_PATH, "rb") as f:
        classifier = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    y_raw = [r["winner"] for r in rows]
    le_check = LabelEncoder()
    y = le_check.fit_transform(y_raw)

    # Reproduce the EXACT same train/test split used in train_classifier.py
    indices = list(range(len(rows)))
    try:
        idx_train, idx_test, _, _ = train_test_split(
            indices, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        idx_train, idx_test, _, _ = train_test_split(
            indices, y, test_size=0.2, random_state=42
        )

    test_rows = [rows[i] for i in idx_test]
    print(f"Evaluating on {len(test_rows)} held-out test queries (never used in training).\n")

    router_cost, router_quality = 0.0, 0.0
    baseline_cost, baseline_quality = 0.0, 0.0
    random_cost, random_quality = 0.0, 0.0

    random.seed(42)

    for row in test_rows:
        query_text = row["query"]
        available_models = list(row["responses"].keys())

        # --- Router's actual choice ---
        embedding = embed_query(query_text)
        predicted_idx = classifier.predict([embedding])[0]
        predicted_model = label_encoder.classes_[predicted_idx]

        if predicted_model in row["responses"]:
            resp = row["responses"][predicted_model]
            score = row["scores"].get(predicted_model, 0)
            router_cost += estimate_cost(query_text, resp, predicted_model)
            router_quality += score

        # --- Baseline: always use the strongest model ---
        if BASELINE_MODEL in row["responses"]:
            resp = row["responses"][BASELINE_MODEL]
            score = row["scores"].get(BASELINE_MODEL, 0)
            baseline_cost += estimate_cost(query_text, resp, BASELINE_MODEL)
            baseline_quality += score

        # --- Random routing baseline ---
        random_model = random.choice(available_models)
        resp = row["responses"][random_model]
        score = row["scores"].get(random_model, 0)
        random_cost += estimate_cost(query_text, resp, random_model)
        random_quality += score

    n = len(test_rows)
    router_avg_quality = router_quality / n
    baseline_avg_quality = baseline_quality / n
    random_avg_quality = random_quality / n

    cost_savings_pct = (1 - router_cost / baseline_cost) * 100 if baseline_cost > 0 else 0
    quality_delta_pct = (router_avg_quality - baseline_avg_quality) / baseline_avg_quality * 100

    print("=" * 60)
    print(f"{'Strategy':<25}{'Total Cost ($)':<18}{'Avg Quality':<12}")
    print("=" * 60)
    print(f"{'Router (ours)':<25}{router_cost:<18.6f}{router_avg_quality:<12.2f}")
    print(f"{'Always ' + BASELINE_MODEL:<25}{baseline_cost:<18.6f}{baseline_avg_quality:<12.2f}")
    print(f"{'Random routing':<25}{random_cost:<18.6f}{random_avg_quality:<12.2f}")
    print("=" * 60)
    print(f"\nCost savings vs always using {BASELINE_MODEL}: {cost_savings_pct:.1f}%")
    print(f"Quality change vs always using {BASELINE_MODEL}: {quality_delta_pct:+.1f}%")

    # --- Chart ---
    strategies = ["Router (ours)", f"Always {BASELINE_MODEL}", "Random routing"]
    costs = [router_cost, baseline_cost, random_cost]
    qualities = [router_avg_quality, baseline_avg_quality, random_avg_quality]

    fig, ax1 = plt.subplots(figsize=(7, 5))
    color1 = "tab:blue"
    ax1.set_xlabel("Strategy")
    ax1.set_ylabel("Total Cost ($)", color=color1)
    ax1.bar(strategies, costs, color=color1, alpha=0.6, label="Cost")
    ax1.tick_params(axis="y", labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Avg Quality Score", color=color2)
    ax2.plot(strategies, qualities, color=color2, marker="o", linewidth=2, label="Quality")
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.title(f"Cost vs Quality: Router vs Baselines (n={n} held-out queries)")
    fig.tight_layout()
    os.makedirs("data", exist_ok=True)
    plt.savefig(OUTPUT_CHART, dpi=150)
    print(f"\nChart saved to {OUTPUT_CHART}")


if __name__ == "__main__":
    main()