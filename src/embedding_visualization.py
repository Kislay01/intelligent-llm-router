import json
import os
import sys

sys.path.append(os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP

from embed import embed_query

LABELED_FILE = "data/labeled_results.jsonl"
OUTPUT_CHART = "data/embedding_clusters.png"


def load_rows():
    rows = []
    with open(LABELED_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    rows = load_rows()
    print(f"Loaded {len(rows)} queries. Computing embeddings...")

    embeddings = np.array([embed_query(r["query"]) for r in rows])
    categories = [r["category"] for r in rows]
    winners = [r["winner"] for r in rows]

    print("Running UMAP (reducing 384D -> 2D)...")
    reducer = UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
    coords_2d = reducer.fit_transform(embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    unique_categories = sorted(set(categories))
    cmap1 = plt.get_cmap("tab10")
    for i, cat in enumerate(unique_categories):
        mask = [c == cat for c in categories]
        axes[0].scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            label=cat, color=cmap1(i), alpha=0.7, s=40
        )
    axes[0].set_title("Query Embeddings Colored by Category")
    axes[0].legend()
    axes[0].set_xlabel("UMAP Dimension 1")
    axes[0].set_ylabel("UMAP Dimension 2")

    unique_winners = sorted(set(winners))
    cmap2 = plt.get_cmap("Set2")
    for i, model in enumerate(unique_winners):
        mask = [w == model for w in winners]
        axes[1].scatter(
            coords_2d[mask, 0], coords_2d[mask, 1],
            label=model, color=cmap2(i), alpha=0.7, s=40
        )
    axes[1].set_title("Query Embeddings Colored by Winning Model")
    axes[1].legend()
    axes[1].set_xlabel("UMAP Dimension 1")
    axes[1].set_ylabel("UMAP Dimension 2")

    plt.tight_layout()
    os.makedirs("data", exist_ok=True)
    plt.savefig(OUTPUT_CHART, dpi=150)
    print(f"Saved chart to {OUTPUT_CHART}")


if __name__ == "__main__":
    main()