from pricing import MODEL_PRICING_PER_1M_TOKENS, SERVED_MODELS

# A model counts as "good enough" if its judge score is within this many points of the
# best score for the query. 1 on a 1-10 scale absorbs judge noise (ties and 9-vs-10 splits).
DEFAULT_TOLERANCE = 1


def cost_aware_label(scores: dict, tolerance: int = DEFAULT_TOLERANCE) -> str:
    """Cheapest served model whose score is within `tolerance` of the best served score.

    Ties on price go to the higher score, then to the name (deterministic).
    Raises ValueError if no served model has a score.
    """
    served = {m: s for m, s in scores.items() if m in SERVED_MODELS}
    if not served:
        raise ValueError("no served model has a score")
    best = max(served.values())
    candidates = [m for m, s in served.items() if s >= best - tolerance]
    return min(
        candidates,
        key=lambda m: (MODEL_PRICING_PER_1M_TOKENS[m], -served[m], m),
    )


def get_label(row: dict, tolerance: int | None = DEFAULT_TOLERANCE) -> str:
    """Training label for one row. tolerance=None keeps the original judge winner."""
    if tolerance is None:
        return row["winner"]
    return cost_aware_label(row["scores"], tolerance)
