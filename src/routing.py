"""Routing decision for one message, with three ways to handle conversation context."""
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

CONFIDENCE_THRESHOLD = 0.30
FALLBACK_MODEL = "llama3.1:8b"
MODES = ("none", "sticky", "rewrite")

Predict = Callable[[str], Tuple[str, float, Dict[str, float]]]  # text -> (model, confidence, probabilities)
Rewrite = Callable[[List[dict], str], str]                      # (history, query) -> standalone query


@dataclass
class Decision:
    model: str
    confidence: float
    probabilities: Dict[str, float]
    routing_query: str      # the text the classifier actually saw
    fallback_triggered: bool


def resolve_route(mode: str, query: str, history: List[dict], predict: Predict, rewrite: Rewrite,
                  threshold: float = CONFIDENCE_THRESHOLD, fallback_model: str = FALLBACK_MODEL) -> Decision:
    """
    none:    classify the raw message (what the router did before multi-turn).
    sticky:  classify the raw message; if confidence is low and the chat has a previous turn,
             stay with the previous turn's model instead of the global fallback.
    rewrite: if the chat has history, classify a standalone rewrite of the message.
    The first message of a chat is always classified as-is.
    """
    if mode not in MODES:
        raise ValueError(f"unknown routing mode {mode!r}; expected one of {MODES}")

    routing_query = rewrite(history, query) if (mode == "rewrite" and history) else query
    model, confidence, probabilities = predict(routing_query)

    fallback = confidence < threshold
    if fallback:
        previous = history[-1].get("routed_model") if history else None
        model = previous if (mode == "sticky" and previous) else fallback_model
    return Decision(model, confidence, probabilities, routing_query, fallback)
