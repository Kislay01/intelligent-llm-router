import os
import pickle
import sys
import time
from typing import Optional

sys.path.append(os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from embed import embed_query
from ollama_client import query_model, query_chat
from context import build_chat_messages, rewrite_query
from routing import CONFIDENCE_THRESHOLD, FALLBACK_MODEL, MODES, resolve_route
from db import (init_db, log_request, log_learning, get_stats, create_session, session_exists,
                list_sessions, get_session, delete_session)
from v2_serve import DEFAULT_COST_WEIGHT, lam_from_cost_weight, load_curve, load_router
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Chosen from src/compare_followup_routing.py (24 follow-ups): the strategies were statistically
# indistinguishable, "rewrite" added ~5s per follow-up and sometimes answered or embellished
# instead of rewriting, so the free strategy wins.
DEFAULT_ROUTING_MODE = "sticky"

MODEL_PATH = "models/classifier.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

app = FastAPI(title="Intelligent LLM Router")

app.mount("/static", StaticFiles(directory="static"), name="static")

init_db()

with open(MODEL_PATH, "rb") as f:
    classifier = pickle.load(f)
with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)


# ROUTER_VERSION=v1 keeps the original single-label classifier (rollback switch). v2 needs models/router_v2.pkl
# (python src/v2_train.py); without it the app falls back to v1.
ROUTER_VERSION = os.getenv("ROUTER_VERSION", "v2")
v2 = load_router() if ROUTER_VERSION == "v2" else None
if ROUTER_VERSION == "v2" and v2 is None:
    print("WARNING: models/router_v2.pkl not found, serving the v1 router. Run: python src/v2_train.py")


def make_v2_predict(lam: float, seen: dict):
    """predict function for resolve_route: remembers the embedding so the explanation reuses it."""
    def predict(text: str):
        seen["vec"] = embed_query(text)
        return v2.predict(seen["vec"], lam)
    return predict


def predict_text(text: str):
    """Classifier output for one piece of text: (model, confidence, probabilities)."""
    probabilities = classifier.predict_proba([embed_query(text)])[0]
    index = int(probabilities.argmax())
    labels = [str(c) for c in label_encoder.classes_]
    return labels[index], float(probabilities[index]), {l: float(p) for l, p in zip(labels, probabilities)}


class RouteRequest(BaseModel):
    query: str
    session_id: Optional[int] = None  # omit to start a new chat
    # v2 only: 0 = best quality, 1 = cheapest. Omit for the default.
    cost_weight: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class RouteResponse(BaseModel):
    query: str
    routed_model: str
    confidence: float
    all_probabilities: dict
    fallback_triggered: bool
    response: str
    latency_seconds: float
    session_id: int
    request_id: int
    routing_query: str
    router_version: str = "v1"
    cost_weight: Optional[float] = None
    explanation: Optional[list] = None  # v2: the most similar training queries and which models passed them


@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/route", response_model=RouteResponse)
def route_query(request: RouteRequest):
    if request.session_id is not None and not session_exists(request.session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    history = []
    if request.session_id is not None:
        history = get_session(request.session_id)["messages"]

    mode = os.getenv("ROUTING_CONTEXT_MODE", DEFAULT_ROUTING_MODE)
    if mode not in MODES:
        mode = DEFAULT_ROUTING_MODE
    seen: dict = {}
    if v2 is not None:
        cost_weight = DEFAULT_COST_WEIGHT if request.cost_weight is None else request.cost_weight
        predict = make_v2_predict(lam_from_cost_weight(cost_weight), seen)
    else:
        cost_weight, predict = None, predict_text
    decision = resolve_route(mode, request.query, history, predict, rewrite_query)
    explanation = v2.explain(seen["vec"]) if v2 is not None and "vec" in seen else None
    final_model = decision.model

    start_time = time.perf_counter()
    if history:  # the answering model sees the recent conversation
        model_response = query_chat(final_model, build_chat_messages(history, request.query))
    else:
        model_response = query_model(final_model, request.query)
    latency = time.perf_counter() - start_time

    session_id = request.session_id
    if session_id is None:
        session_id = create_session(request.query.strip()[:60] or "New chat")

    request_id = log_request(
        query=request.query,
        routed_model=final_model,
        confidence=decision.confidence,
        fallback_triggered=decision.fallback_triggered,
        latency_seconds=round(latency, 2),
        response=model_response,
        session_id=session_id,
        routing_query=decision.routing_query,
        why={"probabilities": decision.probabilities, "neighbours": explanation} if explanation else None,
    )

    # Only standalone text is useful for learning: a first message, or a follow-up that was rewritten.
    # A bare "now in Java" would be a misleading training query.
    if not history or decision.routing_query != request.query:
        log_learning(decision.routing_query, final_model, decision.confidence, decision.fallback_triggered)

    return RouteResponse(
        query=request.query,
        routed_model=final_model,
        confidence=decision.confidence,
        all_probabilities=decision.probabilities,
        fallback_triggered=decision.fallback_triggered,
        response=model_response,
        latency_seconds=round(latency, 2),
        session_id=session_id,
        request_id=request_id,
        routing_query=decision.routing_query,
        router_version="v2" if v2 is not None else "v1",
        cost_weight=cost_weight,
        explanation=explanation,
    )


@app.get("/dial-curve")
def dial_curve():
    """Measured quality/cost curve for the slider (empty if data/v2_results.json is not shipped)."""
    curve = load_curve() if v2 is not None else {}
    return {"router_version": "v2" if v2 is not None else "v1", "default_cost_weight": DEFAULT_COST_WEIGHT,
            "points": curve.get("points", []), "always_8b_judge": curve.get("always_8b_judge")}


@app.get("/sessions")
def sessions():
    return list_sessions()


@app.get("/sessions/{session_id}")
def session_detail(session_id: int):
    session = get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@app.delete("/sessions/{session_id}")
def session_delete(session_id: int):
    if not delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": session_id}


@app.get("/stats")
def stats():
    return get_stats()