import os
import pickle
import sys
import time
from typing import Optional

sys.path.append(os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from embed import embed_query
from ollama_client import query_model
from db import (init_db, log_request, log_learning, get_stats, create_session, session_exists,
                list_sessions, get_session, delete_session)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

MODEL_PATH = "models/classifier.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

CONFIDENCE_THRESHOLD = 0.30
FALLBACK_MODEL = "llama3.1:8b"

app = FastAPI(title="Intelligent LLM Router")

app.mount("/static", StaticFiles(directory="static"), name="static")

init_db()

with open(MODEL_PATH, "rb") as f:
    classifier = pickle.load(f)
with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)


class RouteRequest(BaseModel):
    query: str
    session_id: Optional[int] = None  # omit to start a new chat


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

    embedding = embed_query(request.query)

    probabilities = classifier.predict_proba([embedding])[0]
    predicted_index = probabilities.argmax()
    predicted_model = label_encoder.classes_[predicted_index]
    confidence = float(probabilities[predicted_index])

    all_probs = {
        label_encoder.classes_[i]: float(p)
        for i, p in enumerate(probabilities)
    }

    fallback_triggered = confidence < CONFIDENCE_THRESHOLD
    final_model = FALLBACK_MODEL if fallback_triggered else predicted_model

    start_time = time.perf_counter()
    model_response = query_model(final_model, request.query)
    latency = time.perf_counter() - start_time

    session_id = request.session_id
    if session_id is None:
        session_id = create_session(request.query.strip()[:60] or "New chat")

    request_id = log_request(
        query=request.query,
        routed_model=final_model,
        confidence=confidence,
        fallback_triggered=fallback_triggered,
        latency_seconds=round(latency, 2),
        response=model_response,
        session_id=session_id,
    )

    log_learning(request.query, final_model, confidence, fallback_triggered)

    return RouteResponse(
        query=request.query,
        routed_model=final_model,
        confidence=confidence,
        all_probabilities=all_probs,
        fallback_triggered=fallback_triggered,
        response=model_response,
        latency_seconds=round(latency, 2),
        session_id=session_id,
        request_id=request_id,
    )


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