import os
import pickle
import sys
import time

sys.path.append(os.path.dirname(__file__))

from fastapi import FastAPI
from pydantic import BaseModel

from embed import embed_query
from ollama_client import query_model

MODEL_PATH = "models/classifier.pkl"
ENCODER_PATH = "models/label_encoder.pkl"

app = FastAPI(title="Intelligent LLM Router")

with open(MODEL_PATH, "rb") as f:
    classifier = pickle.load(f)
with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)


class RouteRequest(BaseModel):
    query: str


class RouteResponse(BaseModel):
    query: str
    routed_model: str
    confidence: float
    all_probabilities: dict
    response: str
    latency_seconds: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/route", response_model=RouteResponse)
def route_query(request: RouteRequest):
    embedding = embed_query(request.query)

    probabilities = classifier.predict_proba([embedding])[0]
    predicted_index = probabilities.argmax()
    predicted_model = label_encoder.classes_[predicted_index]
    confidence = float(probabilities[predicted_index])

    all_probs = {
        label_encoder.classes_[i]: float(p)
        for i, p in enumerate(probabilities)
    }

    start_time = time.perf_counter()
    model_response = query_model(predicted_model, request.query)
    latency = time.perf_counter() - start_time

    return RouteResponse(
        query=request.query,
        routed_model=predicted_model,
        confidence=confidence,
        all_probabilities=all_probs,
        response=model_response,
        latency_seconds=round(latency, 2)
    )