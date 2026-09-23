import os
import sys
from unittest.mock import patch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from fastapi.testclient import TestClient

import api

client = TestClient(api.app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("api.query_model")
def test_route_endpoint(mock_query_model):
    mock_query_model.return_value = "mocked response text"

    response = client.post("/route", json={"query": "What is the capital of France?"})

    assert response.status_code == 200
    data = response.json()
    assert "routed_model" in data
    assert data["response"] == "mocked response text"
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["fallback_triggered"], bool)