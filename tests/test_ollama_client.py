import os
import sys
from unittest.mock import MagicMock, patch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from ollama_client import query_model


@patch("ollama_client.requests.post")
def test_query_model_returns_response_text(mock_post):
    mock_response = MagicMock()
    mock_response.json.return_value = {"response": "Paris"}
    mock_response.raise_for_status.return_value = None
    mock_post.return_value = mock_response

    result = query_model("llama3.2:3b", "What is the capital of France?")

    assert result == "Paris"
    mock_post.assert_called_once()