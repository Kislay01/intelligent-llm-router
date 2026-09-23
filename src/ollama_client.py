import os
import requests

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_URL = f"{OLLAMA_HOST}/api/generate"


def query_model(model: str, prompt: str) -> str:
    """Send a prompt to a local Ollama model and return its text response."""
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()["response"]


if __name__ == "__main__":
    result = query_model("llama3.2:3b", "What is 2 + 2?")
    print(result)