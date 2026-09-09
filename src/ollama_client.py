import requests

OLLAMA_URL = "http://localhost:11434/api/generate"


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
    # Quick manual test
    result = query_model("llama3.2:3b", "What is 2 + 2?")
    print(result)