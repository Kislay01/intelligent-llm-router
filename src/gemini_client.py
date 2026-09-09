import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        _client = genai.Client(api_key=api_key)
    return _client


def ask_gemini(prompt: str) -> str:
    """Send a prompt to Gemini and return its text response."""
    client = get_client()
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )
    return response.text


if __name__ == "__main__":
    result = ask_gemini("Say hello in exactly 5 words.")
    print(result)