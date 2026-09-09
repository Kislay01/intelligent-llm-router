from sentence_transformers import SentenceTransformer

_model = None


def get_embedder() -> SentenceTransformer:
    """Load the embedding model once and reuse it (avoids reloading on every call)."""
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_query(text: str):
    """Convert a text query into a semantic embedding vector."""
    model = get_embedder()
    return model.encode(text)


if __name__ == "__main__":
    # Quick manual test
    vector = embed_query("What is the capital of France?")
    print("Shape:", vector.shape)
    print("First 5 values:", vector[:5])