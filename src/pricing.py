# Single source of truth for model pricing and which models the router may serve.
# Prices: Together AI, USD per 1M tokens. codellama:7b is not listed there, so its
# value is a placeholder; it is also not in SERVED_MODELS, so it never affects routing.
MODEL_PRICING_PER_1M_TOKENS = {
    "llama3.2:3b": 0.06,
    "llama3.1:8b": 0.18,
    "mistral:7b": 0.06,
    "qwen2.5-coder:7b": 0.07,
    "codellama:7b": 0.06,
}

# Models that are actually pulled into Ollama (see README). Labels must come from this set,
# otherwise the router could pick a model that is not installed.
SERVED_MODELS = ["llama3.2:3b", "llama3.1:8b", "qwen2.5-coder:7b", "mistral:7b"]


def estimate_tokens(text: str) -> float:
    """Rough heuristic: ~4 characters per token for English text."""
    return len(text) / 4


def estimate_cost(query_text: str, response_text: str, model: str) -> float:
    """USD cost of one query/response pair under the Together AI price proxy."""
    price_per_1k = MODEL_PRICING_PER_1M_TOKENS[model] / 1000
    return (estimate_tokens(query_text) + estimate_tokens(response_text)) / 1000 * price_per_1k
