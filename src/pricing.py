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
