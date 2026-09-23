# Intelligent LLM Router

A cost-aware LLM query router that learns — from data, not hardcoded rules — which locally-hosted open-source model should answer a given query. Instead of sending every request to the most powerful (and most expensive) model, the router uses sentence embeddings and a trained classifier to pick the most cost-appropriate model per query, backed by a one-time offline labeling pipeline using an LLM-as-judge.

## Problem

Most LLM deployments route every query to the same model regardless of complexity — wasting compute and cost on tasks a smaller model could handle equally well. This project demonstrates a learned alternative: a classifier trained on real outcome data (not category-based rules) decides which model answers each query.

## Architecture

```
User Query
    │
    ▼
Sentence-Transformer Embedding (all-MiniLM-L6-v2, 384-dim)
    │
    ▼
Trained Classifier (Logistic Regression, class_weight=balanced)
    │
    ├── confidence ≥ threshold → route to predicted model
    └── confidence < threshold → fallback to strongest general model
    │
    ▼
Selected Ollama Model generates response
    │
    ▼
Response + routing metadata returned, logged to SQLite
```

## Features

- **Learned routing** — no hardcoded category-to-model rules; a classifier trained on real outcome data (query → best-performing model) makes the decision
- **LLM-as-judge labeling pipeline** — Gemini scores 4-5 candidate model responses per training query to generate labels
- **Confidence-threshold fallback** — low-confidence predictions escalate to a safe default model instead of guessing
- **Cost-vs-quality evaluation** — quantified comparison against an always-use-the-strongest-model baseline, using real commercial pricing as a cost proxy
- **Judge-order bias audit** — tested the labeling pipeline for LLM-judge positional bias
- **UMAP embedding visualization** — 2D projection showing query clusters by category and by winning model
- **Full test suite + CI/CD** — pytest tests (mocked external calls) run automatically via GitHub Actions on every push
- **Chat UI + live dashboard** — single-page frontend for querying the router and viewing usage stats
- **Request logging** — every routed query logged to SQLite (model, confidence, fallback status, latency)
- **Dockerized** — fully containerized via Docker Compose (app + Ollama), runs identically on any machine

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI + Uvicorn |
| Query embedding | sentence-transformers (all-MiniLM-L6-v2) |
| Classifier | scikit-learn Logistic Regression |
| Model backends | Ollama (llama3.2:3b, llama3.1:8b, mistral:7b, qwen2.5-coder:7b) |
| Labeling / judging | Google Gemini API (gemini-3.5-flash-lite) |
| Storage | SQLite (request logs) |
| Frontend | Vanilla HTML/CSS/JS |
| Testing | pytest, GitHub Actions |
| Containerization | Docker, Docker Compose |
| Visualization | matplotlib, UMAP |

## Project Structure

```
intelligent-llm-router/
├── src/
│   ├── ollama_client.py          # Wraps Ollama API calls
│   ├── embed.py                   # Sentence-transformer embedding wrapper
│   ├── gemini_client.py           # Gemini API wrapper (LLM-as-judge)
│   ├── label_pipeline.py          # Offline labeling: query → winning model
│   ├── judge_bias_audit.py        # Tests labeling pipeline for positional bias
│   ├── train_classifier.py        # Trains + evaluates the routing classifier
│   ├── cost_quality_analysis.py   # Router vs baseline cost/quality comparison
│   ├── embedding_visualization.py # UMAP cluster visualization
│   ├── db.py                      # SQLite request logging
│   └── api.py                     # FastAPI app (routing, stats, UI serving)
├── static/
│   └── index.html                 # Chat UI + dashboard frontend
├── tests/
│   ├── test_embed.py
│   ├── test_ollama_client.py
│   └── test_api.py
├── data/
│   ├── build_queries.py           # Generates the 150-query synthetic dataset
│   ├── add_supplementary_queries.py
│   ├── queries.json                # 174 synthetic queries (4 categories)
│   └── labeled_results.jsonl       # Labeled training data (query, responses, scores, winner)
├── models/                         # Trained classifier + label encoder (generated, not tracked)
├── .github/workflows/ci.yml        # CI: install deps → train → test
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Setup

### Option A: Docker (recommended for demo)

Prerequisites: Docker Desktop, no local Python setup needed.

```bash
docker-compose up -d --build
```

Pull the required models into the Ollama container (one-time):

```bash
docker exec -it intelligent-llm-router-ollama-1 ollama pull llama3.2:3b
docker exec -it intelligent-llm-router-ollama-1 ollama pull llama3.1:8b
docker exec -it intelligent-llm-router-ollama-1 ollama pull mistral:7b
docker exec -it intelligent-llm-router-ollama-1 ollama pull qwen2.5-coder:7b
```

Open `http://localhost:8000/`.

To stop: `docker-compose down`. To restart later: `docker-compose up -d` (no rebuild needed).

### Option B: Local development

Prerequisites: Python 3.11+, [Ollama](https://ollama.com) installed locally.

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

ollama pull llama3.2:3b
ollama pull llama3.1:8b
ollama pull mistral:7b
ollama pull qwen2.5-coder:7b

python src/train_classifier.py  # generates models/classifier.pkl
uvicorn src.api:app --reload
```

Open `http://127.0.0.1:8000/`.

### Rebuilding the dataset/classifier from scratch (optional)

Requires a Gemini API key in `.env` as `GEMINI_API_KEY=...`.

```bash
python data/build_queries.py           # generates 150 synthetic queries
python src/label_pipeline.py           # labels queries via Ollama + Gemini judge (run repeatedly, processes in batches)
python src/train_classifier.py         # trains classifier, prints held-out test accuracy
python src/cost_quality_analysis.py    # cost vs quality evaluation + chart
python src/embedding_visualization.py  # UMAP cluster chart
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Chat UI + dashboard |
| `/health` | GET | Health check |
| `/route` | POST | Routes a query, returns chosen model, confidence, response, latency |
| `/stats` | GET | Aggregated usage stats (model distribution, recent requests) |

Example:

```bash
curl -X POST http://localhost:8000/route \
  -H "Content-Type: application/json" \
  -d '{"query": "Write a Python function to reverse a linked list"}'
```

## Testing

```bash
pytest tests/ -v
```

Runs automatically via GitHub Actions on every push (`.github/workflows/ci.yml`) — trains the classifier fresh from committed labeled data, then runs the full suite. Ollama and Gemini calls are mocked in tests, so no live services are required for CI.

## Evaluation Results

- **Classifier**: trained on 174 labeled examples (150 synthetic + 24 supplementary coding queries), 80/20 train/test split, held-out test accuracy 57.1%, macro F1 0.43 with `class_weight="balanced"` (vs 0.27 macro F1 unweighted — a deliberate trade-off favoring fair treatment of minority classes over raw accuracy)
- **Cost vs quality**: router achieved a **43.0% cost reduction** for a **2.9% quality drop** versus always using the strongest model, evaluated on 35 held-out test queries (see `data/cost_quality_chart.png`)
- **Judge-order bias audit**: 77.8% label consistency (14/18) under model-order randomization; all inconsistencies were isolated to one model's marginal wins, not a systemic ordering bias (see `src/judge_bias_audit.py`)
- **Embedding visualization**: query categories separate cleanly in embedding space (UMAP); winning-model distribution does not cleanly align with category clusters, confirming routing is a distinct sub-problem from topic classification (see `data/embedding_clusters.png`)

## Limitations

- **Dataset size**: 174 labeled examples is small by ML standards; one class (`codellama:7b`) has only 2 real examples. Sufficient to demonstrate the routing mechanism; a production system would train on real query logs at much larger scale.
- **Local models, not commercial APIs**: uses Ollama-hosted open models for zero-cost demonstration. The routing architecture (embedding → classifier → model call) transfers directly to commercial APIs (OpenAI, Anthropic, Azure); only the backend calls would change.
- **Single-turn only**: the router considers each query in isolation, with no conversation history.
- **No content moderation layer**: a production router would need safety checks before dispatching to a model.
- **Synchronous request handling**: no async queuing (e.g., Celery/Redis) for concurrent load at scale.
- **Single LLM judge, no inter-rater reliability check**: labels come from one Gemini pass per query, with a partial bias audit rather than full multi-judge validation.
- **"Zero cost" refers to marginal API cost only** — local compute/hardware costs are not zero in a real deployment.

## Future Work

- Scale the labeling pipeline to real production query logs
- Add async request handling for concurrent traffic
- Multi-turn context awareness
- Periodic (batch, not continuous) retraining on newly logged queries
- Safety/content-moderation pre-filtering

## References

1. Chen, L., Zaharia, M., & Zou, J. (2023). FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. arXiv:2305.05176.
2. Ding, D., Mallick, A., Zhang, S., Wang, C., et al. (2025). BEST-Route: Adaptive LLM Routing with Test-Time Optimal Compute. arXiv:2506.22716.
3. Zhao, Z., Jin, S., & Mao, Z. M. (2024). Eagle: Efficient Training-Free Router for Multi-LLM Inference. arXiv:2409.15518.
4. Ong, I., Almahairi, A., Wu, V., Chiang, W.-L., Wu, T., Gonzalez, J. E., Kadous, M. W., & Stoica, I. (2024). RouteLLM: Learning to Route LLMs with Preference Data. arXiv:2406.18665.
5. Ding, D., Mallick, A., Wang, C., Sim, R., Mukherjee, S., Rühle, V., Lakshmanan, L. V. S., & Awadallah, A. H. (2024). Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing. ICLR 2024.
6. Hu, Q. J., Bieker, J., Li, X., Jiang, N., Keigwin, B., Ranganath, G., Keutzer, K., & Upadhyay, S. K. (2024). RouterBench: A Benchmark for Multi-LLM Routing System. arXiv:2403.12031.

## Author

Kislay Upadhyay — BTech CSE (AI), Vishwakarma Institute of Technology, Pune