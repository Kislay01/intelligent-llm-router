FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY static/ ./static/
COPY data/labeled_results.jsonl ./data/labeled_results.jsonl

RUN python src/train_classifier.py

# Chat history lives here; docker-compose mounts a named volume so it survives `down` and rebuilds.
RUN mkdir -p /app/db

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]