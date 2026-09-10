import json
import os
import re
import sys
import time

sys.path.append(os.path.dirname(__file__))

from ollama_client import query_model
from gemini_client import ask_gemini

QUERIES_FILE = "data/queries.json"
RESULTS_FILE = "data/labeled_results.jsonl"
BATCH_SIZE = 24

OLLAMA_MODELS = ["llama3.2:3b", "llama3.1:8b", "qwen2.5-coder:7b", "mistral:7b"]

JUDGE_PROMPT_TEMPLATE = """You are evaluating 4 AI model responses to the same user query.
Rate each response from 1 to 10 based on accuracy, helpfulness, and relevance to the query.

Query: {query}

Response from {m0}:
{r0}

Response from {m1}:
{r1}

Response from {m2}:
{r2}

Response from {m3}:
{r3}

Respond with ONLY valid JSON, no markdown formatting, in exactly this format:
{{"scores": {{"{m0}": <score>, "{m1}": <score>, "{m2}": <score>, "{m3}": <score>}}, "winner": "<model name with the highest score>"}}
"""


def load_queries():
    with open(QUERIES_FILE, encoding="utf-8") as f:
        return json.load(f)


def load_done_ids():
    if not os.path.exists(RESULTS_FILE):
        return set()
    done = set()
    with open(RESULTS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add(row["id"])
    return done


def get_model_responses(query_text):
    responses = {}
    for model in OLLAMA_MODELS:
        try:
            responses[model] = query_model(model, query_text)
        except Exception as e:
            responses[model] = f"[ERROR: {e}]"
    return responses


def parse_judge_response(raw_text):
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    return json.loads(cleaned)


def label_one_query(item):
    query_text = item["query"]
    responses = get_model_responses(query_text)

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        query=query_text,
        m0=OLLAMA_MODELS[0], r0=responses[OLLAMA_MODELS[0]],
        m1=OLLAMA_MODELS[1], r1=responses[OLLAMA_MODELS[1]],
        m2=OLLAMA_MODELS[2], r2=responses[OLLAMA_MODELS[2]],
        m3=OLLAMA_MODELS[3], r3=responses[OLLAMA_MODELS[3]],
    )

    judge_raw = None
    for attempt in range(3):
        try:
            judge_raw = ask_gemini(prompt)
            break
        except Exception as e:
            if attempt < 2:
                wait = 10 * (attempt + 1)  
                print(f"  Gemini error ({e}), retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    judgment = parse_judge_response(judge_raw)

    result = {
        "id": item["id"],
        "category": item["category"],
        "subtopic": item["subtopic"],
        "query": query_text,
        "responses": responses,
        "scores": judgment["scores"],
        "winner": judgment["winner"],
    }
    return result


def main():
    all_queries = load_queries()
    done_ids = load_done_ids()
    remaining = [q for q in all_queries if q["id"] not in done_ids]

    print(f"Total queries: {len(all_queries)}")
    print(f"Already labeled: {len(done_ids)}")
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("All queries already labeled. Nothing to do.")
        return

    batch = remaining[:BATCH_SIZE]
    print(f"Processing this batch: {len(batch)} queries\n")

    with open(RESULTS_FILE, "a", encoding="utf-8") as out_f:
        for i, item in enumerate(batch, start=1):
            print(f"[{i}/{len(batch)}] Query {item['id']} ({item['category']}/{item['subtopic']}): {item['query'][:60]}...")
            try:
                result = label_one_query(item)
            except Exception as e:
                print(f"  -> FAILED: {e}. Skipping, will retry on next run.")
                continue

            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"  -> Winner: {result['winner']} | Scores: {result['scores']}")

            time.sleep(4)  

    new_done = len(done_ids) + len(batch)
    print(f"\nBatch complete. Total labeled so far: {new_done}/{len(all_queries)}")
    print("Run this script again to process the next batch.")


if __name__ == "__main__":
    main()