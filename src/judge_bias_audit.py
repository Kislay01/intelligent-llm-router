import json
import random
import re
import sys
import os
import time

sys.path.append(os.path.dirname(__file__))
from gemini_client import ask_gemini

LABELED_FILE = "data/labeled_results.jsonl"
SAMPLE_SIZE = 18

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


def parse_judge_response(raw_text):
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    return json.loads(cleaned)


def load_rows():
    rows = []
    with open(LABELED_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    rows = load_rows()
    random.seed(7)
    sample = random.sample(rows, min(SAMPLE_SIZE, len(rows)))

    matches, mismatches = 0, 0

    for row in sample:
        models = list(row["responses"].keys())
        shuffled = models[:]
        random.shuffle(shuffled)

        prompt = JUDGE_PROMPT_TEMPLATE.format(
            query=row["query"],
            m0=shuffled[0], r0=row["responses"][shuffled[0]],
            m1=shuffled[1], r1=row["responses"][shuffled[1]],
            m2=shuffled[2], r2=row["responses"][shuffled[2]],
            m3=shuffled[3], r3=row["responses"][shuffled[3]],
        )

        try:
            raw = ask_gemini(prompt)
            judgment = parse_judge_response(raw)
            new_winner = judgment["winner"]
        except Exception as e:
            print(f"Query {row['id']}: FAILED ({e}), skipping")
            continue

        original_winner = row["winner"]
        same = new_winner == original_winner
        matches += same
        mismatches += not same

        print(f"Query {row['id']} | order={shuffled} | "
              f"original={original_winner} | reshuffled={new_winner} | "
              f"{'MATCH' if same else 'CHANGED'}")

        time.sleep(4)

    total = matches + mismatches
    print("\n" + "=" * 50)
    print(f"Total tested: {total}")
    print(f"Consistent (no positional bias detected): {matches} ({matches/total*100:.1f}%)")
    print(f"Changed (possible positional bias): {mismatches} ({mismatches/total*100:.1f}%)")


if __name__ == "__main__":
    main()