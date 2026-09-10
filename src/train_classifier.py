import json
import os
import pickle
import sys

sys.path.append(os.path.dirname(__file__))

from embed import embed_query
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

LABELED_FILE = "data/labeled_results.jsonl"
MODEL_OUT = "models/classifier.pkl"
ENCODER_OUT = "models/label_encoder.pkl"


def load_labeled_data():
    rows = []
    with open(LABELED_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main():
    rows = load_labeled_data()
    print(f"Loaded {len(rows)} labeled examples.")

    print("Computing embeddings for all queries (this may take a minute)...")
    X = [embed_query(r["query"]) for r in rows]
    y_raw = [r["winner"] for r in rows]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"Classes found: {list(le.classes_)}")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError as e:
        print(f"Stratified split failed ({e}), falling back to plain random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nHeld-out test accuracy: {acc:.3f}")
    print("\nClassification report:")
    print(classification_report(
        y_test, preds,
        labels=range(len(le.classes_)),
        target_names=le.classes_,
        zero_division=0
    ))

    os.makedirs("models", exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(clf, f)
    with open(ENCODER_OUT, "wb") as f:
        pickle.dump(le, f)

    print(f"Saved classifier to {MODEL_OUT}")
    print(f"Saved label encoder to {ENCODER_OUT}")


if __name__ == "__main__":
    main()