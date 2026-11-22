"""Run with python -m ticket_router.train; test data never selects the threshold."""

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from ticket_router.data import SOURCE_COMMIT, SOURCE_SHA256, download_dataset, prepare_data, split_data

# Require majority probability for auto-routing, even on an easy validation set.
THRESHOLDS = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99, 1.0)


def routing_metrics(y_true, probabilities, classes, threshold: float) -> dict:
    predictions = classes[np.argmax(probabilities, axis=1)]
    mask = (np.max(probabilities, axis=1) >= threshold) & (threshold < 1.0)
    count = int(mask.sum())
    return {
        "threshold": float(threshold), "auto_routed": count,
        "coverage": float(mask.mean()),
        "accuracy": float(np.mean(predictions[mask] == np.asarray(y_true)[mask])) if count else None,
    }


def select_threshold(rows: list[dict], target_accuracy: float) -> dict:
    eligible = [r for r in rows if r["auto_routed"] > 0 and r["accuracy"] >= target_accuracy]
    if eligible:
        return max(eligible, key=lambda r: (r["coverage"], r["accuracy"], r["threshold"]))
    # A target that cannot be met must not silently enable lower quality routing.
    return {"threshold": 1.0, "auto_routed": 0, "coverage": 0.0, "accuracy": None}


def evaluate(y_true, predictions, labels) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro")),
        "confusion_matrix": confusion_matrix(y_true, predictions, labels=labels).tolist(),
        "classification_report": classification_report(
            y_true, predictions, labels=labels, output_dict=True, zero_division=0
        ),
    }


def train(data_path: Path, artifact_path: Path, report_dir: Path, target_accuracy: float = 0.98):
    download_dataset(data_path)
    clean, stats = prepare_data(pd.read_csv(data_path))
    training, validation, test = split_data(clean)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("classifier", LogisticRegression(C=2.0, max_iter=1000, random_state=42)),
    ])
    pipeline.fit(training["text"], training["category"])
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(training["text"], training["category"])
    labels = pipeline.classes_
    validation_probabilities = pipeline.predict_proba(validation["text"])
    threshold_rows = [routing_metrics(validation["category"], validation_probabilities, labels, t)
                      for t in THRESHOLDS]
    chosen = select_threshold(threshold_rows, target_accuracy)
    threshold = chosen["threshold"]
    validation_mistakes = []
    validation_predictions = labels[np.argmax(validation_probabilities, axis=1)]
    for (_, row), prediction, probabilities in zip(
        validation.iterrows(), validation_predictions, validation_probabilities
    ):
        if prediction != row["category"]:
            confidence = float(max(probabilities))
            validation_mistakes.append({
                "text": row["text"], "actual": row["category"], "predicted": str(prediction),
                "confidence": confidence,
                "routing_decision": "auto_route" if threshold < 1 and confidence >= threshold else "manual_review",
            })
    test_probabilities = pipeline.predict_proba(test["text"])
    predictions = labels[np.argmax(test_probabilities, axis=1)]
    mistakes = []
    for (_, row), prediction, probabilities in zip(test.iterrows(), predictions, test_probabilities):
        if prediction != row["category"]:
            confidence = float(max(probabilities))
            mistakes.append({
                "text": row["text"], "actual": row["category"], "predicted": str(prediction),
                "confidence": confidence,
                "routing_decision": "auto_route" if threshold < 1 and confidence >= threshold else "manual_review",
            })
    split_manifest = {}
    for name, frame in (("train", training), ("validation", validation), ("test", test)):
        split_manifest[name] = [hashlib.sha256(key.encode()).hexdigest() for key in frame["key"]]
    report = {
        "source_commit": SOURCE_COMMIT, "source_sha256": SOURCE_SHA256, "seed": 42,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "data": stats,
        "splits": {name: {"rows": len(frame), "class_counts": frame["category"].value_counts().sort_index().to_dict()}
                   for name, frame in (("train", training), ("validation", validation), ("test", test))},
        "labels": labels.tolist(),
        "threshold_policy": {"target_validation_accuracy": target_accuracy,
                             "minimum_threshold": 0.5,
                             "rule": "Maximize coverage among candidates meeting target; otherwise manual review all"},
        "validation_thresholds": threshold_rows, "chosen_validation_routing": chosen,
        "baseline_test": evaluate(test["category"], baseline.predict(test["text"]), labels),
        "model_test": evaluate(test["category"], predictions, labels),
        "test_routing": routing_metrics(test["category"], test_probabilities, labels, threshold),
        "mistakes": mistakes[:10],
        "validation_mistakes": validation_mistakes[:10],
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"schema_version": 1, "pipeline": pipeline, "threshold": threshold,
                 "source_sha256": SOURCE_SHA256, "sklearn_version": sklearn.__version__}, artifact_path)
    (report_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (report_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")
    pd.DataFrame(threshold_rows).to_csv(report_dir / "validation_thresholds.csv", index=False)
    pd.DataFrame(report["model_test"]["confusion_matrix"], index=labels, columns=labels).to_csv(
        report_dir / "confusion_matrix.csv", index_label="actual / predicted"
    )
    print(json.dumps({"artifact": str(artifact_path), "data": stats,
                      "baseline_macro_f1": report["baseline_test"]["macro_f1"],
                      "model_macro_f1": report["model_test"]["macro_f1"],
                      "test_routing": report["test_routing"]}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/bitext.csv"))
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--target-accuracy", type=float, default=0.98)
    args = parser.parse_args()
    if not 0 < args.target_accuracy <= 1:
        parser.error("--target-accuracy must be in (0, 1]")
    train(args.data, args.artifact, args.reports, args.target_accuracy)


if __name__ == "__main__":
    main()
