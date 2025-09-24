#!/usr/bin/env python3
"""Train an emotion classifier on cached WavLM features.

The script expects feature files produced by extract_features.py with
`emotion` labels in the index. It trains a multinomial logistic
regression on L10 embeddings (after standardisation) and saves a
joblib bundle for runtime use.

Usage:
    python scripts/train_emotion_probe.py \
        --index artifacts/features/features_index.json \
        --output artifacts/probes/emotion_probe.joblib
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("train_emotion_probe")


def load_index(path: Path) -> List[dict]:
    return json.loads(path.read_text())


def load_dataset(index: List[dict]) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    features = []
    labels = []
    for entry in index:
        label = entry.get("emotion")
        if not label:
            continue
        feat_path = Path(entry["feature_path"])
        if not feat_path.exists():
            logger.warning("Missing feature file %s", feat_path)
            continue
        blob = np.load(feat_path)
        l10 = blob["l10"].astype(np.float32)
        features.append(l10)
        labels.append(label)
    if not features:
        raise ValueError("No emotion-labelled samples found.")
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    return np.stack(features), y, encoder


def train_probe(
    features: np.ndarray,
    labels: np.ndarray,
    penalty: str,
    c: float,
    max_iter: int,
) -> Dict[str, object]:
    scaler = StandardScaler().fit(features)
    feats_z = scaler.transform(features)

    clf = LogisticRegression(
        penalty=penalty,
        C=c,
        max_iter=max_iter,
        solver="lbfgs",
        multi_class="auto",
    )
    clf.fit(feats_z, labels)

    return {
        "scaler": scaler,
        "classifier": clf,
    }


def evaluate(model_bundle: Dict[str, object], X: np.ndarray, y: np.ndarray, encoder: LabelEncoder) -> None:
    scaler: StandardScaler = model_bundle["scaler"]
    clf: LogisticRegression = model_bundle["classifier"]
    preds = clf.predict(scaler.transform(X))
    report = classification_report(y, preds, target_names=encoder.classes_)
    logger.info("Validation report\n%s", report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train emotion classifier on WavLM features")
    parser.add_argument("--index", required=True, type=Path, help="features_index.json")
    parser.add_argument("--output", required=True, type=Path, help="Output joblib file")
    parser.add_argument("--penalty", default="l2", choices=["l2", "none"], help="Logistic regression penalty")
    parser.add_argument("--c", type=float, default=1.0, help="Inverse regularisation strength (C)")
    parser.add_argument("--max-iter", type=int, default=200, help="Max iterations for logistic regression")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    index = load_index(args.index)
    X, y, encoder = load_dataset(index)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    model_bundle = train_probe(X_train, y_train, penalty=args.penalty, c=args.c, max_iter=args.max_iter)
    evaluate(model_bundle, X_val, y_val, encoder)

    payload = {
        "scaler": model_bundle["scaler"],
        "classifier": model_bundle["classifier"],
        "labels": encoder.classes_,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, args.output)
    logger.info("Saved emotion probe to %s", args.output)


if __name__ == "__main__":
    main()
