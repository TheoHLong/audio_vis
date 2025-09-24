#!/usr/bin/env python3
"""Train a speaker-aware projection for WavLM features.

Uses speaker labels to learn a discriminative 2D plane (PCA + LDA).
Outputs a .npz with whitening parameters and projection matrix.

Usage:
    python scripts/train_speaker_projection.py \
        --index artifacts/features/features_index.json \
        --output artifacts/projections/speaker_projection.npz
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("train_speaker_projection")


def load_index(path: Path) -> List[dict]:
    return json.loads(path.read_text())


def load_dataset(index: List[dict]) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    features = []
    speakers = []
    for entry in index:
        speaker = entry.get("speaker")
        if not speaker:
            continue
        feat_path = Path(entry["feature_path"])
        if not feat_path.exists():
            logger.warning("Missing feature file %s", feat_path)
            continue
        blob = np.load(feat_path)
        l2 = blob["l2"].astype(np.float32)
        features.append(l2)
        speakers.append(speaker)
    if not features:
        raise ValueError("No speaker-labelled samples found.")
    encoder = LabelEncoder()
    y = encoder.fit_transform(speakers)
    return np.stack(features), y, encoder


def train_projection(features: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
    scaler = StandardScaler().fit(features)
    feats_z = scaler.transform(features)

    lda_components = min(len(np.unique(y)) - 1, 2)
    if lda_components < 1:
        raise ValueError("Need at least two speakers for LDA projection")

    lda = LinearDiscriminantAnalysis(n_components=lda_components)
    lda.fit(feats_z, y)
    projection = lda.scalings_[:, :lda_components]

    return {
        "feature_mean": scaler.mean_.astype(np.float32),
        "feature_scale": scaler.scale_.astype(np.float32),
        "projection": projection.astype(np.float32),
    }


def save_artifact(artifact: Dict[str, np.ndarray], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **artifact)
    logger.info("Saved speaker projection to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train speaker projection for comet")
    parser.add_argument("--index", required=True, type=Path, help="features_index.json")
    parser.add_argument("--output", required=True, type=Path, help="Output .npz file")
    args = parser.parse_args()

    index = load_index(args.index)
    features, labels, encoder = load_dataset(index)
    logger.info("Loaded %d samples across %d speakers", len(features), len(encoder.classes_))

    artifact = train_projection(features, labels)
    artifact["speakers"] = encoder.classes_
    save_artifact(artifact, args.output)


if __name__ == "__main__":
    main()
