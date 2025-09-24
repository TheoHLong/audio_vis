#!/usr/bin/env python3
"""Train a semantic projection head for WavLM features.

Loads cached feature vectors (from extract_features.py) plus transcripts,
then learns a 2D semantic plane by aligning L10 embeddings with
SentenceTransformer text embeddings. Outputs an .npz file containing
whitening stats, ridge regression weights, and PCA components for the
2D comet plane.

Usage:
    python scripts/train_semantic_projection.py \
        --index artifacts/features/features_index.json \
        --output artifacts/projections/semantic_projection.npz
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import List

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("train_semantic_projection")


def load_index(path: Path) -> List[dict]:
    data = json.loads(path.read_text())
    return data


def load_vectors(index: List[dict], min_transcript_len: int = 5) -> tuple[np.ndarray, list[str]]:
    l10_list = []
    texts = []
    for entry in index:
        transcript = (entry.get("transcript") or "").strip()
        if len(transcript) < min_transcript_len:
            continue
        feat_path = Path(entry["feature_path"])
        if not feat_path.exists():
            logger.warning("Missing feature file %s", feat_path)
            continue
        blob = np.load(feat_path)
        l10 = blob["l10"].astype(np.float32)
        l10_list.append(l10)
        texts.append(transcript)
    if not l10_list:
        raise ValueError("No samples with transcripts found; cannot train semantic projection.")
    return np.stack(l10_list), texts


def train_projection(
    features: np.ndarray,
    transcripts: list[str],
    text_model: str,
    pca_components: int,
    ridge_alpha: float,
) -> dict:
    logger.info("Encoding %d transcripts with %s", len(transcripts), text_model)
    encoder = SentenceTransformer(text_model)
    text_embeddings = encoder.encode(transcripts, normalize_embeddings=True, show_progress_bar=True)

    feature_scaler = StandardScaler().fit(features)
    feat_z = feature_scaler.transform(features)

    logger.info("Fitting ridge regression (alpha=%.4f)", ridge_alpha)
    reg = Ridge(alpha=ridge_alpha, fit_intercept=True)
    reg.fit(feat_z, text_embeddings)
    predicted = reg.predict(feat_z)

    logger.info("Running PCA (%d components) on predicted semantic space", pca_components)
    semantic_pca = PCA(n_components=pca_components, whiten=False)
    coords = semantic_pca.fit_transform(predicted)

    artifact = {
        "feature_mean": feature_scaler.mean_.astype(np.float32),
        "feature_scale": feature_scaler.scale_.astype(np.float32),
        "ridge_coef": reg.coef_.astype(np.float32),
        "ridge_intercept": reg.intercept_.astype(np.float32),
        "semantic_pca_components": semantic_pca.components_.astype(np.float32),
        "semantic_pca_mean": semantic_pca.mean_.astype(np.float32),
        "explained_variance": semantic_pca.explained_variance_ratio_.astype(np.float32),
        "text_model": text_model,
        "ridge_alpha": ridge_alpha,
    }
    return artifact


def save_artifact(artifact: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **artifact)
    logger.info("Saved semantic projection to %s", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train semantic projection for comet")
    parser.add_argument("--index", required=True, type=Path, help="features_index.json produced by extract_features")
    parser.add_argument("--output", required=True, type=Path, help="Destination .npz file")
    parser.add_argument("--text-model", default="all-mpnet-base-v2", help="SentenceTransformer model")
    parser.add_argument("--components", type=int, default=2, help="Number of PCA components for the plane")
    parser.add_argument("--ridge-alpha", type=float, default=1e-2, help="Ridge regression strength")
    parser.add_argument("--min-transcript-len", type=int, default=5, help="Minimum transcript length to use")
    args = parser.parse_args()

    index = load_index(args.index)
    features, texts = load_vectors(index, min_transcript_len=args.min_transcript_len)
    artifact = train_projection(
        features=features,
        transcripts=texts,
        text_model=args.text_model,
        pca_components=args.components,
        ridge_alpha=args.ridge_alpha,
    )
    save_artifact(artifact, args.output)


if __name__ == "__main__":
    main()
