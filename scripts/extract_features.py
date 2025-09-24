#!/usr/bin/env python3
"""Offline feature extraction for WavLM comet projection training.

Given a manifest (CSV/JSON lines) with audio paths and optional labels,
this script computes WavLM hidden states (L2/L6/L10), waveform heuristics,
and stores them as individual `.npz` files plus an index JSON.

Expected manifest columns/keys:
- `path` (or `audio_path`): path to audio file (wav/flac/etc.)
- optional `transcript`: string transcript
- optional `emotion`: categorical label
- optional `speaker`: speaker identifier

Usage:
    python scripts/extract_features.py \
        --manifest data/manifest.csv \
        --output-dir artifacts/features \
        --model microsoft/wavlm-base
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import librosa
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("extract_features")

TARGET_SR = 16_000


@dataclass
class SampleMeta:
    uid: str
    audio_path: str
    feature_path: str
    duration: float
    rms: float
    pitch: float
    transcript: Optional[str] = None
    emotion: Optional[str] = None
    speaker: Optional[str] = None


def parse_manifest(path: Path) -> List[Dict[str, str]]:
    if path.suffix.lower() in {".json", ".jsonl"}:
        records: List[Dict[str, str]] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        return records
    # assume CSV
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return [row for row in reader]


def load_audio(path: Path) -> np.ndarray:
    audio, sr = librosa.load(path, sr=TARGET_SR, mono=True)
    return audio.astype(np.float32)


def compute_pitch(samples: np.ndarray, sr: int = TARGET_SR) -> float:
    if samples.size == 0:
        return 0.0
    f0 = librosa.yin(samples, fmin=60, fmax=400, sr=sr, frame_length=2048, hop_length=256)
    if np.all(np.isnan(f0)):
        return 0.0
    return float(np.nanmean(f0))


def rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))


def make_uid(audio_path: Path, index: int) -> str:
    base = audio_path.stem.lower().replace(" ", "_")
    return f"{base}_{index:05d}"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def process_manifest(
    records: List[Dict[str, str]],
    base_dir: Path,
    output_dir: Path,
    model_name: str,
    device: Optional[str],
    batch_size: int,
) -> List[SampleMeta]:
    ensure_dir(output_dir)

    chosen_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info("Loading WavLM '%s' on %s", model_name, chosen_device)
    try:
        processor = AutoProcessor.from_pretrained(model_name)
    except Exception as proc_exc:
        logger.warning("AutoProcessor load failed (%s); falling back to AutoFeatureExtractor", proc_exc)
        processor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(chosen_device)
    model.eval()
    torch.set_grad_enabled(False)

    metadata: List[SampleMeta] = []

    for idx, record in enumerate(tqdm(records, desc="Extracting", unit="clip")):
        raw_path = record.get("path") or record.get("audio_path")
        if not raw_path:
            logger.warning("Skipping row %s – missing 'path'", record)
            continue
        audio_path = (base_dir / raw_path).expanduser().resolve()
        if not audio_path.exists():
            logger.warning("Audio not found: %s", audio_path)
            continue

        try:
            samples = load_audio(audio_path)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", audio_path, exc)
            continue

        duration = samples.shape[0] / TARGET_SR
        rms_value = rms(samples)
        pitch_value = compute_pitch(samples)

        inputs = processor(
            samples,
            sampling_rate=TARGET_SR,
            return_tensors="pt",
            padding=False,
        )
        input_values = inputs["input_values"].to(chosen_device)
        with torch.no_grad():
            outputs = model(input_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        l2_vec = hidden_states[2].mean(dim=1).squeeze(0).cpu().numpy()
        l6_vec = hidden_states[6].mean(dim=1).squeeze(0).cpu().numpy()
        l10_vec = hidden_states[10].mean(dim=1).squeeze(0).cpu().numpy()

        uid = make_uid(audio_path, idx)
        feat_path = output_dir / f"{uid}.npz"
        np.savez(
            feat_path,
            l2=l2_vec.astype(np.float32),
            l6=l6_vec.astype(np.float32),
            l10=l10_vec.astype(np.float32),
        )

        meta = SampleMeta(
            uid=uid,
            audio_path=str(audio_path),
            feature_path=str(feat_path),
            duration=float(duration),
            rms=rms_value,
            pitch=pitch_value,
            transcript=record.get("transcript"),
            emotion=record.get("emotion"),
            speaker=record.get("speaker"),
        )
        metadata.append(meta)

    logger.info("Extracted %d clips", len(metadata))
    return metadata


def write_index(metadata: List[SampleMeta], output_dir: Path) -> Path:
    index_path = output_dir / "features_index.json"
    with index_path.open("w", encoding="utf-8") as fh:
        json.dump([asdict(item) for item in metadata], fh, ensure_ascii=False, indent=2)
    logger.info("Wrote index %s", index_path)
    return index_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract WavLM features for comet training")
    parser.add_argument("--manifest", required=True, type=Path, help="CSV/JSON manifest of audio clips")
    parser.add_argument("--base-dir", type=Path, default=Path("."), help="Base directory for relative audio paths")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory to store npz features")
    parser.add_argument("--model", default="microsoft/wavlm-base", help="Hugging Face model name")
    parser.add_argument("--device", default=None, help="Torch device override (cpu/cuda)")
    parser.add_argument("--batch-size", type=int, default=1, help="Reserved for future batching")
    args = parser.parse_args()

    records = parse_manifest(args.manifest)
    if not records:
        logger.error("No records found in manifest %s", args.manifest)
        return

    metadata = process_manifest(
        records=records,
        base_dir=args.base_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
    )
    if metadata:
        write_index(metadata, args.output_dir)


if __name__ == "__main__":
    main()
