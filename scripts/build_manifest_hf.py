#!/usr/bin/env python3
"""Download Hugging Face datasets and build a comet-ready manifest.

This script pulls an audio dataset from the Hugging Face Hub using the
`datasets` library, writes each clip to disk (WAV), and produces a
manifest CSV/JSONL with columns compatible with `extract_features.py`.

Examples
--------
Common Voice (English subset):
    python scripts/build_manifest_hf.py \
        --dataset mozilla-foundation/common_voice_17_0 \
        --config en \
        --split train \
        --audio-column audio \
        --transcript-column sentence \
        --speaker-column client_id \
        --sample-rate 16000 \
        --limit 500 \
        --output-audio-dir data/common_voice/audio \
        --output-manifest data/common_voice/manifest.csv

MELD emotion clips:
    python scripts/build_manifest_hf.py \
        --dataset declare-lab/MELD \
        --split train \
        --audio-column audio \
        --transcript-column utterance \
        --emotion-column emotion \
        --speaker-column speaker \
        --output-audio-dir data/meld/audio \
        --output-manifest data/meld/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import datasets
import numpy as np
import soundfile as sf
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("build_manifest_hf")

SUPPORTED_AUDIO_FORMATS = {
    "wav": ("PCM_16", "wav"),
    "flac": ("FLAC", "flac"),
}


def write_audio(array: np.ndarray, sr: int, path: Path, audio_format: str) -> None:
    subtype, ext = SUPPORTED_AUDIO_FORMATS.get(audio_format, SUPPORTED_AUDIO_FORMATS["wav"])
    path = path.with_suffix(f".{ext}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, array, sr, subtype=subtype)


def normalise_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download HF dataset and build manifest")
    parser.add_argument("--dataset", required=True, help="Dataset name on Hugging Face (e.g. mozilla-foundation/common_voice_17_0)")
    parser.add_argument("--config", default=None, help="Dataset configuration / language (optional)")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--audio-column", default="audio", help="Column with Audio feature")
    parser.add_argument("--transcript-column", default=None, help="Column containing transcript text")
    parser.add_argument("--emotion-column", default=None, help="Column with emotion label")
    parser.add_argument("--speaker-column", default=None, help="Column with speaker id")
    parser.add_argument("--sample-rate", type=int, default=16_000, help="Resample audio to this rate")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of clips")
    parser.add_argument("--output-audio-dir", required=True, type=Path, help="Where to write audio files")
    parser.add_argument("--output-manifest", required=True, type=Path, help="Manifest file (csv/jsonl) path")
    parser.add_argument("--output-format", choices=["csv", "jsonl"], default=None, help="Manifest format (inferred from extension if not set)")
    parser.add_argument("--audio-format", choices=list(SUPPORTED_AUDIO_FORMATS.keys()), default="wav", help="File format for saved audio")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N samples (useful for pagination)")
    parser.add_argument("--hf-token", default=None, help="Optional Hugging Face access token")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Allow execution of dataset repository code (required for script-based datasets)")
    args = parser.parse_args()

    load_kwargs = {"token": args.hf_token} if args.hf_token else {}
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    try:
        ds = datasets.load_dataset(args.dataset, args.config, split=args.split, **load_kwargs)
    except RuntimeError as exc:
        msg = str(exc)
        if "Dataset scripts are no longer supported" in msg:
            logger.error(
                "This dataset still relies on a loading script. Install an earlier version of 'datasets' (e.g. pip install \"datasets<2.19\") or choose a parquet-based dataset."
            )
            return
        raise
    logger.info("Loaded dataset %s (%s) with %d records", args.dataset, args.split, len(ds))

    if args.sample_rate:
        ds = ds.cast_column(args.audio_column, datasets.features.Audio(sampling_rate=args.sample_rate))

    limit = args.limit if args.limit is not None else len(ds)
    start = args.offset
    end = min(start + limit, len(ds))
    if start >= len(ds):
        logger.error("Offset %d beyond dataset length %d", start, len(ds))
        return

    out_audio_dir = args.output_audio_dir.expanduser().resolve()
    out_manifest = args.output_manifest.expanduser().resolve()
    out_audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []

    for idx in tqdm(range(start, end), desc="Saving audio"):
        example = ds[idx]
        audio = example[args.audio_column]
        array = audio["array"]
        sr = audio["sampling_rate"]
        if array is None:
            continue
        filename = f"{Path(args.dataset).name}_{args.split}_{idx:06d}.wav"
        save_path = out_audio_dir / filename
        write_audio(np.asarray(array), sr, save_path, args.audio_format)

        row = {
            "path": str(save_path.relative_to(out_audio_dir.parent)),
            "transcript": normalise_text(example.get(args.transcript_column)) if args.transcript_column else "",
            "emotion": normalise_text(example.get(args.emotion_column)) if args.emotion_column else "",
            "speaker": normalise_text(example.get(args.speaker_column)) if args.speaker_column else "",
        }
        manifest_rows.append(row)

    fmt = args.output_format or ("jsonl" if out_manifest.suffix.lower() == ".jsonl" else "csv")
    out_manifest.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "csv":
        with out_manifest.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["path", "transcript", "emotion", "speaker"])
            writer.writeheader()
            for row in manifest_rows:
                writer.writerow(row)
    else:
        with out_manifest.open("w", encoding="utf-8") as fh:
            for row in manifest_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    logger.info("Wrote %d examples", len(manifest_rows))
    logger.info("Audio saved under %s", out_audio_dir)
    logger.info("Manifest saved to %s", out_manifest)


if __name__ == "__main__":
    main()
