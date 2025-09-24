#!/usr/bin/env python3
"""Helper script to pre-fetch model weights for offline use."""

from __future__ import annotations

import argparse
import logging

from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor, pipeline

logging.basicConfig(level=logging.INFO, format="%(message)s")


def fetch_wavlm(model: str) -> None:
    logging.info("Downloading %s …", model)
    try:
        AutoProcessor.from_pretrained(model)
    except Exception as proc_exc:
        logging.warning("AutoProcessor load failed (%s); trying AutoFeatureExtractor", proc_exc)
        AutoFeatureExtractor.from_pretrained(model)
    AutoModel.from_pretrained(model)
    logging.info("✓ %s ready", model)


def fetch_whisper(model: str) -> None:
    logging.info("Downloading %s …", model)
    pipeline("automatic-speech-recognition", model=model)
    logging.info("✓ %s ready", model)


def fetch_emotion(model: str) -> None:
    logging.info("Downloading %s …", model)
    pipeline("audio-classification", model=model, top_k=None)
    logging.info("✓ %s ready", model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download models for the comet visualiser")
    parser.add_argument(
        "--wavlm",
        default="microsoft/wavlm-base",
        help="WavLM model identifier (default: microsoft/wavlm-base)",
    )
    parser.add_argument(
        "--whisper",
        default="openai/whisper-tiny.en",
        help="Optional Whisper model identifier for keyword extraction",
    )
    parser.add_argument(
        "--skip-whisper",
        action="store_true",
        help="Skip downloading the Whisper model",
    )
    parser.add_argument(
        "--emotion",
        default="superb/hubert-large-superb-er",
        help="Optional speech emotion model identifier",
    )
    parser.add_argument(
        "--skip-emotion",
        action="store_true",
        help="Skip downloading the emotion model",
    )
    args = parser.parse_args()

    fetch_wavlm(args.wavlm)
    if not args.skip_whisper:
        fetch_whisper(args.whisper)
    if not args.skip_emotion:
        fetch_emotion(args.emotion)


if __name__ == "__main__":
    main()
