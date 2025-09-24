#!/usr/bin/env python3
"""Assemble a unified manifest for audio feature extraction.

This helper walks one or more audio directories, optionally merges
sidecar metadata (CSV/JSONL) with transcripts/emotion labels, and
writes a manifest CSV/JSONL containing the columns expected by
`extract_features.py`.

Typical usage (MELD-style layout, transcripts as .txt files):

    python scripts/build_manifest.py \
        --audio-root data/meld/audio \
        --pattern "**/*.wav" \
        --transcript-ext .txt \
        --speaker-from-parent 2 \
        --output data/manifest.csv

Example with metadata CSV providing transcripts/emotions:

    python scripts/build_manifest.py \
        --audio-root data/meld/audio \
        --metadata data/meld/metadata.csv \
        --metadata-format csv \
        --metadata-key-col clip_id \
        --metadata-transcript-col utterance \
        --metadata-emotion-col emotion \
        --key-type stem \
        --output data/manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger("build_manifest")

SUPPORTED_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


def collect_audio_files(root: Path, pattern: str) -> List[Path]:
    files = sorted({p for p in root.glob(pattern) if p.suffix.lower() in SUPPORTED_AUDIO_EXTS})
    logger.info("Found %d audio files under %s", len(files), root)
    return files


def compute_key(path: Path, key_type: str, base_dir: Optional[Path]) -> str:
    if key_type == "stem":
        return path.stem
    if key_type == "name":
        return path.name
    if key_type == "relative":
        if base_dir is None:
            raise ValueError("--base-dir is required when key-type=relative")
        return str(path.relative_to(base_dir))
    if key_type == "absolute":
        return str(path.resolve())
    raise ValueError(f"Unsupported key type: {key_type}")


def load_metadata(
    path: Optional[Path],
    key_col: str,
    transcript_col: Optional[str],
    emotion_col: Optional[str],
    speaker_col: Optional[str],
    fmt: str,
) -> Dict[str, Dict[str, Optional[str]]]:
    if path is None:
        return {}
    fmt = fmt.lower()
    records: Dict[str, Dict[str, Optional[str]]] = {}
    if fmt == "auto":
        if path.suffix.lower() in {".jsonl", ".json"}:
            fmt = "jsonl"
        else:
            fmt = "csv"
    if fmt == "csv":
        with path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                key = row.get(key_col)
                if not key:
                    continue
                records[key] = {
                    "transcript": row.get(transcript_col) if transcript_col else None,
                    "emotion": row.get(emotion_col) if emotion_col else None,
                    "speaker": row.get(speaker_col) if speaker_col else None,
                }
    elif fmt == "jsonl":
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = row.get(key_col)
                if not key:
                    continue
                records[key] = {
                    "transcript": row.get(transcript_col) if transcript_col else None,
                    "emotion": row.get(emotion_col) if emotion_col else None,
                    "speaker": row.get(speaker_col) if speaker_col else None,
                }
    else:
        raise ValueError(f"Unsupported metadata format: {fmt}")
    logger.info("Loaded %d metadata rows from %s", len(records), path)
    return records


def maybe_read_transcript(
    audio_path: Path,
    transcript_root: Optional[Path],
    transcript_ext: Optional[str],
) -> Optional[str]:
    if transcript_ext is None:
        return None
    if transcript_root:
        rel = audio_path.relative_to(audio_path.parents[len(audio_path.parts) - len(transcript_root.parts)])
        candidate = transcript_root / rel
        candidate = candidate.with_suffix(transcript_ext)
    else:
        candidate = audio_path.with_suffix(transcript_ext)
    if candidate.exists():
        text = candidate.read_text(encoding="utf-8", errors="ignore").strip()
        return text if text else None
    return None


def derive_speaker(path: Path, depth: int) -> Optional[str]:
    if depth <= 0:
        return None
    parts = path.parts
    if len(parts) <= depth:
        return None
    return parts[-(depth + 1)]


def write_manifest(records: List[Dict[str, Optional[str]]], output: Path, fmt: str, base_dir: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        fieldnames = ["path", "transcript", "emotion", "speaker"]
        with output.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in records:
                writer.writerow({k: row.get(k, "") or "" for k in fieldnames})
    elif fmt == "jsonl":
        with output.open("w", encoding="utf-8") as fh:
            for row in records:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        raise ValueError(f"Unsupported output format: {fmt}")
    logger.info("Wrote %d entries to %s", len(records), output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manifest for comet feature extraction")
    parser.add_argument("--audio-root", required=True, type=Path, help="Directory containing audio files")
    parser.add_argument("--pattern", default="**/*.wav", help="Glob pattern for audio files (relative to audio-root)")
    parser.add_argument("--base-dir", type=Path, default=None, help="Base directory for relative paths (default audio-root)")
    parser.add_argument("--output", required=True, type=Path, help="Destination manifest (csv/jsonl)")
    parser.add_argument("--output-format", choices=["csv", "jsonl"], default="csv")
    parser.add_argument("--key-type", choices=["stem", "name", "relative", "absolute"], default="stem",
                        help="How to derive keys when merging metadata")
    parser.add_argument("--metadata", type=Path, default=None, help="Optional metadata file (csv/jsonl)")
    parser.add_argument("--metadata-format", choices=["auto", "csv", "jsonl"], default="auto")
    parser.add_argument("--metadata-key-col", default="path", help="Column in metadata used to match audio")
    parser.add_argument("--metadata-transcript-col", default=None, help="Column containing transcripts")
    parser.add_argument("--metadata-emotion-col", default=None, help="Column containing emotion labels")
    parser.add_argument("--metadata-speaker-col", default=None, help="Column containing speaker IDs")
    parser.add_argument("--transcript-root", type=Path, default=None,
                        help="Optional directory mirroring audio layout that holds transcript files")
    parser.add_argument("--transcript-ext", default=None,
                        help="If set, attempts to read transcript files with this extension (e.g. .txt)")
    parser.add_argument("--speaker-from-parent", type=int, default=0,
                        help="If >0, derive speaker from parent directory depth (1=immediate parent)")
    args = parser.parse_args()

    audio_root = args.audio_root.expanduser().resolve()
    base_dir = (args.base_dir or audio_root).expanduser().resolve()

    audio_files = collect_audio_files(audio_root, args.pattern)
    if not audio_files:
        logger.error("No audio files found")
        return

    metadata = load_metadata(
        path=args.metadata,
        key_col=args.metadata_key_col,
        transcript_col=args.metadata_transcript_col,
        emotion_col=args.metadata_emotion_col,
        speaker_col=args.metadata_speaker_col,
        fmt=args.metadata_format,
    )

    transcript_root = args.transcript_root.expanduser().resolve() if args.transcript_root else None

    manifest_rows: List[Dict[str, Optional[str]]] = []

    for audio_path in audio_files:
        key = compute_key(audio_path, args.key_type, base_dir if args.key_type == "relative" else None)
        md = metadata.get(key, {})
        transcript = md.get("transcript") or maybe_read_transcript(audio_path, transcript_root, args.transcript_ext)
        emotion = md.get("emotion")
        speaker = md.get("speaker")
        if not speaker and args.speaker_from_parent > 0:
            speaker = derive_speaker(audio_path.relative_to(audio_root), args.speaker_from_parent)

        rel_path = audio_path.relative_to(base_dir)
        manifest_rows.append({
            "path": str(rel_path),
            "transcript": transcript or "",
            "emotion": emotion or "",
            "speaker": speaker or "",
        })

    write_manifest(manifest_rows, args.output, args.output_format, base_dir)


if __name__ == "__main__":
    main()
