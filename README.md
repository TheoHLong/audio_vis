# Speech → Neural Trajectories

A live “speech-to-visuals” experience where microphone input becomes a layered 3D line chart: Layer 10 positions semantic trajectories, Layer 6 controls pulse height, and Layer 2 colours segments by speaker voiceprint. Whisper transcripts remain available for context.

---

## What you get

- **Neural Trajectory Plot** – Streamed WavLM layers become a 3D-style line chart: X = time, Y = neuron index, Z = activity. L10 positions guide the path, L2 colours segments by speaker voiceprint, and L6 controls pulse amplitude.
- **Layer separation** – L10 feeds the frozen semantic plane (spatial clusters), L2 encodes speaker voiceprints (star colour + connection affinity), and L6 drives star pulsing / nebula energy.
- **Real-time loop (<200 ms)** – 40 ms windows, 20 ms hop, EMA smoothing. Only ~120 recent stars are streamed so updates stay live.
- **Live transcript overlay** – Whisper tiny can still stream rolling transcripts for context; interface degrades gracefully without it.
- **Diagnostics panel** – Readiness indicators for semantic projector, speaker clustering, and Whisper.

---

## Repository layout

```
backend/
  config.py           # Pipeline configuration
  main.py             # FastAPI app + websocket
  pipeline.py         # Streaming WavLM → constellation payloads
  projection.py       # Frozen semantic projector / incremental PCA fallback
  keywords.py         # Whisper keyword probe
  utils.py            # DSP helpers (RMS, YIN, EMA)
frontend/
  index.html          # UI shell
  main.js             # Constellation renderer + audio capture
  styles.css          # Styling
scripts/
  download_models.py  # Pre-fetch WavLM + Whisper
  build_manifest_hf.py # HF dataset fetch + manifest builder
  extract_features.py  # Cache WavLM features
  train_semantic_projection.py # Train 2D semantic map
requirements.txt
README.md
```

---

## Prerequisites

- Python 3.10+
- Modern browser
- Microphone access

> **Models:** install `microsoft/wavlm-base` and `openai/whisper-tiny.en`. The helper script pulls both.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# optional but recommended
python scripts/download_models.py
```

If you already cached models elsewhere, set `TRANSFORMERS_CACHE` before launching the server.

> **Semantic plane:** after training with `scripts/train_semantic_projection.py`, supply the artefact path:
>
> ```bash
> export SEMANTIC_PROJECTION_PATH=artifacts/projections/semantic_librispeech.npz
> ```
>
> (Or set `PipelineConfig.semantic_projector_path`.)

---

## Running the demo

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Visit [http://localhost:8000](http://localhost:8000) and grant microphone access.

### Controls

- **Start Listening** toggles the mic stream.
- **Performance Mode** adjusts visual gain.
- **Show/Hide Grid** toggles auxiliary guide/grid lines in the 3D projection.
- **Reset** clears the buffered trajectories for a fresh window.

### Visual encodings

| Channel          | Source (WavLM)                 | Visual cue                       |
|------------------|-------------------------------|----------------------------------|
| Semantic map     | L10 → frozen projector        | Star position / clustering       |
| Speaker identity | L2 voiceprint clustering      | Star colour                      |
| Prosody energy   | L6 norm & variation           | Star pulsing & nebula intensity  |
| Loudness         | RMS (frame energy)            | Ridge amplitude baseline         |
| Audio waveform   | RMS (raw waveform)            | Bottom ridge waveform            |
| Transcript       | Whisper tiny                  | Rolling text panel               |

Diagnostics show when the semantic plane, speaker clustering, or Whisper probe are warming up (≈6 s).

---

## Offline training workflow

1. **Build manifest** (e.g. Librispeech subset):
   ```bash
   python scripts/build_manifest_hf.py \
     --dataset librispeech_asr \
     --config clean \
     --split train.100 \
     --limit 1000 \
     --audio-column audio \
     --transcript-column text \
     --output-audio-dir data/librispeech/audio \
     --output-manifest data/librispeech/manifest.csv
   ```
2. **Extract features**:
   ```bash
   python scripts/extract_features.py \
     --manifest data/librispeech/manifest.csv \
     --base-dir data/librispeech \
     --output-dir artifacts/features/librispeech
   ```
3. **Train semantic plane**:
   ```bash
   python scripts/train_semantic_projection.py \
     --index artifacts/features/librispeech/features_index.json \
     --output artifacts/projections/semantic_librispeech.npz
   ```

Drop the artefact into `SEMANTIC_PROJECTION_PATH` and restart the server.

---

## Implementation notes

- **Layer separation** – L10 ridge tracks semantics, L6 ridge shows prosodic energy, and L2 ridge highlights voiceprint changes. All three are updated frame-by-frame from WavLM.
- **Optional projector** – You can still supply a frozen L10 projector to stabilise neuron selection; the pipeline falls back gracefully if none is provided.
- **Streaming efficiency** – Layer histories are capped (≈15 s) and payloads stay small; updates go out roughly 6 Hz for smooth animation.
- **Keyword resilience** – If Whisper weights are missing, the UI shows placeholders but continues plotting trajectories.

---

## Verification

```bash
python -m compileall backend
python -m compileall frontend/main.js
```

Run end-to-end once models are cached to validate audio flow.
