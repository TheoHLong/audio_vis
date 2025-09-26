# AudioVis: Speech → Neural Trajectories

AudioVis streams microphone audio into a FastAPI backend that runs WavLM, optional Whisper keyword spotting, and lightweight neuron analytics. The frontend renders heatmaps, principal component traces, and audio diagnostics so you can inspect how layers L2, L6, and L10 respond to live speech.

## Highlights

- **Real-time pipeline** – 40 ms frames with 20 ms hop, exponential smoothing, and automatic throttling keep latency below ~200 ms.
- **Layer analytics** – Correlate neurons with pitch/energy/spectral features, cluster by tuning, and expose PCA traces for each layer.
- **Rich frontend** – Canvas-based heatmaps, time-aligned spectrogram, keyword markers, transcript panel, and diagnostics sidebar.
- **Extensible tooling** – Scripts to pre-download models, cache features, and train semantic projectors for offline experiments.

## Project Layout

```
backend/
  config.py              # Runtime configuration helpers
  keywords.py            # Whisper-backed keyword extraction
  neuron_analysis.py     # Neuron sorting, PCA, and summary stats
  pipeline.py            # Core streaming pipeline (WavLM → payloads)
  projection.py          # Semantic projector + speaker clustering
  utils.py               # DSP utilities (RMS, YIN pitch, EMA, etc.)
frontend/
  index.html             # UI shell
  main.js                # Rendering + websocket client
  styles.css             # Layout and styling
scripts/
  download_models.py     # Grab WavLM, Whisper, and optional emotion models
  extract_features.py    # Batch WavLM feature extraction
  build_manifest_hf.py   # Create manifests from Hugging Face datasets
  train_semantic_projection.py  # Train 2‑D semantic projector
artifacts/               # Sample feature/projection artifacts (optional)
start_app.py             # Helper launcher for Whisper init + FastAPI
requirements.txt         # Python dependencies
```

## Prerequisites

- Python 3.10+
- Node-capable browser with microphone access
- Sufficient disk space for Hugging Face checkpoints (`~1.5 GB` if using Whisper + WavLM)

> ℹ️ Set `TRANSFORMERS_CACHE` (and optionally `HF_HOME`) to reuse existing model downloads.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Optional: warm the Hugging Face caches
python scripts/download_models.py
```

If you have a trained semantic projector, point the pipeline at it before launch:

```bash
export SEMANTIC_PROJECTION_PATH=artifacts/projections/semantic_librispeech.npz
```

## Running the App

### 1. Start the backend

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
# or
python start_app.py  # pre-inits Whisper then runs uvicorn
```

### 2. Open the frontend

Visit [http://localhost:8000](http://localhost:8000) and grant microphone access. The websocket stream sends batched layer activity along with diagnostics and optional transcripts.

### Controls

- **Start Listening** – Toggle microphone capture.
- **Performance Mode** – Switch between analysis/performance rendering modes.
- **Hide Grid** – Toggle the 3‑D grid overlay in the projection.
- **Reset** – Clear rolling buffers (audio, layers, clustering, keywords).

## Offline Workflows

The repository includes scripts to reproduce the semantic projector workflow:

1. **Create dataset manifest** – `python scripts/build_manifest_hf.py ...`
2. **Extract features** – `python scripts/extract_features.py --manifest ...`
3. **Train projector** – `python scripts/train_semantic_projection.py --index ...`

See `ENHANCEMENTS_SUMMARY.md` for detailed walkthroughs of the visualization improvements.

## Testing & Diagnostics

- Quick sanity check: `python -m compileall backend` to ensure sources import cleanly.
- Analytic smoke test: `python test_enhancements.py` (requires WavLM + Whisper checkpoints and a functional numpy/pytorch stack).
- Whisper-only probe: `python test_whisper.py` to validate your ASR environment.

If you customise the pipeline, regenerate the frontend bundle or rebuild any cached semantic assets as needed.

## Troubleshooting

- **Model downloads blocked** – Use `scripts/download_models.py --skip-*` flags to avoid optional checkpoints, or run on a machine with cached Hugging Face models.
- **High latency** – Lower `PipelineConfig.activity_history_seconds` or reduce `activity_neurons` to shrink payloads.
- **Missing transcripts** – The UI falls back to keyword markers if Whisper fails; check backend logs for Whisper/transformers errors.
- **Semantic projector unavailable** – The pipeline gracefully degrades, but diagnostics will show `projector_ready = false` until an artifact path is supplied.

Happy visualising!
