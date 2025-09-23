# Speech → Visual Comet

A live "speech-to-visuals" experience inspired by the **WavLM Base (L2/L6/L10)** comet trail concept. Audio from the microphone streams to a FastAPI backend that projects selected WavLM layers into 2D in real time. The browser renders a comet-like trail whose colour, width, glow, and transparency reflect speaker identity, loudness, pitch, and semantic intensity. An optional Whisper probe surfaces floating keywords, while an emotion dial summarises live valence–arousal estimates.

---

## What you get

- **Real-time pipeline (<200 ms budget)** – 40 ms windows, 20 ms stride, incremental PCA→2D, EMA smoothing per the implementation notes.
- **Layer-aware mappings** – L2 drives speaker clusters/colour, L6 forms the comet trajectory, L10 governs semantic intensity and tail transparency.
- **Energy & prosody cues** – RMS widens the tail, a lightweight YIN pitch estimator brightens the glow, transient surges spawn particle bursts.
- **Emotion dial** – Heuristic valence/arousal derived from loudness, pitch deviation, and semantic energy.
- **Performance toggle** – Switch between *Analysis* (calm) and *Performance* (amped) render styles; reset tail with one click.
- **Optional keywords** – If `openai/whisper-tiny.en` is cached locally, floating keyword bubbles appear near the comet head.
- **Diagnostics panel** – Reassure evaluators with live readiness indicators for projector, speaker clustering, and keyword modules.

---

## Repository layout

```
backend/
  config.py           # Pipeline configuration dataclass
  main.py             # FastAPI application + websocket
  pipeline.py         # Streaming WavLM feature pipeline → payloads
  projection.py       # Incremental PCA + speaker clustering helpers
  keywords.py         # Optional Whisper-based keyword extractor
  utils.py            # DSP helpers (RMS, YIN, EMA, rolling stats)
frontend/
  index.html          # UI shell and controls
  main.js             # Web audio capture + canvas visualisation
  styles.css          # Glassmorphism UI styling
scripts/
  download_models.py  # Helper to pre-fetch Hugging Face weights
WavLM_Base_Comet_V2.md # Original design brief / parameter doc
requirements.txt
README.md
```

---

## Prerequisites

- Python 3.10+
- A modern browser (tested with recent Chromium/WebKit builds)
- Microphone access
- GPU optional (CPU works; expect higher latency)

> **Models:** the backend requires `microsoft/wavlm-base`. Keyword bubbles are enabled when `openai/whisper-tiny.en` is available. The helper script below will download both.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Optional but recommended: pre-fetch weights while online
python scripts/download_models.py           # adds WavLM + Whisper to local cache
python scripts/download_models.py --skip-whisper  # skip keywords if bandwidth-limited
```

If you already have the models cached elsewhere, set `TRANSFORMERS_CACHE` (or symlink the files) before launching the server.

---

## Running the demo

```bash
# Activate your virtualenv first
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open [http://localhost:8000](http://localhost:8000) in a browser and grant microphone access.

### Controls

- **Start Listening** toggles the microphone stream.
- **Performance Mode** amplifies line width, glow, and particles for stage demos. Click again to return to Analysis.
- **Reset Tail** clears accumulated states without restarting the server.

### Visual encodings

| Channel | Source | Visual cue |
| --- | --- | --- |
| Speaker identity | WavLM L2 → MiniBatchKMeans | Colour hue |
| Loudness (RMS) | Waveform | Tail width / head radius |
| Pitch | YIN estimate | Glow intensity |
| Semantic intensity | L10 vector norm | Tail opacity |
| Prosodic bursts | RMS surges | Particle flares |
| Emotion | RMS + pitch + L10 norms | Background hue + emotion dial |
| Keywords (optional) | Whisper tiny | Floating labels near head |

Diagnostics in the sidebar show when the PCA projector, speaker clustering, or keyword probe are ready. Until the warm-up buffers fill (~6 s), those items report “warming”.

---

## Implementation notes

- **Incremental PCA** – The projector maintains a 32-D latent space before folding down to 2D; it gracefully falls back to raw axes until enough data is seen.
- **Smoothing** – EMA constants follow the document (τ≈0.25 s). Tail retention is capped at 2.5 s with timestamp-based eviction.
- **Audio ingestion** – The browser resamples microphone audio to 16 kHz float32 frames and streams them over a websocket. The backend re-frames at 40 ms / 20 ms and pushes batched payloads roughly 6 Hz.
- **Emotion heuristics** – Valence leans on pitch deviation plus semantic energy; arousal tracks loudness. Treat them as expressive hints rather than ground truth.
- **Keywords** – If Whisper weights are missing, the extractor logs a warning and the UI shows a fallback state with dashes.

---

## Verification

Lightweight checks that avoid heavyweight inference:

```bash
# Python syntax check
python -m compileall backend

# Front-end lint-free build (basic formatting / no bundler needed)
python -m http.server  # optional quick static serve for manual inspection
```

Run the live pipeline once models are cached to validate audio flow.

---

## Extending the demo

- Swap the 2D projector for Parametric UMAP once you have an anchor dataset.
- Plug in a stronger ASR or topic classifier for richer semantic overlays.
- Mirror the comet onto a “state metro” mini-map: the backend already exposes speaker cluster IDs; add a heatmap if desired.
- Push processed frames to a recorder (e.g. `MediaRecorder` + WebM) to capture rehearsals.

---

## Known limitations

- Latency depends on hardware; CPU-only laptops may drift above the 200 ms target under load.
- Whisper download is skipped offline; keywords revert to a placeholder.
- The heuristic emotion probe uses lightweight cues and should not be interpreted as an absolute classifier.

Enjoy exploring the bridge between voice intention and visual expression!
