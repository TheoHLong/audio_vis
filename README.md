# Speech → Visual Comet

A live "speech-to-visuals" experience inspired by the **WavLM Base (L2/L6/L10)** comet trail concept. Audio from the microphone streams to a FastAPI backend that projects selected WavLM layers into 2D in real time. The browser renders a comet-like trail whose colour, width, glow, and transparency reflect speaker identity, loudness, pitch, and semantic intensity. An optional Whisper probe surfaces floating keywords, while an emotion dial summarises live valence–arousal estimates.

---

## What you get

- **Constellation Builder** – Speech segments become stars on a semantic map trained offline; brightness tracks loudness, links track thematic proximity, and a nebula background breathes with overall mood.
- **Layer-aware mappings** – L2 still feeds speaker clustering, while L10 drives the frozen semantic plane (SentenceTransformer + ridge regression) so positions stay interpretable.
- **Energy & prosody cues** – RMS powers star brightness, pitch adds twinkle, and semantic intensity sets star radius.
- **Emotion dial & nebula** – A speech emotion recogniser colours stars (red giants for anger, ice tones for calm) and animates the backdrop.
- **Live transcript & keywords** – Whisper snippets populate a rolling transcript; recent keywords label constellations and feed the theme list.
- **Diagnostics panel** – Readiness indicators for semantic projector, speaker clustering, Whisper, and the SER head keep the demo trustworthy.

---

## Repository layout

```
backend/
  config.py           # Pipeline configuration dataclass
  main.py             # FastAPI application + websocket
  pipeline.py         # Streaming WavLM feature pipeline → payloads
  projection.py       # Incremental PCA + speaker clustering helpers
  emotion.py          # Optional SER head + emotion state dataclass
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

> **Models:** the backend requires `microsoft/wavlm-base`. Keyword bubbles and transcripts need `openai/whisper-tiny.en`. The optional emotion dial upgrade pulls `speechbrain/emotion-recognition-wav2vec2-large-960h`. The helper script below pre-fetches all three.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Optional but recommended: pre-fetch weights while online
python scripts/download_models.py           # adds WavLM + Whisper + SER model to cache
python scripts/download_models.py --skip-whisper --skip-emotion  # lighter download
```

If you already have the models cached elsewhere, set `TRANSFORMERS_CACHE` (or symlink the files) before launching the server.

> **Offline semantic plane:** after downloading data and running the scripts in `scripts/train_semantic_projection.py`, point the backend at the resulting artefact:
>
> ```bash
> export SEMANTIC_PROJECTION_PATH=artifacts/projections/semantic_librispeech.npz
> ```
>
> (Alternatively wire the path into `PipelineConfig.semantic_projector_path`.)

---

## Running the demo

```bash
# Activate your virtualenv first
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open [http://localhost:8000](http://localhost:8000) in a browser and grant microphone access.

### Controls

- **Start Listening** toggles the microphone stream.
- **Performance Mode** still nudges stylistic rendering choices (calm vs stage).
- **Show/Hide Links** toggles the semantic connection lines between stars.
- **Reset Tail** clears accumulated stars/transcripts for a fresh constellation.

### Visual encodings

| Channel | Source | Visual cue |
| --- | --- | --- |
| Speaker identity | WavLM L2 → MiniBatchKMeans | Sidebar colour key |
| Loudness (RMS) | Waveform | Star brightness |
| Pitch | YIN estimate | Twinkle amplitude |
| Semantic intensity | L10 vector norm | Star radius |
| Themes/Topics | Whisper keywords | Star labels & constellation list |
| Emotion | SER model (speechbrain) + heuristic fallback | Star colour + nebula hue |
| Transcript | Whisper tiny | Rolling text panel |
| Links | Nearest neighbours in semantic plane | Toolbar toggle |

Diagnostics in the sidebar show when the semantic projector, speaker clustering, Whisper probe, or emotion head are ready. Until the warm-up buffers fill (~6 s), those items report “warming”.

---

## Implementation notes

- **Frozen semantic plane** – The offline ridge regression + PCA artefact (SentenceTransformer-aligned) replaces incremental PCA when provided; the pipeline falls back automatically if no file is configured.
- **Rolling smoothing** – EMA constants follow the document (τ≈0.25 s). The star deque keeps ~240 recent points so the map stays fluid without overwhelming the viewer.
- **Audio ingestion** – The browser resamples microphone audio to 16 kHz float32 frames and streams them over a websocket. The backend re-frames at 40 ms / 20 ms and pushes constellation updates roughly 6 Hz.
- **Emotion analyser** – The speech emotion recogniser colours stars and sets the nebula mood; heuristics blend in whenever the model is unavailable.
- **Keywords** – If Whisper weights are missing, the extractor logs a warning and the UI shows a placeholder while still plotting stars.

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
