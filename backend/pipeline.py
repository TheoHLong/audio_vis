from __future__ import annotations

import json
import logging
import math
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor

from .config import PipelineConfig
from .keywords import KeywordCandidate, KeywordExtractor
from .projection import FrozenSemanticProjector, SpeakerClusterer
from .utils import (
    RollingStat,
    ema_update,
    estimate_pitch_yin,
    normalize,
    rms,
    softclip,
)

logger = logging.getLogger(__name__)


@dataclass
class Diagnostics:
    projector_ready: bool
    speaker_ready: bool
    keyword_ready: bool


class CometPipeline:
    """Streaming speech-to-visualisation pipeline."""

    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        model_name: str = "microsoft/wavlm-base",
        device: Optional[str] = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        logger.info("Loading WavLM model '%s' on %s", model_name, self.device)
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
        except Exception as proc_exc:
            logger.warning("AutoProcessor load failed (%s); falling back to AutoFeatureExtractor", proc_exc)
            try:
                self.processor = AutoFeatureExtractor.from_pretrained(model_name)
            except Exception as feat_exc:
                logger.error("Failed to load processor for '%s': %s", model_name, feat_exc)
                raise
        try:
            self.model = AutoModel.from_pretrained(model_name)
        except Exception as model_exc:
            logger.error("Failed to load model '%s': %s", model_name, model_exc)
            raise

        self.model.to(self.device)
        self.model.eval()

        hidden_dim = self.model.config.hidden_size

        semantic_path = self.config.semantic_projector_path or os.environ.get("SEMANTIC_PROJECTION_PATH")
        self.semantic_projector: Optional[FrozenSemanticProjector] = None
        if semantic_path:
            try:
                self.semantic_projector = FrozenSemanticProjector(semantic_path)
                logger.info("Loaded semantic projector from %s", semantic_path)
            except Exception as exc:  # pragma: no cover - runtime artifact issues
                logger.warning("Failed to load semantic projector '%s': %s", semantic_path, exc)
                self.semantic_projector = None

        self.speaker_clusterer = SpeakerClusterer(feature_dim=hidden_dim, clusters=self.config.speaker_cluster_count)

        self.keyword_extractor = KeywordExtractor(
            sample_rate=self.config.sample_rate,
            window_seconds=self.config.keyword_window_seconds,
            stride_seconds=self.config.keyword_stride_seconds,
        )
        self.keywords_cache: List[KeywordCandidate] = []

        self.frame_buffer = np.zeros(0, dtype=np.float32)
        self.frame_index = 0
        self.last_emit_timestamp = 0.0
        history_frames = max(10, int(self.config.activity_history_seconds * self.config.frames_per_second))
        self.layer_buffers: Dict[str, Deque[dict]] = {
            "L2": deque(maxlen=history_frames),
            "L6": deque(maxlen=history_frames),
            "L10": deque(maxlen=history_frames),
        }
        self.audio_history: Deque[dict] = deque(maxlen=history_frames)

        hop_seconds = self.config.hop_size / self.config.sample_rate
        tau = max(1e-3, self.config.ema_tau_seconds)
        self.ema_alpha = 1.0 - math.exp(-hop_seconds / tau)

        self.rms_ema = math.nan
        self.pitch_ema = math.nan
        self.semantic_ema = math.nan

        self.rms_tracker = RollingStat(window_seconds=3.0, sample_rate=self.config.frames_per_second)
        self.pitch_tracker = RollingStat(window_seconds=3.0, sample_rate=self.config.frames_per_second)
        self.semantic_tracker = RollingStat(window_seconds=3.0, sample_rate=self.config.frames_per_second)
        self.l6_tracker = RollingStat(window_seconds=3.0, sample_rate=self.config.frames_per_second)
        self.palette = [
            "#6366f1",
            "#22d3ee",
            "#f97316",
            "#a855f7",
            "#ef4444",
            "#10b981",
            "#facc15",
            "#818cf8",
            "#fb7185",
        ]
        self.default_color = "#94a3b8"
        self.speaker_colors: Dict[int, str] = {}
        self.next_color_index = 0

        self.mode = "analysis"

        torch.set_grad_enabled(False)

    # ------------------------------------------------------------------
    def process_samples(self, samples: np.ndarray) -> Optional[dict]:
        if samples.ndim != 1:
            samples = np.asarray(samples).reshape(-1)
        samples = samples.astype(np.float32)
        self.frame_buffer = np.concatenate([self.frame_buffer, samples])
        frame_size = self.config.frame_size
        hop_size = self.config.hop_size

        emitted = None
        last_ts = None
        while self.frame_buffer.shape[0] >= frame_size:
            frame = self.frame_buffer[:frame_size]
            self.frame_buffer = self.frame_buffer[hop_size:]
            timestamp = self.frame_index * (self.config.hop_size / self.config.sample_rate)
            self._process_frame(frame, timestamp)
            self.frame_index += 1
            last_ts = timestamp

        if last_ts is not None and last_ts - self.last_emit_timestamp >= self.config.refresh_interval:
            emitted = self._build_payload(last_ts)
            self.last_emit_timestamp = last_ts
        return emitted

    # ------------------------------------------------------------------
    def _process_frame(self, frame: np.ndarray, timestamp: float) -> None:
        rms_value = rms(frame)
        pitch_value = estimate_pitch_yin(frame, self.config.sample_rate)

        self.rms_tracker.update(rms_value)
        self.pitch_tracker.update(pitch_value)

        self.rms_ema = ema_update(self.rms_ema, rms_value, self.ema_alpha)
        self.pitch_ema = ema_update(self.pitch_ema, pitch_value, self.ema_alpha)

        # Model inference
        inputs = self.processor(frame, sampling_rate=self.config.sample_rate, return_tensors="pt", padding=False)
        input_values = inputs["input_values"].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        l2_vec = hidden_states[2].mean(dim=1).squeeze(0).cpu().numpy()
        l6_vec = hidden_states[6].mean(dim=1).squeeze(0).cpu().numpy()
        l10_vec = hidden_states[10].mean(dim=1).squeeze(0).cpu().numpy()

        semantic_intensity = float(np.linalg.norm(l10_vec))
        self.semantic_tracker.update(semantic_intensity)
        self.semantic_ema = ema_update(self.semantic_ema, semantic_intensity, self.ema_alpha)

        l6_intensity = float(np.linalg.norm(l6_vec))
        self.l6_tracker.update(l6_intensity)
        l6_norm = normalize(
            l6_intensity,
            self.l6_tracker.min(),
            self.l6_tracker.max() + 1e-6,
        )
        speaker_id = self.speaker_clusterer.update(l2_vec)

        self.keyword_extractor.append(frame)
        keywords = self.keyword_extractor.maybe_extract(timestamp)
        if keywords:
            # keep keywords for later payload building
            self.keywords_cache = keywords
        rms_norm = normalize(rms_value, self.rms_tracker.min(), self.rms_tracker.max() + 1e-6)
        pitch_norm = normalize(pitch_value, self.pitch_tracker.min(), self.pitch_tracker.max() + 1e-6)
        semantic_norm = normalize(
            semantic_intensity,
            self.semantic_tracker.min(),
            self.semantic_tracker.max() + 1e-6,
        )

        self.audio_history.append({"t": timestamp, "rms": rms_norm})

        self._record_activity("L2", timestamp, l2_vec, speaker_id, rms=rms_norm)
        self._record_activity("L6", timestamp, l6_vec, speaker_id, energy=l6_norm, pitch=pitch_norm)
        self._record_activity("L10", timestamp, l10_vec, speaker_id, semantics=semantic_norm)

    # ------------------------------------------------------------------
    def _record_activity(
        self,
        layer: str,
        timestamp: float,
        vector: np.ndarray,
        speaker_id: int,
        **extras: float,
    ) -> None:
        truncated = vector[: self.config.activity_neurons].astype(np.float32)
        if truncated.size == 0:
            return
        energies = np.abs(truncated)
        idx = int(np.argmax(energies))
        activity = float(truncated[idx])

        buffer = self.layer_buffers[layer]
        if buffer:
            prev = buffer[-1]
            idx = 0.7 * float(prev["index"]) + 0.3 * idx
            activity = 0.7 * float(prev["activity"]) + 0.3 * activity

        entry = {
            "t": timestamp,
            "index": float(idx),
            "activity": float(activity),
            "vector": truncated.tolist(),
            "speaker": int(speaker_id) if speaker_id >= 0 else None,
        }
        entry.update({k: float(v) for k, v in extras.items()})
        if entry["speaker"] is not None:
            self._color_for_speaker(entry["speaker"])
        buffer.append(entry)

    # ------------------------------------------------------------------
    def _color_for_speaker(self, speaker: int) -> str:
        if speaker is None or speaker < 0:
            return self.default_color
        if speaker not in self.speaker_colors:
            color = self.palette[self.next_color_index % len(self.palette)]
            self.speaker_colors[speaker] = color
            self.next_color_index += 1
        return self.speaker_colors[speaker]

    # ------------------------------------------------------------------
    def _build_payload(self, now_ts: float) -> dict:
        self.keywords_cache = [token for token in self.keywords_cache if token.expires_at > now_ts]
        keywords = self.keywords_cache
        if self.keyword_extractor._ready:
            transcript = self.keyword_extractor.last_transcript
        else:
            transcript = " ".join(token.text for token in keywords)
        keywords_payload = [
            {
                "text": token.text,
                "confidence": round(token.confidence, 2),
                "t": round(token.timestamp, 2),
                "expires": round(token.expires_at, 2),
            }
            for token in keywords
        ]

        projector_ready = self.semantic_projector is not None

        diagnostics = Diagnostics(
            projector_ready=projector_ready,
            speaker_ready=self.speaker_clusterer.is_ready,
            keyword_ready=self.keyword_extractor._ready,
        )

        layers_payload = []
        for name, buffer in self.layer_buffers.items():
            if not buffer:
                continue
            base_time = buffer[0]["t"]
            times = [round(entry["t"] - base_time, 3) for entry in buffer]
            indices = [round(entry["index"], 3) for entry in buffer]
            activities = [round(entry["activity"], 4) for entry in buffer]
        vectors = [entry["vector"] for entry in buffer]
            speakers = [entry.get("speaker") for entry in buffer]
            layers_payload.append(
                {
                    "name": name,
                    "times": times,
                    "indices": indices,
                    "activities": activities,
                    "vectors": vectors,
                    "speakers": speakers,
                }
            )

        audio_payload = {
            "times": [],
            "rms": [],
        }
        if self.audio_history:
            base_audio = self.audio_history[0]["t"]
            audio_payload = {
                "times": [round(entry["t"] - base_audio, 3) for entry in self.audio_history],
                "rms": [round(entry["rms"], 4) for entry in self.audio_history],
            }

        payload = {
            "type": "layer_activity",
            "timestamp": round(now_ts, 3),
            "layers": layers_payload,
            "audio": audio_payload,
            "meta": {
                "diagnostics": diagnostics.__dict__,
                "mode": self.mode,
                "transcript": transcript,
                "keywords": keywords_payload,
                "speaker_colors": {str(k): v for k, v in self.speaker_colors.items()},
            },
        }
        return payload

    # ------------------------------------------------------------------
    def to_json(self, payload: dict) -> str:
        return json.dumps(payload)

    # ------------------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        if mode not in {"analysis", "performance"}:
            raise ValueError("mode must be 'analysis' or 'performance'")
        self.mode = mode

    # ------------------------------------------------------------------
    def reset(self) -> None:
        logger.info("Resetting pipeline state")
        self.frame_buffer = np.zeros(0, dtype=np.float32)
        self.frame_index = 0
        self.rms_ema = math.nan
        self.pitch_ema = math.nan
        self.semantic_ema = math.nan
        self.rms_tracker.reset()
        self.pitch_tracker.reset()
        self.semantic_tracker.reset()
        self.l6_tracker.reset()
        for buffer in self.layer_buffers.values():
            buffer.clear()
        self.audio_history.clear()
        self.keywords_cache.clear()
        if hasattr(self.keyword_extractor, "_buffer"):
            self.keyword_extractor._buffer = np.zeros(0, dtype=np.float32)
            self.keyword_extractor._tokens = []
            if hasattr(self.keyword_extractor, "_transcript_history"):
                self.keyword_extractor._transcript_history = []
                self.keyword_extractor.last_transcript = ""
                self.keyword_extractor._prev_transcript = ""
