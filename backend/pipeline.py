from __future__ import annotations

import json
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor

from .config import PipelineConfig
from .emotion import EmotionAnalyzer, EmotionState
from .keywords import KeywordCandidate, KeywordExtractor
from .projection import IncrementalProjector, SpeakerClusterer
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
class FrameDescriptor:
    index: int
    timestamp: float
    xy: np.ndarray
    speaker: int
    rms: float
    rms_norm: float
    pitch: float
    pitch_norm: float
    semantic_intensity: float
    semantic_norm: float
    tail_alpha: float
    line_width: float
    glow: float
    particle: bool


@dataclass
class Diagnostics:
    projector_ready: bool
    speaker_ready: bool
    keyword_ready: bool
    emotion_ready: bool


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
        self.projector = IncrementalProjector(
            input_dim=hidden_dim,
            primary_components=self.config.primary_pca_components,
            display_components=self.config.display_components,
        )
        self.speaker_clusterer = SpeakerClusterer(feature_dim=hidden_dim, clusters=self.config.speaker_cluster_count)

        self.keyword_extractor = KeywordExtractor(
            sample_rate=self.config.sample_rate,
            window_seconds=self.config.keyword_window_seconds,
            stride_seconds=self.config.keyword_stride_seconds,
        )
        self.emotion_analyzer = EmotionAnalyzer(
            sample_rate=self.config.sample_rate,
            window_seconds=self.config.emotion_window_seconds,
            update_seconds=self.config.emotion_update_seconds,
        )
        self.keywords_cache: List[KeywordCandidate] = []

        self.frame_buffer = np.zeros(0, dtype=np.float32)
        self.frame_index = 0
        self.tail: Deque[FrameDescriptor] = deque(maxlen=int(self.config.frames_per_second * self.config.tail_seconds * 3))
        self.last_emit_timestamp = 0.0

        hop_seconds = self.config.hop_size / self.config.sample_rate
        tau = max(1e-3, self.config.ema_tau_seconds)
        self.ema_alpha = 1.0 - math.exp(-hop_seconds / tau)
        self.position_state: Optional[np.ndarray] = None
        self.position_range = np.ones(self.projector.display_components) * 1e-2

        self.rms_ema = math.nan
        self.pitch_ema = math.nan
        self.semantic_ema = math.nan
        self.prev_rms = None

        self.rms_tracker = RollingStat(window_seconds=3.0, sample_rate=self.config.frames_per_second)
        self.pitch_tracker = RollingStat(window_seconds=3.0, sample_rate=self.config.frames_per_second)
        self.semantic_tracker = RollingStat(window_seconds=3.0, sample_rate=self.config.frames_per_second)

        self.palette = [
            "#60a5fa",
            "#34d399",
            "#f97316",
            "#a855f7",
            "#ef4444",
            "#10b981",
            "#facc15",
            "#818cf8",
            "#fb7185",
        ]
        self.default_color = "#4b5563"
        self.speaker_colors: Dict[int, str] = {}
        self.next_color_index = 0

        self.current_emotion = EmotionState()
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
        while self.frame_buffer.shape[0] >= frame_size:
            frame = self.frame_buffer[:frame_size]
            self.frame_buffer = self.frame_buffer[hop_size:]
            descriptor = self._process_frame(frame)
            if descriptor:
                self.tail.append(descriptor)
                self._trim_tail(descriptor.timestamp)
            self.frame_index += 1

        if self.tail:
            now_ts = self.tail[-1].timestamp
            if now_ts - self.last_emit_timestamp >= self.config.refresh_interval:
                emitted = self._build_payload(now_ts)
                self.last_emit_timestamp = now_ts
        return emitted

    # ------------------------------------------------------------------
    def _process_frame(self, frame: np.ndarray) -> Optional[FrameDescriptor]:
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

        projected = self.projector.update(l6_vec)
        self.position_range = np.maximum(self.position_range * 0.995, np.abs(projected) + 1e-3)
        normalized = projected / self.position_range
        if self.position_state is None:
            self.position_state = normalized
        else:
            self.position_state = (1.0 - self.ema_alpha) * self.position_state + self.ema_alpha * normalized

        speaker_id = self.speaker_clusterer.update(l2_vec)

        rms_norm = normalize(rms_value, self.rms_tracker.min(), self.rms_tracker.max() + 1e-6)
        pitch_norm = normalize(pitch_value, self.pitch_tracker.min(), self.pitch_tracker.max() + 1e-6)
        semantic_norm = normalize(
            semantic_intensity,
            self.semantic_tracker.min(),
            self.semantic_tracker.max() + 1e-6,
        )

        tail_alpha = softclip(0.25 + semantic_norm * 0.75, 0.2, 1.0)
        line_width = softclip(0.8 + (rms_norm ** 1.4) * 3.2, 0.8, 6.0)
        glow = softclip(0.3 + pitch_norm * 0.8, 0.2, 1.0)

        particle = False
        if self.prev_rms is not None and self.rms_ema and not math.isnan(self.rms_ema):
            surge = (rms_value - self.rms_ema) / (self.rms_ema + 1e-6)
            if surge > 0.8 and rms_value > self.prev_rms * 1.1:
                particle = True
        self.prev_rms = rms_value

        timestamp = self.frame_index * (self.config.hop_size / self.config.sample_rate)

        self.keyword_extractor.append(frame)
        keywords = self.keyword_extractor.maybe_extract(timestamp)
        if keywords:
            # keep keywords for later payload building
            self.keywords_cache = keywords

        descriptor = FrameDescriptor(
            index=self.frame_index,
            timestamp=timestamp,
            xy=self.position_state.copy(),
            speaker=speaker_id,
            rms=rms_value,
            rms_norm=rms_norm,
            pitch=pitch_value,
            pitch_norm=pitch_norm,
            semantic_intensity=semantic_intensity,
            semantic_norm=semantic_norm,
            tail_alpha=tail_alpha,
            line_width=line_width,
            glow=glow,
            particle=particle,
        )

        heuristic = self._heuristic_emotion(rms_norm, pitch_norm, semantic_norm)
        self.current_emotion = self.emotion_analyzer.update(frame, timestamp, heuristic)

        return descriptor

    # ------------------------------------------------------------------
    def _trim_tail(self, timestamp: float) -> None:
        threshold = timestamp - self.config.tail_seconds
        while self.tail and self.tail[0].timestamp < threshold:
            self.tail.popleft()

    # ------------------------------------------------------------------
    def _color_for_speaker(self, speaker: int) -> str:
        if speaker < 0:
            return self.default_color
        if speaker not in self.speaker_colors:
            color = self.palette[self.next_color_index % len(self.palette)]
            self.speaker_colors[speaker] = color
            self.next_color_index += 1
        return self.speaker_colors[speaker]

    # ------------------------------------------------------------------
    def _heuristic_emotion(self, rms_norm: float, pitch_norm: float, semantic_norm: float) -> EmotionState:
        arousal = softclip(0.35 + rms_norm * 0.65, 0.0, 1.0)
        valence = softclip(0.45 + (pitch_norm - 0.5) * 0.3 + (semantic_norm - 0.5) * 0.25, 0.0, 1.0)
        label = "heuristic"
        confidence = 0.15 + abs(valence - 0.5) * 0.3 + abs(arousal - 0.5) * 0.3
        return EmotionState(valence=valence, arousal=arousal, label=label, confidence=confidence)

    # ------------------------------------------------------------------
    def _build_payload(self, now_ts: float) -> dict:
        frames_payload = []
        particles = []
        for frame in self.tail:
            payload = {
                "i": frame.index,
                "t": round(frame.timestamp, 3),
                "x": round(float(frame.xy[0]), 4),
                "y": round(float(frame.xy[1]), 4),
                "speaker": frame.speaker,
                "color": self._color_for_speaker(frame.speaker),
                "width": round(frame.line_width, 3),
                "alpha": round(frame.tail_alpha, 3),
                "glow": round(frame.glow, 3),
                "rms": round(frame.rms_norm, 3),
                "pitch": round(frame.pitch, 2),
                "pitch_norm": round(frame.pitch_norm, 3),
                "semantic": round(frame.semantic_norm, 3),
            }
            frames_payload.append(payload)
            if frame.particle:
                particles.append({
                    "i": frame.index,
                    "t": payload["t"],
                    "x": payload["x"],
                    "y": payload["y"],
                    "color": payload["color"],
                    "semantic": payload["semantic"],
                    "pitch_norm": payload["pitch_norm"],
                })

        head = frames_payload[-1] if frames_payload else None
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

        diagnostics = Diagnostics(
            projector_ready=self.projector._display_ready,
            speaker_ready=self.speaker_clusterer.is_ready,
            keyword_ready=self.keyword_extractor._ready,
            emotion_ready=self.emotion_analyzer.is_ready,
        )

        payload = {
            "type": "frame_batch",
            "timestamp": round(now_ts, 3),
            "frames": frames_payload,
            "head": head,
            "particles": particles,
            "keywords": keywords_payload,
            "transcript": transcript,
            "emotion": {
                "valence": round(self.current_emotion.valence, 3),
                "arousal": round(self.current_emotion.arousal, 3),
                "label": self.current_emotion.label,
                "confidence": round(self.current_emotion.confidence, 3),
            },
            "diagnostics": diagnostics.__dict__,
            "mode": self.mode,
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
        self.tail.clear()
        self.frame_buffer = np.zeros(0, dtype=np.float32)
        self.frame_index = 0
        self.position_state = None
        self.position_range = np.ones(self.projector.display_components) * 1e-2
        self.rms_ema = math.nan
        self.pitch_ema = math.nan
        self.semantic_ema = math.nan
        self.prev_rms = None
        self.rms_tracker.reset()
        self.pitch_tracker.reset()
        self.semantic_tracker.reset()
        self.current_emotion = EmotionState()
        self.keywords_cache.clear()
        if hasattr(self.keyword_extractor, "_buffer"):
            self.keyword_extractor._buffer = np.zeros(0, dtype=np.float32)
            self.keyword_extractor._tokens = []
            if hasattr(self.keyword_extractor, "_transcript_history"):
                self.keyword_extractor._transcript_history = []
                self.keyword_extractor.last_transcript = ""
                self.keyword_extractor._prev_transcript = ""
        if hasattr(self.emotion_analyzer, "reset"):
            self.emotion_analyzer.reset()
