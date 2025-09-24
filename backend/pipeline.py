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
from .projection import FrozenSemanticProjector, IncrementalProjector, SpeakerClusterer
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
    emotion_level: float
    tail_alpha: float
    line_width: float
    glow: float
    particle: bool
    acoustic_signature: np.ndarray


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

        self.projector = None
        if self.semantic_projector is None:
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
        self.keywords_cache: List[KeywordCandidate] = []

        self.frame_buffer = np.zeros(0, dtype=np.float32)
        self.frame_index = 0
        self.tail: Deque[FrameDescriptor] = deque(maxlen=int(self.config.frames_per_second * self.config.tail_seconds * 3))
        self.last_emit_timestamp = 0.0
        self.stars: Deque[dict] = deque(maxlen=120)
        self.last_star_timestamp = -1.0
        self.rhythm_pulses: Deque[dict] = deque(maxlen=80)

        hop_seconds = self.config.hop_size / self.config.sample_rate
        tau = max(1e-3, self.config.ema_tau_seconds)
        self.ema_alpha = 1.0 - math.exp(-hop_seconds / tau)
        self.position_state: Optional[np.ndarray] = None
        if self.semantic_projector is not None:
            display_dim = self.semantic_projector.semantic_pca_components.shape[0]
        elif self.projector is not None:
            display_dim = self.projector.display.n_components
        else:
            display_dim = self.config.display_components
        self.position_range = np.ones(display_dim) * 1e-2

        self.rms_ema = math.nan
        self.pitch_ema = math.nan
        self.semantic_ema = math.nan
        self.prev_rms = None

        self.rms_tracker = RollingStat(window_seconds=3.0, sample_rate=self.config.frames_per_second)
        self.pitch_tracker = RollingStat(window_seconds=3.0, sample_rate=self.config.frames_per_second)
        self.semantic_tracker = RollingStat(window_seconds=3.0, sample_rate=self.config.frames_per_second)
        self.l6_tracker = RollingStat(window_seconds=3.0, sample_rate=self.config.frames_per_second)

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

        l6_intensity = float(np.linalg.norm(l6_vec))
        self.l6_tracker.update(l6_intensity)
        l6_norm = normalize(
            l6_intensity,
            self.l6_tracker.min(),
            self.l6_tracker.max() + 1e-6,
        )
        l2_norm = np.linalg.norm(l2_vec) + 1e-6
        acoustic_signature = (l2_vec / l2_norm)[:8].astype(np.float32)

        if self.semantic_projector is not None:
            projected = self.semantic_projector.transform(l10_vec)
        else:
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
            emotion_level=l6_norm,
            tail_alpha=tail_alpha,
            line_width=line_width,
            glow=glow,
            particle=particle,
            acoustic_signature=acoustic_signature,
        )

        self._maybe_add_star(descriptor)
        self._update_rhythm(descriptor)

        return descriptor

    # ------------------------------------------------------------------
    def _trim_tail(self, timestamp: float) -> None:
        threshold = timestamp - self.config.tail_seconds
        while self.tail and self.tail[0].timestamp < threshold:
            self.tail.popleft()

    # ------------------------------------------------------------------
    def _maybe_add_star(self, descriptor: FrameDescriptor) -> None:
        interval = max(1e-3, self.config.star_interval_seconds)
        if self.last_star_timestamp >= 0.0 and descriptor.timestamp - self.last_star_timestamp < interval:
            return

        theme = None
        if self.keywords_cache:
            theme = self.keywords_cache[-1].text

        color = self._color_from_emotion_level(descriptor.emotion_level)
        brightness = round(0.55 + descriptor.semantic_norm * 0.35, 4)
        star = {
            "id": int(descriptor.index),
            "t": round(descriptor.timestamp, 3),
            "x": round(float(descriptor.xy[0]), 4),
            "y": round(float(descriptor.xy[1]), 4),
            "brightness": brightness,
            "twinkle": round(softclip(descriptor.pitch_norm, 0.0, 1.0), 4),
            "semantic": round(softclip(descriptor.semantic_norm, 0.0, 1.0), 4),
            "color": color,
            "theme": theme,
            "emotion_level": round(descriptor.emotion_level, 3),
            "speaker": descriptor.speaker,
            "acoustic": descriptor.acoustic_signature.tolist(),
        }
        self.stars.append(star)
        self.last_star_timestamp = descriptor.timestamp

    # ------------------------------------------------------------------
    def _update_rhythm(self, descriptor: FrameDescriptor) -> None:
        strength = descriptor.rms_norm
        if self.prev_rms is None:
            return
        if strength > 0.45 and strength > self.prev_rms + 0.05:
            self.rhythm_pulses.append({
                "t": round(descriptor.timestamp, 3),
                "strength": round(softclip(strength, 0.0, 1.0), 3),
            })

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
    def _color_from_emotion_level(self, level: float) -> str:
        lvl = softclip(level, 0.0, 1.0)
        calm = np.array([88, 162, 255])
        mid = np.array([255, 205, 102])
        intense = np.array([255, 80, 80])
        if lvl < 0.5:
            alpha = lvl / 0.5
            rgb = (1 - alpha) * calm + alpha * mid
        else:
            alpha = (lvl - 0.5) / 0.5
            rgb = (1 - alpha) * mid + alpha * intense
        rgb = np.clip(rgb, 0, 255).astype(int)
        return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

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

        projector_ready = True if self.semantic_projector is not None else (self.projector._display_ready if self.projector else False)

        diagnostics = Diagnostics(
            projector_ready=projector_ready,
            speaker_ready=self.speaker_clusterer.is_ready,
            keyword_ready=self.keyword_extractor._ready,
        )

        stars_list = list(self.stars)[-120:]
        connections = self._build_connections(stars_list)
        nebula = self._build_nebula(np.mean([star.get("emotion_level", 0.5) for star in stars_list[-20:]] or [0.5]))
        themes = self._collect_themes(stars_list)
        rhythm = list(self.rhythm_pulses)[-40:]

        payload = {
            "type": "constellation",
            "timestamp": round(now_ts, 3),
            "stars": stars_list,
            "connections": connections,
            "nebula": nebula,
            "rhythm": rhythm,
            "meta": {
                "diagnostics": diagnostics.__dict__,
                "mode": self.mode,
                "transcript": transcript,
                "keywords": keywords_payload,
                "themes": themes,
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
        self.tail.clear()
        self.frame_buffer = np.zeros(0, dtype=np.float32)
        self.frame_index = 0
        self.position_state = None
        if self.semantic_projector is not None:
            display_dim = self.semantic_projector.semantic_pca_components.shape[0]
        elif self.projector is not None:
            display_dim = self.projector.display.n_components
        else:
            display_dim = self.config.display_components
        self.position_range = np.ones(display_dim) * 1e-2
        self.rms_ema = math.nan
        self.pitch_ema = math.nan
        self.semantic_ema = math.nan
        self.prev_rms = None
        self.rms_tracker.reset()
        self.pitch_tracker.reset()
        self.semantic_tracker.reset()
        self.keywords_cache.clear()
        if hasattr(self.keyword_extractor, "_buffer"):
            self.keyword_extractor._buffer = np.zeros(0, dtype=np.float32)
            self.keyword_extractor._tokens = []
            if hasattr(self.keyword_extractor, "_transcript_history"):
                self.keyword_extractor._transcript_history = []
                self.keyword_extractor.last_transcript = ""
                self.keyword_extractor._prev_transcript = ""
        self.stars.clear()
        self.last_star_timestamp = -1.0
        self.rhythm_pulses.clear()

    # ------------------------------------------------------------------
    def _build_connections(self, stars: List[dict]) -> List[dict]:
        if len(stars) < 2:
            return []
        recent = stars[-60:]
        signatures = []
        for star in recent:
            sig = np.asarray(star.get("acoustic"), dtype=np.float32)
            if sig.size == 0:
                sig = None
            signatures.append(sig)

        connections = []
        added = set()
        for idx, star in enumerate(recent):
            sig_i = signatures[idx]
            if sig_i is None:
                continue
            sims: list[tuple[int, float]] = []
            norm_i = float(np.linalg.norm(sig_i)) + 1e-8
            for jdx, other in enumerate(recent):
                if idx == jdx:
                    continue
                sig_j = signatures[jdx]
                if sig_j is None:
                    continue
                dot = float(np.dot(sig_i, sig_j))
                norm_j = float(np.linalg.norm(sig_j)) + 1e-8
                sim = dot / (norm_i * norm_j)
                if star.get("speaker") is not None and star.get("speaker") == other.get("speaker"):
                    sim += 0.2
                sims.append((jdx, sim))
            sims.sort(key=lambda item: item[1], reverse=True)
            for jdx, sim in sims[:2]:
                if sim <= 0:
                    continue
                a = star["id"]
                b = recent[jdx]["id"]
                key = tuple(sorted((a, b)))
                if key in added:
                    continue
                strength = softclip((sim + 1) / 2, 0.0, 1.0)
                connections.append({
                    "source": a,
                    "target": b,
                    "strength": round(strength, 3),
                })
                added.add(key)
        return connections

    # ------------------------------------------------------------------
    def _build_nebula(self, emotion_level: float) -> dict:
        level = softclip(emotion_level, 0.0, 1.0)
        hue = 220 - level * 180
        intensity = 0.2 + level * 0.55
        return {
            "hue": round(float(hue), 3),
            "intensity": round(float(intensity), 3),
            "level": round(level, 3),
        }

    # ------------------------------------------------------------------
    def _collect_themes(self, stars: List[dict]) -> List[dict]:
        themes: Dict[str, int] = {}
        for star in stars[-100:]:
            theme = star.get("theme")
            if not theme:
                continue
            key = theme.lower()
            themes[key] = themes.get(key, 0) + 1
        top = sorted(themes.items(), key=lambda item: item[1], reverse=True)[:5]
        return [{"text": theme, "count": count} for theme, count in top]
