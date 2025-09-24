from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency load
    from transformers import pipeline
except Exception:  # pragma: no cover - transformers handles availability
    pipeline = None  # type: ignore

logger = logging.getLogger(__name__)


@dataclass
class EmotionState:
    valence: float = 0.5
    arousal: float = 0.5
    label: str = "neutral"
    confidence: float = 0.0


class EmotionAnalyzer:
    """Optional speech emotion recogniser using a Hugging Face pipeline."""

    LABEL_MAP: Dict[str, Tuple[float, float]] = {
        "anger": (0.2, 0.85),
        "disgust": (0.25, 0.45),
        "fear": (0.15, 0.9),
        "joy": (0.85, 0.7),
        "neutral": (0.55, 0.35),
        "sadness": (0.2, 0.2),
        "surprise": (0.65, 0.85),
        "calm": (0.6, 0.25),
        "happy": (0.85, 0.7),
        "angry": (0.2, 0.85),
        "fearful": (0.2, 0.9),
    }

    def __init__(
        self,
        sample_rate: int,
        window_seconds: float,
        update_seconds: float,
        model_name: str = "speechbrain/emotion-recognition-wav2vec2-large-960h",
    ) -> None:
        self.sample_rate = sample_rate
        self.window_seconds = max(0.6, window_seconds)
        self.update_seconds = max(0.6, update_seconds)
        self.window_size = int(self.sample_rate * self.window_seconds)
        self._buffer = np.zeros(0, dtype=np.float32)
        self._last_eval = 0.0
        self.state = EmotionState()
        self.model_name = model_name

        self._pipeline = None
        self._ready = False
        if pipeline is None:
            logger.info("Transformers pipeline unavailable; emotion analysis disabled")
            return
        try:
            self._pipeline = pipeline(
                "audio-classification",
                model=model_name,
                top_k=None,
            )
            self._ready = True
            logger.info("Emotion analyzer initialised with %s", model_name)
        except Exception as exc:  # pragma: no cover - model availability
            logger.warning("Emotion analyzer disabled (%s)", exc)
            self._pipeline = None
            self._ready = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    def reset(self) -> None:
        self._buffer = np.zeros(0, dtype=np.float32)
        self._last_eval = 0.0
        self.state = EmotionState()

    def update(
        self,
        samples: np.ndarray,
        timestamp: float,
        heuristic: EmotionState,
    ) -> EmotionState:
        if samples.ndim != 1:
            samples = np.asarray(samples).reshape(-1)
        self._buffer = np.concatenate([self._buffer, samples.astype(np.float32)])
        max_len = self.window_size * 3
        if self._buffer.shape[0] > max_len:
            self._buffer = self._buffer[-max_len:]

        if not self._ready or self._pipeline is None:
            self.state = heuristic
            return self.state

        if timestamp - self._last_eval < self.update_seconds:
            return self.state

        if self._buffer.shape[0] < self.window_size:
            self.state = heuristic
            return self.state

        window = self._buffer[-self.window_size :]
        try:
            results = self._pipeline(window, sampling_rate=self.sample_rate)
        except Exception as exc:  # pragma: no cover - runtime inference
            logger.warning("Emotion inference failed: %s", exc)
            self._ready = False
            self.state = heuristic
            return self.state

        valence = 0.0
        arousal = 0.0
        total = 0.0
        top_label = heuristic.label
        top_conf = heuristic.confidence

        for item in results:
            label = item.get("label", "").lower()
            score = float(item.get("score", 0.0))
            mapping = self.LABEL_MAP.get(label)
            if mapping is None:
                continue
            v, a = mapping
            valence += v * score
            arousal += a * score
            total += score
            if score > top_conf:
                top_label = label
                top_conf = score

        if total > 0:
            valence /= total
            arousal /= total
        else:
            valence = heuristic.valence
            arousal = heuristic.arousal

        self.state = EmotionState(
            valence=float(np.clip(valence, 0.0, 1.0)),
            arousal=float(np.clip(arousal, 0.0, 1.0)),
            label=top_label,
            confidence=float(np.clip(top_conf, 0.0, 1.0)),
        )
        self._last_eval = timestamp
        return self.state
