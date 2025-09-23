from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class KeywordCandidate:
    text: str
    confidence: float
    timestamp: float
    expires_at: float


@dataclass
class KeywordExtractor:
    sample_rate: int
    window_seconds: float = 0.9
    stride_seconds: float = 0.45
    max_keywords: int = 3

    _buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32), init=False)
    _tokens: List[KeywordCandidate] = field(default_factory=list, init=False)
    _transcript_history: List[Tuple[float, str]] = field(default_factory=list, init=False)
    last_transcript: str = field(default="", init=False)
    _prev_transcript: str = field(default="", init=False)
    _last_emit: float = field(default=0.0, init=False)
    _pipeline: Optional[object] = field(default=None, init=False)
    _ready: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        disable = math.isclose(self.window_seconds, 0.0)
        if disable:
            logger.info("KeywordExtractor disabled by configuration")
            return
        try:
            from transformers import pipeline

            self._pipeline = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-tiny.en",
                device=-1,
                generation_config={"task": "transcribe", "max_new_tokens": 64},
            )
            self._ready = True
            logger.info("KeywordExtractor initialised with openai/whisper-tiny.en")
        except Exception as exc:  # pragma: no cover - hardware/env dependent
            logger.warning("Keyword extraction disabled (%s)", exc)
            self._pipeline = None
            self._ready = False
            self.last_transcript = ""

    @property
    def window_size(self) -> int:
        return int(self.sample_rate * self.window_seconds)

    @property
    def stride_size(self) -> int:
        return int(self.sample_rate * self.stride_seconds)

    def append(self, samples: np.ndarray) -> None:
        if not self._ready:
            return
        self._buffer = np.concatenate([self._buffer, samples])
        # Cap buffer to 2 * window to avoid runaway memory
        max_len = self.window_size * 2
        if self._buffer.shape[0] > max_len:
            self._buffer = self._buffer[-max_len:]

    def maybe_extract(self, timestamp: float) -> List[KeywordCandidate]:
        if not self._ready:
            return []
        if self._buffer.shape[0] < self.window_size:
            return []
        if timestamp - self._last_emit < self.stride_seconds:
            return []
        window = self._buffer[-self.window_size :]
        try:
            result = self._pipeline({"array": window, "sampling_rate": self.sample_rate})
        except Exception as exc:  # pragma: no cover - runtime dependency
            logger.warning("Keyword extraction failed: %s", exc)
            return []
        text = result.get("text", "")
        cleaned = text.strip()
        if cleaned and cleaned != self._prev_transcript:
            self._transcript_history.append((timestamp, cleaned))
            self._prev_transcript = cleaned
        cutoff = timestamp - 12.0
        if cutoff > 0:
            self._transcript_history = [item for item in self._transcript_history if item[0] >= cutoff]
        self.last_transcript = " ".join(segment for _, segment in self._transcript_history).strip()
        keywords = self._extract_keywords(text)
        self._last_emit = timestamp
        expires_at = timestamp + 2.0
        candidates = [
            KeywordCandidate(text=word, confidence=conf, timestamp=timestamp, expires_at=expires_at)
            for word, conf in keywords
        ]
        self._tokens.extend(candidates)
        # Keep only recent tokens
        self._tokens = [token for token in self._tokens if token.expires_at > timestamp]
        # Return the latest ones (max_keywords)
        return self._tokens[-self.max_keywords :]

    def _extract_keywords(self, text: str) -> List[tuple[str, float]]:
        words = re.findall(r"[a-zA-Z']+", text.lower())
        unique = []
        seen = set()
        for word in words[::-1]:  # reverse to keep latest words first
            if word in seen:
                continue
            seen.add(word)
            if len(word) < 3:
                continue
            score = min(1.0, 0.5 + len(word) / 8.0)
            unique.append((word, score))
            if len(unique) >= self.max_keywords:
                break
        return unique
