from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Optional

import numpy as np


def rms(frame: np.ndarray) -> float:
    """Root-mean-square energy."""
    return float(np.sqrt(np.mean(np.square(frame), dtype=np.float64)))


def ema_update(prev: float, new: float, alpha: float) -> float:
    if math.isnan(prev):
        return new
    return (1.0 - alpha) * prev + alpha * new


def softclip(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def normalize(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return 0.0
    return float((value - minimum) / (maximum - minimum))


def estimate_pitch_yin(frame: np.ndarray, sample_rate: int, fmin: float = 60.0, fmax: float = 500.0) -> float:
    """Very small YIN-like pitch estimator.

    Returns 0.0 when no stable pitch is detected."""

    frame = frame.astype(np.float64)
    frame = frame - np.mean(frame)
    if np.max(np.abs(frame)) < 1e-4:
        return 0.0

    max_period = int(sample_rate / fmin)
    min_period = int(sample_rate / fmax)
    frame_size = frame.shape[0]

    if frame_size < max_period + 2:
        return 0.0

    # Difference function
    diff = np.zeros(max_period)
    for tau in range(1, max_period):
        diff[tau] = np.sum((frame[: frame_size - tau] - frame[tau:]) ** 2)

    # Cumulative mean normalized difference
    cmnd = np.zeros_like(diff)
    running_sum = 0.0
    for tau in range(1, max_period):
        running_sum += diff[tau]
        cmnd[tau] = diff[tau] * tau / running_sum if running_sum > 0 else 1.0

    tau = min_period
    threshold = 0.1
    while tau < max_period:
        if cmnd[tau] < threshold:
            while tau + 1 < max_period and cmnd[tau + 1] < cmnd[tau]:
                tau += 1
            return float(sample_rate / tau)
        tau += 1

    return 0.0


@dataclass
class RollingStat:
    window_seconds: float
    sample_rate: float

    def __post_init__(self) -> None:
        self.window: int = int(self.window_seconds * self.sample_rate)
        self.samples: Deque[float] = deque(maxlen=self.window)

    def update(self, value: float) -> float:
        self.samples.append(value)
        return self.value

    @property
    def value(self) -> float:
        if not self.samples:
            return 0.0
        return float(np.mean(self.samples))

    def max(self) -> float:
        if not self.samples:
            return 0.0
        return float(np.max(self.samples))

    def min(self) -> float:
        if not self.samples:
            return 0.0
        return float(np.min(self.samples))

    def reset(self) -> None:
        self.samples.clear()


def take(iterable: Iterable, count: int):
    result = []
    for idx, item in enumerate(iterable):
        if idx >= count:
            break
        result.append(item)
    return result
