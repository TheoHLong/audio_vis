from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Runtime configuration for the comet pipeline."""

    sample_rate: int = 16_000
    frame_ms: float = 40.0
    hop_ms: float = 20.0
    refresh_rate_hz: float = 6.0
    tail_seconds: float = 2.5
    ema_tau_seconds: float = 0.25
    warmup_seconds: float = 6.0

    primary_pca_components: int = 32
    display_components: int = 2
    speaker_cluster_count: int = 8

    keyword_window_seconds: float = 0.8
    keyword_stride_seconds: float = 0.4

    @property
    def frame_size(self) -> int:
        return int(self.sample_rate * (self.frame_ms / 1000.0))

    @property
    def hop_size(self) -> int:
        return int(self.sample_rate * (self.hop_ms / 1000.0))

    @property
    def frames_per_second(self) -> float:
        return 1000.0 / self.hop_ms

    @property
    def refresh_interval(self) -> float:
        return 1.0 / self.refresh_rate_hz
