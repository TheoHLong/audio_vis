from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA


@dataclass
class FrozenSemanticProjector:
    """Deterministic semantic projector trained offline."""

    artifact_path: str

    def __post_init__(self) -> None:
        blob = np.load(self.artifact_path)
        self.feature_mean = blob["feature_mean"]
        self.feature_scale = blob["feature_scale"]
        self.ridge_coef = blob["ridge_coef"]
        self.ridge_intercept = blob["ridge_intercept"]
        self.semantic_pca_components = blob["semantic_pca_components"]
        self.semantic_pca_mean = blob["semantic_pca_mean"]

    def transform(self, feature_vector: np.ndarray) -> np.ndarray:
        z = (feature_vector - self.feature_mean) / (self.feature_scale + 1e-8)
        semantic = z @ self.ridge_coef.T + self.ridge_intercept
        coords = (semantic - self.semantic_pca_mean) @ self.semantic_pca_components.T
        return coords.astype(np.float32)


@dataclass
class IncrementalProjector:
    """Incremental PCA projector that streams high-d vectors to 2D."""

    input_dim: int
    primary_components: int = 32
    display_components: int = 2
    batch_size: int = 128

    primary: IncrementalPCA = field(init=False)
    display: IncrementalPCA = field(init=False)

    _primary_ready: bool = field(default=False, init=False)
    _display_ready: bool = field(default=False, init=False)
    _primary_buffer: list[np.ndarray] = field(default_factory=list, init=False)
    _display_buffer: list[np.ndarray] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self.primary = IncrementalPCA(n_components=min(self.primary_components, self.input_dim))
        self.display = IncrementalPCA(n_components=min(self.display_components, self.primary_components))
        if self.batch_size < self.primary.n_components * 3:
            self.batch_size = self.primary.n_components * 3

    def update(self, vector: np.ndarray) -> np.ndarray:
        vector = vector.reshape(1, -1)
        self._primary_buffer.append(vector)
        if len(self._primary_buffer) >= self.batch_size:
            stacked = np.vstack(self._primary_buffer)
            self.primary.partial_fit(stacked)
            self._primary_ready = True
            self._primary_buffer.clear()

        if self._primary_ready:
            reduced = self.primary.transform(vector)
            self._display_buffer.append(reduced)
            if len(self._display_buffer) >= max(self.batch_size // 2, self.display.n_components * 3):
                stacked = np.vstack(self._display_buffer)
                self.display.partial_fit(stacked)
                self._display_ready = True
                self._display_buffer.clear()

            if self._display_ready:
                projected = self.display.transform(reduced)[0]
            else:
                projected = reduced[0, : self.display.n_components]
                projected = self._pad_if_needed(projected)
        else:
            projected = vector[0, : self.display.n_components]
            projected = self._pad_if_needed(projected)

        return projected

    def _pad_if_needed(self, vector: np.ndarray) -> np.ndarray:
        if vector.shape[0] >= self.display.n_components:
            return vector[: self.display.n_components]
        return np.pad(vector, (0, self.display.n_components - vector.shape[0]))


@dataclass
class SpeakerClusterer:
    feature_dim: int
    clusters: int = 8
    batch_size: int = 256
    seed: int = 17

    model: MiniBatchKMeans = field(init=False)
    _buffer: list[np.ndarray] = field(default_factory=list, init=False)
    _is_ready: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.model = MiniBatchKMeans(
            n_clusters=self.clusters,
            random_state=self.seed,
            reassignment_ratio=0.01,
            init="k-means++",
        )

    def update(self, vector: np.ndarray) -> int:
        vector = vector.reshape(1, -1)
        self._buffer.append(vector)
        if len(self._buffer) >= self.batch_size:
            stacked = np.vstack(self._buffer)
            if stacked.shape[0] >= self.clusters:
                self.model.partial_fit(stacked)
                self._is_ready = True
            self._buffer.clear()
        if self._is_ready:
            speaker = int(self.model.predict(vector)[0])
        else:
            speaker = -1
        return speaker

    def predict(self, vector: np.ndarray) -> int:
        if not self._is_ready:
            return -1
        return int(self.model.predict(vector.reshape(1, -1))[0])

    @property
    def is_ready(self) -> bool:
        return self._is_ready
