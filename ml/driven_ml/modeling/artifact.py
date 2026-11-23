"""The exported model: a plain JSON file the phone can run without any ML library.

It holds the input order, the NaN fill values, the scaler, the weights, the calibration temperature, the alert
threshold and cooldown, and a sha256 over all of it. ``predict_probability`` is the reference forward pass:
the on-device port must match it to 1e-6 on the parity fixtures.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import numpy.typing as npt

from ..errors import DataError
from ..settings import DataSource, Track
from .calibration import sigmoid

RELU = "relu"
LINEAR = "linear"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class LogisticRegressionWeights:
    coefficients: list[float]
    intercept: float
    type: str = "logistic_regression"


@dataclass(frozen=True)
class DenseLayer:
    weights: list[list[float]]
    bias: list[float]
    activation: str


@dataclass(frozen=True)
class MlpWeights:
    layers: list[DenseLayer]
    type: str = "mlp"


@dataclass(frozen=True)
class ModelArtifact:
    environment: str
    track: str
    data_source: str
    datasets: list[str]
    inputs: list[str]
    fill_values: list[float]
    scaler_mean: list[float]
    scaler_scale: list[float]
    weights: LogisticRegressionWeights | MlpWeights
    temperature: float
    threshold: float
    cooldown_seconds: float
    fixed_threshold_baseline: float
    feature_spec_version: int
    created_at: str = field(default_factory=_utc_now)
    name: str = "driven-drowsiness"
    classes: tuple[str, ...] = ("alert", "drowsy")

    def _body(self) -> dict[str, Any]:
        body = asdict(self)
        body["classes"] = list(self.classes)
        return body

    @property
    def sha256(self) -> str:
        """Hash of everything that decides predictions; ``created_at`` is left out, so retraining the same model
        on the same data gives the same hash and cannot reset its locked test result."""
        content = {key: value for key, value in self._body().items() if key != "created_at"}
        encoded = json.dumps(content, sort_keys=True, allow_nan=False).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def short_sha(self) -> str:
        return self.sha256[:12]

    @property
    def is_shippable_source(self) -> bool:
        """Only real data on the commercial track can ever ship (L9)."""
        return self.data_source == DataSource.REAL and self.track == Track.COMMERCIAL

    def to_json(self) -> dict[str, Any]:
        return {**self._body(), "sha256": self.sha256}

    @classmethod
    def from_json(cls, document: Mapping[str, Any]) -> ModelArtifact:
        body = dict(document)
        claimed_sha = body.pop("sha256", None)
        try:
            body["weights"] = _weights_from_json(body["weights"])
            body["classes"] = tuple(body["classes"])
            artifact = cls(**body)
        except (AttributeError, KeyError, TypeError) as exc:
            raise DataError(f"model file is malformed: {exc!r}") from exc
        if claimed_sha is not None and claimed_sha != artifact.sha256:
            raise DataError("model file was modified after export (sha256 mismatch)")
        return artifact


def _weights_from_json(document: Mapping[str, Any]) -> LogisticRegressionWeights | MlpWeights:
    kind = document.get("type")
    if kind == "logistic_regression":
        return LogisticRegressionWeights(**document)
    if kind == "mlp":
        return MlpWeights(layers=[DenseLayer(**layer) for layer in document["layers"]], type="mlp")
    raise DataError(f"unknown model weights type {kind!r}")


def predict_probability(
    artifact: ModelArtifact,
    features: npt.ArrayLike,
    *,
    temperature: float | None = None,
) -> np.ndarray:
    """Calibrated drowsiness probability per row. ``features`` columns follow ``artifact.inputs``."""
    x = np.atleast_2d(np.asarray(features, dtype=float))
    if x.shape[1] != len(artifact.inputs):
        raise ValueError(f"expected {len(artifact.inputs)} input columns, got {x.shape[1]}")
    x = np.where(np.isnan(x), np.asarray(artifact.fill_values), x)
    x = (x - np.asarray(artifact.scaler_mean)) / np.asarray(artifact.scaler_scale)
    weights = artifact.weights
    if isinstance(weights, LogisticRegressionWeights):
        logits = x @ np.asarray(weights.coefficients) + weights.intercept
    else:
        for layer in weights.layers:
            x = x @ np.asarray(layer.weights) + np.asarray(layer.bias)
            if layer.activation == RELU:
                x = np.maximum(x, 0.0)
        logits = x.ravel()
    return sigmoid(logits / (artifact.temperature if temperature is None else temperature))
