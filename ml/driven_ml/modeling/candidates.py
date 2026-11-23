"""The candidate models and the fixed-threshold baseline they must beat.

Driven's claim is that comparing each driver with their own baseline beats a one-size threshold. So the
comparison is: logistic regression and a 16-unit MLP on features plus baseline deltas, the same logistic
regression without deltas (B3), and a fixed PERCLOS threshold (B1).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from ..spec import INPUTS_WITH_BASELINE, RAW_INPUTS


class EstimatorKind(StrEnum):
    LOGISTIC_REGRESSION = "logistic_regression"
    MLP = "mlp"


@dataclass(frozen=True)
class Candidate:
    name: str
    kind: EstimatorKind
    inputs: tuple[str, ...]


CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        "Driven logistic regression",
        EstimatorKind.LOGISTIC_REGRESSION,
        INPUTS_WITH_BASELINE,
    ),
    Candidate("Driven MLP", EstimatorKind.MLP, INPUTS_WITH_BASELINE),
    Candidate("B3 no personal baseline", EstimatorKind.LOGISTIC_REGRESSION, RAW_INPUTS),
)
FIXED_THRESHOLD_BASELINE = "B1 fixed PERCLOS threshold"
FIXED_THRESHOLD_GRID = np.linspace(0.01, 0.4, 80)

Estimator: TypeAlias = LogisticRegression | MLPClassifier


def build_estimator(kind: EstimatorKind, seed: int) -> Estimator:
    if kind is EstimatorKind.LOGISTIC_REGRESSION:
        return LogisticRegression(C=0.5, max_iter=2000)
    return MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-3, max_iter=1500, random_state=seed)


def decision_logits(estimator: Estimator, features: np.ndarray) -> np.ndarray:
    """Pre-sigmoid scores; calibration needs logits, not probabilities."""
    if isinstance(estimator, LogisticRegression):
        return np.asarray(estimator.decision_function(features), dtype=np.float64)
    activations = features
    for weights, bias in zip(estimator.coefs_[:-1], estimator.intercepts_[:-1], strict=True):
        activations = np.maximum(activations @ weights + bias, 0.0)
    return np.asarray(activations @ estimator.coefs_[-1] + estimator.intercepts_[-1], dtype=np.float64).ravel()


def fixed_threshold_alerts(windows: pd.DataFrame, perclos_threshold: float) -> np.ndarray:
    """B1 as 0/1 scores, so the same alert simulation can score it at threshold 0.5."""
    return (windows["perclos"].to_numpy(dtype=float) >= perclos_threshold).astype(float)
