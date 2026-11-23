"""Temperature scaling (Guo et al., 2017): one number that makes the model's probabilities honest.

The temperature is fitted on out-of-fold logits, never in-fold ones, so it corrects for the overconfidence a
model shows on people it has not seen.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

TEMPERATURE_GRID = np.exp(np.linspace(np.log(0.2), np.log(5.0), 120))
_PROBABILITY_FLOOR = 1e-6


def sigmoid(values: npt.ArrayLike) -> np.ndarray:
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(-np.asarray(values, dtype=float)))


def log_loss(probabilities: npt.ArrayLike, labels: npt.ArrayLike) -> float:
    p = np.clip(
        np.asarray(probabilities, dtype=float),
        _PROBABILITY_FLOOR,
        1 - _PROBABILITY_FLOOR,
    )
    y = np.asarray(labels, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(logits: npt.ArrayLike, labels: npt.ArrayLike) -> float:
    """The grid temperature T that minimizes the log loss of sigmoid(logits / T)."""
    scores = np.asarray(logits, dtype=float)
    losses = [log_loss(sigmoid(scores / temperature), labels) for temperature in TEMPERATURE_GRID]
    return float(TEMPERATURE_GRID[int(np.argmin(losses))])
