"""Metrics by baseline scope and within one recording, the two ways the camera confound can hide.

``person`` scope compares a video with a baseline from a different video, so a camera change can pass for
drowsiness. ``session`` scope compares a drive with a calibration clip from the same mount. Within-recording
AUC pairs drowsy and alert windows only inside one drive clip, so it is the number that counts on real data.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .modeling.alert_simulation import alert_metrics
from .spec import CLIP_DRIVE


@dataclass(frozen=True)
class ScopeScores:
    model: str
    baseline_scope: str
    people: int
    windows: int
    auc: float
    balanced_accuracy: float
    false_alerts_per_hour: float
    drowsy_sessions_caught: float


@dataclass(frozen=True)
class WithinRecording:
    """Pooled pairs of (drowsy window, alert window) from the same drive clip; AUC is concordant / pairs."""

    sessions: int
    pairs: int
    concordant: float

    @property
    def auc(self) -> float:
        return self.concordant / self.pairs if self.pairs else float("nan")

    def __add__(self, other: WithinRecording) -> WithinRecording:
        return WithinRecording(
            self.sessions + other.sessions,
            self.pairs + other.pairs,
            self.concordant + other.concordant,
        )


NO_RECORDINGS = WithinRecording(sessions=0, pairs=0, concordant=0.0)


def _auc(labels: np.ndarray, ranking: np.ndarray) -> float:
    return float(roc_auc_score(labels, ranking)) if len(set(labels.tolist())) == 2 else float("nan")


def scores_by_scope(
    model: str,
    windows: pd.DataFrame,
    ranking: npt.ArrayLike,
    probabilities: npt.ArrayLike,
    threshold: float,
) -> list[ScopeScores]:
    """The usual four metrics for each ``baseline_scope`` present; AUC is NaN where a scope has one class."""
    ranking_values, probability_values = np.asarray(ranking, dtype=float), np.asarray(probabilities, dtype=float)
    results = []
    for scope, rows in windows.groupby("baseline_scope", sort=True).indices.items():
        subset = windows.iloc[rows]
        labels = subset["y"].to_numpy()
        metrics = alert_metrics(subset, probability_values[rows], threshold)
        results.append(
            ScopeScores(
                model=model,
                baseline_scope=str(scope),
                people=subset["subject"].nunique(),
                windows=len(subset),
                auc=_auc(labels, ranking_values[rows]),
                balanced_accuracy=float(balanced_accuracy_score(labels, probability_values[rows] >= threshold)),
                false_alerts_per_hour=metrics.false_alerts_per_hour,
                drowsy_sessions_caught=metrics.drowsy_sessions_caught,
            )
        )
    return results


def within_recording(windows: pd.DataFrame, scores: npt.ArrayLike) -> WithinRecording:
    """Pairs of drowsy and alert windows inside the same drive clip, over every session in ``windows``."""
    values = np.asarray(scores, dtype=float)
    total = NO_RECORDINGS
    drive = windows["clip"].to_numpy() == CLIP_DRIVE
    for _, rows in windows.loc[drive].groupby("session", sort=False).indices.items():
        positions = np.flatnonzero(drive)[rows]
        labels = windows["y"].to_numpy()[positions]
        drowsy, alert = values[positions][labels == 1], values[positions][labels == 0]
        if drowsy.size and alert.size:
            difference = drowsy[:, None] - alert[None, :]
            concordant = float(np.count_nonzero(difference > 0) + 0.5 * np.count_nonzero(difference == 0))
            total = total + WithinRecording(1, drowsy.size * alert.size, concordant)
    return total


def log_scopes(log: logging.Logger, stage: str, scores: Iterable[ScopeScores]) -> None:
    for score in scores:
        log.info(
            "%s: %s, %s scope (%d people): AUC %.3f, balanced accuracy %.3f, %.2f false alerts/h",
            stage,
            score.model,
            score.baseline_scope,
            score.people,
            score.auc,
            score.balanced_accuracy,
            score.false_alerts_per_hour,
        )


def pooled(parts: Iterable[WithinRecording]) -> WithinRecording:
    total = NO_RECORDINGS
    for part in parts:
        total = total + part
    return total
