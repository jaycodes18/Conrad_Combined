"""Simulate the in-car alert and pick the threshold by the false-alert budget.

An alert fires when a window's probability reaches the threshold and the last alert was at least the cooldown
ago. An alert is judged by the label of the window that fired it. The threshold is the lowest one that keeps
false alerts per hour of alert driving within the budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..spec import ALERT_COOLDOWN_SECONDS, FALSE_ALERT_BUDGET_PER_HOUR, FEATURE_SPEC

THRESHOLD_GRID = np.linspace(0.05, 0.99, 95)
MAX_THRESHOLD = 0.99
_SECONDS_PER_HOUR = 3600


@dataclass(frozen=True)
class AlertMetrics:
    false_alerts_per_hour: float
    drowsy_sessions_caught: float


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float
    meets_budget: bool


def alert_window_starts(
    t0: npt.ArrayLike,
    probabilities: npt.ArrayLike,
    threshold: float,
    cooldown_seconds: float,
) -> list[float]:
    """Start times of the windows that fire, in time order: the phone's alert loop for one session."""
    starts, scores = np.asarray(t0, dtype=float), np.asarray(probabilities, dtype=float)
    fired: list[float] = []
    for start, probability in sorted(zip(starts, scores, strict=True)):
        if probability >= threshold and (not fired or start - fired[-1] >= cooldown_seconds):
            fired.append(float(start))
    return fired


def alert_metrics(windows: pd.DataFrame, probabilities: npt.ArrayLike, threshold: float) -> AlertMetrics:
    """False alerts are alerts fired by an alert (y = 0) window; a drowsy session is caught when an alert fires on
    one of its drowsy windows. A team drive holds both kinds of window, so the label of the firing window
    decides; on single-label sessions this is the same as counting every alert in an alert session.
    """
    p = np.asarray(probabilities, dtype=float)
    if len(p) != len(windows):
        raise ValueError(f"{len(p)} probabilities for {len(windows)} windows")
    scored = pd.DataFrame(
        {
            "session": windows["session"].to_numpy(),
            "t0": windows["t0"].to_numpy(),
            "p": p,
            "y": windows["y"],
        }
    )
    false_alerts = drowsy_sessions = caught_sessions = 0
    for _, group in scored.groupby("session", sort=False):
        fired = alert_window_starts(group["t0"], group["p"], threshold, ALERT_COOLDOWN_SECONDS)
        fired_labels = group.loc[group["t0"].isin(fired), "y"]
        false_alerts += int(np.count_nonzero(fired_labels == 0))
        if (group["y"] == 1).any():
            drowsy_sessions += 1
            caught_sessions += bool((fired_labels == 1).any())
    alert_hours = max(
        np.count_nonzero(scored["y"] == 0) * FEATURE_SPEC.hop_seconds / _SECONDS_PER_HOUR,
        1e-9,
    )
    caught = caught_sessions / drowsy_sessions if drowsy_sessions else float("nan")
    return AlertMetrics(false_alerts_per_hour=false_alerts / alert_hours, drowsy_sessions_caught=caught)


def choose_threshold(windows: pd.DataFrame, probabilities: npt.ArrayLike) -> ThresholdChoice:
    """The lowest grid threshold within the false-alert budget, or 0.99 flagged as missing it."""
    for threshold in THRESHOLD_GRID:
        if alert_metrics(windows, probabilities, threshold).false_alerts_per_hour <= FALSE_ALERT_BUDGET_PER_HOUR:
            return ThresholdChoice(threshold=float(threshold), meets_budget=True)
    return ThresholdChoice(threshold=MAX_THRESHOLD, meets_budget=False)
