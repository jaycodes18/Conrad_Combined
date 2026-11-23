"""Erratic eye movement over 5 s and 10 s, against the driver's own calibration.

The signal is the horizontal iris position from the MediaPipe iris landmarks (468 and 473, the centres of the
468-477 iris rings) relative to each eye's corners, minus the driver's calibrated centre. Over each window it
measures three things: dispersion (standard deviation of the gaze offset), direction reversals per second whose
swing is at least ``min_reversal_amplitude``, and the share of time off centre. Each becomes a z-score against
the same metric's distribution over the calibration period, and ``instability`` (the ``value``) is the mean of
the three. Short glances to the mirror zones seen during calibration are left out of the reversal count.

What this is not: at 30 fps a phone camera cannot resolve saccade velocity, and saccades last tens of
milliseconds, so this measures dispersion and reversals, not nystagmus. The reference system (Koch et al., CHI
2023, doi:10.1145/3544548.3580975) used a dedicated eye tracker in a driving simulator with n = 30; the
real-vehicle follow-up is Koch et al., CHI 2025, doi:10.1145/3706598.3714007. Driven's numbers are not comparable
to theirs until they are measured on Driven data.

States: ``steady``, ``scanning`` (instability at or above ``scanning_instability`` but under the erratic line),
``erratic_5s`` (the 5 s instability has held at or above 3.0 for 5 s), ``erratic_10s`` (it has held for 10 s and
the 10 s window's own instability is also at or above 3.0), ``unavailable``. The hold matters: one large glance
lifts a window's z-scores past 3.0 within a second, and that is not 5 s of erratic movement.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..spec import INDICATOR_SPEC, OcularConfig
from .base import (
    DriverCalibration,
    FrameSignals,
    IndicatorName,
    IndicatorReading,
    clamp01,
    finite,
    unavailable_reading,
    within_gap,
)
from .eyes import eye_input_problem

STEADY = "steady"
SCANNING = "scanning"
ERRATIC_5S = "erratic_5s"
ERRATIC_10S = "erratic_10s"
ERRATIC_STATES = frozenset({ERRATIC_5S, ERRATIC_10S})
CALIBRATION_STEP_SECONDS = 1.0


@dataclass(frozen=True)
class MetricStats:
    mean: float
    std: float


@dataclass(frozen=True)
class OcularBaseline:
    """Per-window distributions of the three metrics during calibration, plus the learned mirror zones."""

    short: tuple[MetricStats, MetricStats, MetricStats]
    long: tuple[MetricStats, MetricStats, MetricStats]
    mirror_zones: tuple[float, ...]


def zigzag_reversals(values: np.ndarray, amplitude: float) -> int:
    """Count direction changes whose swing from the last extreme is at least ``amplitude``."""
    if values.size < 2:
        return 0
    reversals = 0
    direction = 0  # unknown until the signal first moves by ``amplitude``
    high = low = extreme = float(values[0])
    for raw in values[1:]:
        value = float(raw)
        if direction == 0:
            high, low = max(high, value), min(low, value)
            if high - low >= amplitude:
                direction, extreme = (1, value) if value >= high else (-1, value)
        elif direction == 1:
            if value > extreme:
                extreme = value
            elif extreme - value >= amplitude:
                reversals, direction, extreme = reversals + 1, -1, value
        elif value < extreme:
            extreme = value
        elif value - extreme >= amplitude:
            reversals, direction, extreme = reversals + 1, 1, value
    return reversals


def _mirror_mask(times: np.ndarray, offsets: np.ndarray, zones: Sequence[float], config: OcularConfig) -> np.ndarray:
    """True for samples inside a short off-centre glance that lands in a learned mirror zone."""
    mask = np.zeros(offsets.size, dtype=bool)
    if not zones:
        return mask
    away = np.abs(offsets) > config.off_centre_ratio
    index = 0
    while index < offsets.size:
        if not away[index]:
            index += 1
            continue
        end = index
        while end + 1 < offsets.size and away[end + 1]:
            end += 1
        duration = float(times[end] - times[index])
        position = float(np.mean(offsets[index : end + 1]))
        if duration <= config.mirror_glance_max_seconds and any(
            abs(position - zone) <= config.mirror_zone_tolerance for zone in zones
        ):
            mask[index : end + 1] = True
        index = end + 1
    return mask


def window_metrics(
    times: np.ndarray, offsets: np.ndarray, window_seconds: float, zones: Sequence[float], config: OcularConfig
) -> tuple[float, float, float]:
    """(dispersion, reversals per second, off-centre fraction) for one window of valid samples."""
    dispersion = float(np.std(offsets))
    for_reversals = np.where(_mirror_mask(times, offsets, zones, config), 0.0, offsets)
    reversals = zigzag_reversals(for_reversals, config.min_reversal_amplitude) / window_seconds
    off_centre = float(np.mean(np.abs(offsets) > config.off_centre_ratio))
    return dispersion, reversals, off_centre


def _stats(values: list[float], floor: float) -> MetricStats:
    array = np.asarray(values, dtype=float)
    return MetricStats(float(np.mean(array)), max(float(np.std(array)), floor))


def learn_mirror_zones(times: np.ndarray, offsets: np.ndarray, config: OcularConfig) -> tuple[float, ...]:
    """Median position of short off-centre glances on each side, when there are enough of them."""
    away = np.abs(offsets) > config.off_centre_ratio
    glances: list[float] = []
    index = 0
    while index < offsets.size:
        if not away[index]:
            index += 1
            continue
        end = index
        while end + 1 < offsets.size and away[end + 1]:
            end += 1
        if float(times[end] - times[index]) <= config.mirror_glance_max_seconds:
            glances.append(float(np.mean(offsets[index : end + 1])))
        index = end + 1
    zones = []
    for side in (-1.0, 1.0):
        on_side = [glance for glance in glances if glance * side > 0]
        if len(on_side) >= config.min_mirror_glances:
            zones.append(float(np.median(on_side)))
    return tuple(zones)


class OcularIndicator:
    """Fit a baseline from the calibration frames, then ``push`` once per frame."""

    def __init__(
        self, config: OcularConfig = INDICATOR_SPEC.ocular, eye_visibility_cutoff: float | None = None
    ) -> None:
        self.config = config
        self.eye_visibility_cutoff = (
            INDICATOR_SPEC.eyes.eye_visibility_cutoff if eye_visibility_cutoff is None else eye_visibility_cutoff
        )
        self.baseline: OcularBaseline | None = None
        self.reset()

    def reset(self) -> None:
        self._samples: deque[tuple[float, float]] = deque()
        self.instability_short: float | None = None
        self.instability_long: float | None = None
        self._erratic_since: float | None = None
        self._last_valid_t: float | None = None

    def fit_baseline(self, frames: Sequence[FrameSignals], calibration: DriverCalibration) -> OcularBaseline | None:
        """Slide both windows over the calibration in 1 s steps; None when there is too little valid gaze."""
        config = self.config
        valid = [
            (frame.t_seconds, frame.gaze_horizontal - calibration.gaze_centre)
            for frame in frames
            if eye_input_problem(frame, self.eye_visibility_cutoff) is None and finite(frame.gaze_horizontal)
        ]
        if len(valid) < 10:
            self.baseline = None
            return None
        times = np.array([item[0] for item in valid], dtype=float)
        offsets = np.array([item[1] for item in valid], dtype=float)
        zones = learn_mirror_zones(times, offsets, config)
        floors = (config.dispersion_std_floor, config.reversal_std_floor, config.off_centre_std_floor)
        per_window: list[tuple[MetricStats, MetricStats, MetricStats]] = []
        for window in (config.short_window_seconds, config.long_window_seconds):
            rows: list[tuple[float, float, float]] = []
            end = times[0] + window
            while end <= times[-1] + 1e-9:
                inside = (times > end - window) & (times <= end)
                if inside.sum() >= 3:
                    rows.append(window_metrics(times[inside], offsets[inside], window, zones, config))
                end += CALIBRATION_STEP_SECONDS
            if not rows:
                self.baseline = None
                return None
            per_window.append(
                (
                    _stats([row[0] for row in rows], floors[0]),
                    _stats([row[1] for row in rows], floors[1]),
                    _stats([row[2] for row in rows], floors[2]),
                )
            )
        self.baseline = OcularBaseline(short=per_window[0], long=per_window[1], mirror_zones=zones)
        return self.baseline

    def push(self, frame: FrameSignals, calibration: DriverCalibration | None) -> IndicatorReading:
        t = frame.t_seconds
        config = self.config
        if calibration is None or self.baseline is None:
            return unavailable_reading(
                IndicatorName.OCULAR, t, "calibrating", ("Learning this driver's normal eye movement.",)
            )
        problem = eye_input_problem(frame, self.eye_visibility_cutoff)
        if problem is None and not finite(frame.gaze_horizontal):
            problem = "gaze_missing"
        self._samples.append((t, math.nan if problem else frame.gaze_horizontal - calibration.gaze_centre))
        while self._samples and self._samples[0][0] <= t - config.long_window_seconds:
            self._samples.popleft()
        if problem is not None:
            self.instability_short = self.instability_long = None
            return unavailable_reading(
                IndicatorName.OCULAR, t, problem, ("Eye movement is not measured while the eyes cannot be seen.",)
            )
        short = self._instability(t, config.short_window_seconds, self.baseline.short)
        long = self._instability(t, config.long_window_seconds, self.baseline.long)
        self.instability_short = None if short is None else short[0]
        self.instability_long = None if long is None else long[0]
        if self._last_valid_t is not None and not within_gap(
            t - self._last_valid_t, INDICATOR_SPEC.eyes.bridge_gap_seconds
        ):
            self._erratic_since = None  # a gap longer than the bridge breaks the hold, as for eyes and head
        self._last_valid_t = t
        if short is None:
            self._erratic_since = None
            return unavailable_reading(
                IndicatorName.OCULAR,
                t,
                "insufficient_gaze_samples",
                ("Collecting enough eye-movement samples for a 5 s window.",),
            )
        instability, coverage = short
        if instability < config.erratic_instability:
            self._erratic_since = None
        elif self._erratic_since is None:
            self._erratic_since = t
        held = 0.0 if self._erratic_since is None else t - self._erratic_since
        long_erratic = long is not None and long[0] >= config.erratic_instability
        if long_erratic and long is not None and within_gap(config.long_window_seconds, held):
            state, value, severity = ERRATIC_10S, long[0], 0.9
            evidence = f"Eye movement has been unusually scattered for {held:.0f} s (instability {long[0]:.1f})."
        elif self._erratic_since is not None and within_gap(config.short_window_seconds, held):
            state, value, severity = ERRATIC_5S, instability, 0.6
            evidence = f"Eye movement has been unusually scattered for {held:.0f} s (instability {instability:.1f})."
        elif instability >= config.scanning_instability:
            state, value, severity = SCANNING, instability, 0.2
            evidence = f"More eye movement than during calibration (instability {instability:.1f})."
        else:
            state, value, severity = STEADY, instability, 0.0
            evidence = "Eye movement is within this driver's calibrated range."
        return IndicatorReading(
            name=IndicatorName.OCULAR,
            t_seconds=t,
            available=True,
            state=state,
            value=value,
            severity=severity,
            confidence=clamp01(coverage * min(1.0, frame.eye_visibility)),
            evidence=(evidence,),
            unavailable_reason=None,
        )

    def _instability(
        self, t: float, window: float, stats: tuple[MetricStats, MetricStats, MetricStats]
    ) -> tuple[float, float] | None:
        """(instability, coverage), or None when under ``min_coverage`` of the window has a valid sample."""
        inside = [(when, offset) for when, offset in self._samples if when > t - window]
        valid = [(when, offset) for when, offset in inside if math.isfinite(offset)]
        spanned = inside and t - inside[0][0] >= window * self.config.min_coverage
        coverage = len(valid) / max(len(inside), 1)
        if not spanned or len(valid) < 3 or coverage < self.config.min_coverage:
            return None
        times = np.array([item[0] for item in valid], dtype=float)
        offsets = np.array([item[1] for item in valid], dtype=float)
        zones = self.baseline.mirror_zones if self.baseline is not None else ()
        metrics = window_metrics(times, offsets, window, zones, self.config)
        z_scores = [(value - stat.mean) / stat.std for value, stat in zip(metrics, stats, strict=True)]
        return float(np.mean(z_scores)), coverage
