"""Erratic eye movement: baseline z-scores over 5 s and 10 s, mirror glances, and the unavailable invariant."""

from __future__ import annotations

import numpy as np
from conftest import make_frame, times

from driven_ml.indicators import DriverCalibration, FrameSignals, OcularIndicator
from driven_ml.indicators.ocular import zigzag_reversals


def noisy_gaze(start: float, seconds: float, seed: int, *, mirror_every: float | None = None) -> list[FrameSignals]:
    """Calibration-like gaze: small jitter, plus optional 0.5 s glances to a mirror at +0.3."""
    rng = np.random.default_rng(seed)
    frames = []
    for t in times(start, seconds):
        gaze = float(rng.normal(0.0, 0.03))
        if mirror_every is not None and (t - start) % mirror_every < 0.5:
            gaze = 0.3 + float(rng.normal(0.0, 0.01))
        frames.append(make_frame(t, gaze=gaze))
    return frames


def fitted(calibration: DriverCalibration, *, mirror_every: float | None = None) -> OcularIndicator:
    ocular = OcularIndicator()
    assert ocular.fit_baseline(noisy_gaze(0.0, 120.0, 1, mirror_every=mirror_every), calibration) is not None
    return ocular


def test_zigzag_counts_only_swings_above_the_amplitude() -> None:
    assert zigzag_reversals(np.array([0.0, 0.1, 0.0, 0.1, 0.0]), 0.05) == 3
    assert zigzag_reversals(np.array([0.0, 0.01, 0.0, 0.01, 0.0]), 0.05) == 0
    assert zigzag_reversals(np.array([0.0]), 0.05) == 0


def test_steady_gaze_stays_steady_and_square_wave_becomes_erratic(calibration: DriverCalibration) -> None:
    ocular = fitted(calibration)
    steady = [ocular.push(frame, calibration) for frame in noisy_gaze(120.0, 30.0, 2)]
    assert {reading.state for reading in steady if reading.available} <= {"steady", "scanning"}
    rng = np.random.default_rng(3)
    erratic = []
    for index, t in enumerate(times(150.0, 13.0)):
        gaze = (0.3 if (index // 3) % 2 == 0 else -0.3) + float(rng.normal(0.0, 0.02))
        erratic.append(ocular.push(make_frame(t, gaze=gaze), calibration))
    states = [reading.state for reading in erratic]
    first_5s = states.index("erratic_5s") if "erratic_5s" in states else None
    first_10s = states.index("erratic_10s")
    assert first_5s is not None
    assert first_5s < first_10s
    assert erratic[first_5s].t_seconds - 150.0 >= 5.0  # held for 5 s, not one glance
    assert erratic[first_10s].t_seconds - 150.0 >= 10.0
    assert erratic[-1].value is not None
    assert erratic[-1].value >= 3.0


def test_learned_mirror_glances_do_not_count_as_erratic(calibration: DriverCalibration) -> None:
    ocular = fitted(calibration, mirror_every=8.0)
    assert ocular.baseline is not None
    assert ocular.baseline.mirror_zones
    readings = [ocular.push(frame, calibration) for frame in noisy_gaze(120.0, 60.0, 4, mirror_every=8.0)]
    assert not any(reading.state.startswith("erratic") for reading in readings)


def test_unavailable_without_baseline_or_visible_eyes(calibration: DriverCalibration) -> None:
    assert OcularIndicator().push(make_frame(0.0), calibration).unavailable_reason == "calibrating"
    ocular = fitted(calibration)
    for t in times(120.0, 5.0):
        reading = ocular.push(make_frame(t, gaze=0.4, eye_visibility=0.1), calibration)
        assert not reading.available
        assert reading.severity == 0.0
    assert reading.unavailable_reason == "eyes_not_visible"
    lost = ocular.push(make_frame(126.0, face=False), calibration)
    assert lost.unavailable_reason == "face_not_tracked"
