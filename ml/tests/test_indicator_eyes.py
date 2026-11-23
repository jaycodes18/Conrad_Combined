"""Eyelid closure: states, the 2 s and 5 s lines, the unavailable invariant, winks and gap bridging."""

from __future__ import annotations

import pytest
from conftest import CLOSED_EAR, make_frame, times

from driven_ml.indicators import DriverCalibration, EyesIndicator
from driven_ml.spec import INDICATOR_SPEC, SignalThresholds


def run(eyes: EyesIndicator, calibration: DriverCalibration, start: float, seconds: float, **kwargs: object):
    return [eyes.push(make_frame(t, **kwargs), calibration) for t in times(start, seconds)]  # type: ignore[arg-type]


def test_closure_passes_warning_then_critical(calibration: DriverCalibration) -> None:
    eyes = EyesIndicator()
    run(eyes, calibration, 0.0, 5.0)
    closed = run(eyes, calibration, 5.0, 5.0, ear=CLOSED_EAR)  # 50 frames at 10 fps: exactly 5.0 s
    states = [reading.state for reading in closed]
    assert states[0] == "blinking"
    assert states[18] == "blinking"  # 19 frames: 1.9 s
    assert states[19] == "closure_warning"  # 20 frames: 2.0 s
    assert states[48] == "closure_warning"
    assert states[49] == "closure_critical"
    assert closed[-1].value == pytest.approx(5.0)
    assert closed[-1].severity == pytest.approx(1.0)
    assert INDICATOR_SPEC.eyes.closure_critical_seconds == 5.0


def test_short_blink_records_duration_and_reopens(calibration: DriverCalibration) -> None:
    eyes = EyesIndicator()
    run(eyes, calibration, 0.0, 1.0)
    blink = run(eyes, calibration, 1.0, 0.3, ear=CLOSED_EAR)
    after = run(eyes, calibration, 1.3, 1.0)
    assert {reading.state for reading in blink} == {"blinking"}
    assert after[0].state == "open"
    assert eyes.last_blink_seconds == pytest.approx(0.3)
    assert eyes.perclos is not None
    assert 0.0 < eyes.perclos < 0.2


def test_hidden_eyes_and_lost_face_are_never_evidence_of_closure(calibration: DriverCalibration) -> None:
    eyes = EyesIndicator()
    hidden = run(eyes, calibration, 0.0, 8.0, ear=CLOSED_EAR, eye_visibility=0.1)
    lost = run(eyes, calibration, 8.0, 8.0, face=False)
    for reading in (*hidden, *lost):
        assert not reading.available
        assert reading.severity == 0.0
        assert reading.value is None
    assert hidden[-1].unavailable_reason == "eyes_not_visible"
    assert lost[-1].unavailable_reason == "face_not_tracked"
    assert eyes.push(make_frame(20.0), None).unavailable_reason == "calibrating"


def test_one_eye_shut_is_a_wink_not_closure(calibration: DriverCalibration) -> None:
    eyes = EyesIndicator()
    wink = run(eyes, calibration, 0.0, 6.0, ear_left=CLOSED_EAR, ear_right=0.30)
    assert {reading.state for reading in wink} == {"open"}
    assert "wink" in wink[-1].evidence[0]


def test_short_gaps_bridge_and_long_gaps_end_the_episode(calibration: DriverCalibration) -> None:
    eyes = EyesIndicator()
    run(eyes, calibration, 0.0, 2.0)
    run(eyes, calibration, 2.0, 1.5, ear=CLOSED_EAR)
    bridged = run(eyes, calibration, 3.5, 0.2, face=False)  # two lost frames: under the 0.3 s bridge
    assert all(reading.available and reading.state == "blinking" for reading in bridged)
    resumed = run(eyes, calibration, 3.7, 0.5, ear=CLOSED_EAR)
    assert resumed[-1].state == "closure_warning"  # the episode kept its start at 2.0 s
    assert resumed[-1].value == pytest.approx(2.2)  # frames 2.0 s to 4.1 s, the two lost frames included
    lost = run(eyes, calibration, 4.2, 0.5, face=False)  # 0.5 s: too long to bridge
    assert lost[-1].unavailable_reason == "face_not_tracked"
    assert lost[-1].severity == 0.0
    restarted = run(eyes, calibration, 4.7, 0.5, ear=CLOSED_EAR)
    assert restarted[-1].state == "blinking"
    assert restarted[-1].value == pytest.approx(0.5)


def test_thresholds_reuse_the_signal_spec() -> None:
    config, signal = INDICATOR_SPEC.eyes, SignalThresholds()
    assert config.closed_fraction == signal.closed_fraction
    assert config.perclos_fraction == signal.perclos_fraction
