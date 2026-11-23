"""The visual response check: idle, waiting, responded with latency, no response, and the head-only path."""

from __future__ import annotations

import pytest
from conftest import CLOSED_EAR, make_frame, times

from driven_ml.indicators import DriverCalibration, ResponsivenessIndicator


def run(
    check: ResponsivenessIndicator, calibration: DriverCalibration, start: float, seconds: float, **kwargs: float | bool
):
    return [check.push(make_frame(t, **kwargs), calibration) for t in times(start, seconds)]  # type: ignore[arg-type]


def test_idle_until_an_alert_then_responded_with_latency(calibration: DriverCalibration) -> None:
    check = ResponsivenessIndicator()
    assert run(check, calibration, 0.0, 1.0)[-1].state == "idle"
    assert check.open_window(1.0)
    run(check, calibration, 1.0, 2.0, ear=CLOSED_EAR)
    reply = run(check, calibration, 3.0, 1.0)
    assert reply[-1].state == "responded"
    assert reply[-1].value == pytest.approx(2.0)
    assert reply[-1].severity == 0.0


def test_no_response_after_five_seconds_and_a_late_response(calibration: DriverCalibration) -> None:
    check = ResponsivenessIndicator()
    check.open_window(0.0)
    waiting = run(check, calibration, 0.0, 4.9, ear=CLOSED_EAR, pitch=-30.0)
    assert waiting[-1].state == "waiting"
    assert not check.open_window(3.0)  # already waiting: ignored
    expired = run(check, calibration, 4.9, 0.3, ear=CLOSED_EAR, pitch=-30.0)
    assert expired[-1].state == "no_response"
    assert expired[-1].severity == pytest.approx(0.9)
    late = run(check, calibration, 5.2, 1.0)
    assert late[-1].state == "responded"
    assert late[-1].value == pytest.approx(5.2)


def test_hidden_eyes_use_the_head_based_half(calibration: DriverCalibration) -> None:
    check = ResponsivenessIndicator()
    check.open_window(0.0)
    dropped = run(check, calibration, 0.0, 5.5, eye_visibility=0.1, pitch=-30.0)
    assert dropped[-1].state == "no_response"
    check = ResponsivenessIndicator()
    check.open_window(0.0)
    upright = run(check, calibration, 0.0, 1.0, eye_visibility=0.1, ear=CLOSED_EAR)
    assert upright[-1].state == "responded"  # closed-looking eyes behind sunglasses are not read


def test_untrackable_face_during_the_check_is_unavailable(calibration: DriverCalibration) -> None:
    check = ResponsivenessIndicator()
    check.open_window(0.0)
    lost = run(check, calibration, 0.0, 5.5, face=False)
    assert all(not reading.available and reading.severity == 0.0 for reading in lost)
    back = check.push(make_frame(5.6, ear=CLOSED_EAR), calibration)
    assert back.unavailable_reason == "face_not_tracked_during_check"
    assert check.push(make_frame(6.0), None).unavailable_reason == "calibrating"
