"""Head control: nods counted per minute, a held head drop, a slump, gaps, and the shared nod threshold."""

from __future__ import annotations

import pytest
from conftest import make_frame, times

from driven_ml.indicators import DriverCalibration, HeadIndicator
from driven_ml.spec import INDICATOR_SPEC, SignalThresholds


def run(head: HeadIndicator, calibration: DriverCalibration, start: float, seconds: float, **kwargs: float | bool):
    return [head.push(make_frame(t, **kwargs), calibration) for t in times(start, seconds)]  # type: ignore[arg-type]


def test_short_dips_are_nods_and_three_per_minute_is_nodding(calibration: DriverCalibration) -> None:
    head = HeadIndicator()
    run(head, calibration, 0.0, 2.0)
    for index in range(3):
        start = 2.0 + index * 4.0
        run(head, calibration, start, 1.0, pitch=-20.0)
        last = run(head, calibration, start + 1.0, 3.0)
    assert head.nods_last_minute == 3
    assert last[-1].state == "nodding"
    assert last[-1].value == 3.0
    later = run(head, calibration, 14.0, 60.0)
    assert later[-1].state == "stable"


def test_a_long_dip_is_not_a_nod_and_a_held_drop_is_head_drop(calibration: DriverCalibration) -> None:
    head = HeadIndicator()
    run(head, calibration, 0.0, 1.0)
    shallow = run(head, calibration, 1.0, 4.0, pitch=-20.0)
    assert {reading.state for reading in shallow} == {"stable"}
    run(head, calibration, 5.0, 1.0)
    assert head.nods_last_minute == 0
    drop = run(head, calibration, 6.0, 3.0, pitch=-30.0)
    assert drop[18].state == "stable"  # 19 frames: 1.9 s
    assert drop[19].state == "head_drop"  # 20 frames: 2.0 s
    assert drop[-1].value == pytest.approx(3.0)


def test_roll_past_thirty_degrees_held_is_slumped(calibration: DriverCalibration) -> None:
    head = HeadIndicator()
    run(head, calibration, 0.0, 1.0)
    slump = run(head, calibration, 1.0, 2.5, roll=35.0)
    assert slump[-1].state == "slumped"


def test_lost_face_is_unavailable_after_the_bridge(calibration: DriverCalibration) -> None:
    head = HeadIndicator()
    run(head, calibration, 0.0, 1.0)
    run(head, calibration, 1.0, 2.5, pitch=-30.0)
    bridged = run(head, calibration, 3.5, 0.2, face=False)
    assert all(reading.state == "head_drop" for reading in bridged)
    lost = run(head, calibration, 3.7, 0.5, face=False)
    assert not lost[-1].available
    assert lost[-1].severity == 0.0
    assert lost[-1].unavailable_reason == "face_not_tracked"
    again = run(head, calibration, 4.2, 0.5, pitch=-30.0)
    assert again[-1].state == "stable"  # the hold restarted after the long gap
    assert head.push(make_frame(10.0), None).unavailable_reason == "calibrating"


def test_nod_minimum_reuses_the_signal_threshold() -> None:
    assert INDICATOR_SPEC.head.nod_min_seconds == SignalThresholds().nod_min_seconds
