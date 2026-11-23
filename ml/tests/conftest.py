"""Shared factories for the indicator, fusion and escalation tests: frames, a calibration, a contact book."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from driven_ml.indicators import DriverCalibration, FrameSignals
from driven_ml.product import load_product
from driven_ml.risk import ConsentState, ContactBook, EscalationConfig

FPS = 10
OPEN_EAR = 0.30
CLOSED_EAR = 0.05

FrameFactory = Callable[..., FrameSignals]


def make_frame(
    t: float,
    *,
    ear: float = OPEN_EAR,
    ear_left: float | None = None,
    ear_right: float | None = None,
    face: bool = True,
    eye_visibility: float = 1.0,
    mouth_visibility: float = 1.0,
    pitch: float = 0.0,
    roll: float = 0.0,
    yaw: float = 0.0,
    gaze: float = 0.0,
    shoulder_y: float | None = None,
) -> FrameSignals:
    nan = float("nan")
    return FrameSignals(
        t_seconds=t,
        face=face,
        ear_left=(ear if ear_left is None else ear_left) if face else nan,
        ear_right=(ear if ear_right is None else ear_right) if face else nan,
        yaw=yaw if face else nan,
        pitch=pitch if face else nan,
        roll=roll if face else nan,
        gaze_horizontal=gaze if face else nan,
        eye_visibility=eye_visibility if face else 0.0,
        mouth_visibility=mouth_visibility if face else 0.0,
        shoulder_y=shoulder_y,
    )


def times(start: float, seconds: float, fps: int = FPS) -> list[float]:
    """Frame timestamps from ``start`` for ``seconds``, rounded so they never drift."""
    return [round(start + index / fps, 6) for index in range(round(seconds * fps))]


@pytest.fixture
def frame() -> FrameFactory:
    return make_frame


@pytest.fixture
def calibration() -> DriverCalibration:
    return DriverCalibration(
        open_ear_left=OPEN_EAR,
        open_ear_right=OPEN_EAR,
        gaze_centre=0.0,
        yaw_centre=0.0,
        pitch_centre=0.0,
        roll_centre=0.0,
    )


@pytest.fixture
def escalation_config() -> EscalationConfig:
    return EscalationConfig.from_product(load_product())


def consented_book(*, calls: bool = True) -> ContactBook:
    book = ContactBook()
    book.add(
        "parent",
        "Parent",
        "+12145550100",
        consent=ConsentState.RECORDED,
        consent_date="2026-11-01",
        calls_consented=calls,
    )
    return book
