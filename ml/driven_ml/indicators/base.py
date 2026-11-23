"""Shared types for the indicator submodels: the reading each one emits, and the inputs they all read.

Every submodel holds one invariant: input that cannot support a reading (no face, hidden eyes, a faulted or
noisy microphone) gives ``available=False`` with ``severity=0``, never a high reading. A camera that cannot see
the eyes is not evidence of closed eyes, and a quiet microphone is not evidence that breathing stopped.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

UNAVAILABLE = "unavailable"
TIME_EPSILON_SECONDS = 1e-6  # frame timestamps are sums of 0.1 s steps; compare gaps with this slack


class IndicatorName(StrEnum):
    EYES = "eyes"
    OCULAR = "ocular"
    HEAD = "head"
    BREATHING = "breathing"
    RESPONSIVENESS = "responsiveness"
    SENSOR_QUALITY = "sensor_quality"


BEHAVIOUR_INDICATORS: tuple[IndicatorName, ...] = (
    IndicatorName.EYES,
    IndicatorName.OCULAR,
    IndicatorName.HEAD,
    IndicatorName.BREATHING,
    IndicatorName.RESPONSIVENESS,
)


@dataclass(frozen=True)
class IndicatorReading:
    """One submodel's reading at one time. ``value`` units are given by each submodel's docstring."""

    name: IndicatorName
    t_seconds: float
    available: bool
    state: str
    value: float | None
    severity: float
    confidence: float
    evidence: tuple[str, ...]
    unavailable_reason: str | None

    def __post_init__(self) -> None:
        if not math.isfinite(self.t_seconds):
            raise ValueError(f"{self.name} reading needs a finite timestamp")
        for label, number in (("severity", self.severity), ("confidence", self.confidence)):
            if not math.isfinite(number) or not 0.0 <= number <= 1.0:
                raise ValueError(f"{self.name} {label} must be between 0 and 1, got {number}")
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError(f"{self.name} value must be finite or None")
        if not self.available and (self.severity != 0.0 or not self.unavailable_reason):
            raise ValueError(f"an unavailable {self.name} reading needs severity 0 and a reason")
        if self.available and self.unavailable_reason is not None:
            raise ValueError(f"an available {self.name} reading cannot carry an unavailable reason")

    def to_json(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "state": self.state,
            "value": self.value,
            "severity": self.severity,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "unavailable_reason": self.unavailable_reason,
        }


def unavailable_reading(
    name: IndicatorName, t_seconds: float, reason: str, evidence: Sequence[str] = ()
) -> IndicatorReading:
    return IndicatorReading(
        name=name,
        t_seconds=t_seconds,
        available=False,
        state=UNAVAILABLE,
        value=None,
        severity=0.0,
        confidence=0.0,
        evidence=tuple(evidence),
        unavailable_reason=reason,
    )


def clamp01(value: float) -> float:
    return 0.0 if not math.isfinite(value) else min(1.0, max(0.0, value))


def within_gap(elapsed_seconds: float, limit_seconds: float) -> bool:
    """True when a gap between frames is at most ``limit_seconds``, allowing for float rounding of 0.1 s steps."""
    return elapsed_seconds <= limit_seconds + TIME_EPSILON_SECONDS


def finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)


@dataclass(frozen=True)
class FrameSignals:
    """The per-frame inputs every camera indicator reads. NaN marks a signal the frame could not measure.

    ``eye_visibility`` and ``mouth_visibility`` are 0..1: region contrast against the driver's calibration
    (Studio) or the condition test's clean frame. ``shoulder_y`` is the vertical shoulder-line position in image
    heights, or None when the shoulders are not in frame.
    """

    t_seconds: float
    face: bool
    ear_left: float
    ear_right: float
    yaw: float
    pitch: float
    roll: float
    gaze_horizontal: float
    eye_visibility: float = 1.0
    mouth_visibility: float = 1.0
    shoulder_y: float | None = None

    @classmethod
    def from_features(
        cls,
        t_seconds: float,
        features: Mapping[str, float],
        *,
        eye_visibility: float = 1.0,
        mouth_visibility: float = 1.0,
        shoulder_y: float | None = None,
    ) -> FrameSignals:
        """Build from a ``frame_features`` dict (``face`` > 0 means a face was found)."""

        def get(name: str) -> float:
            value = features.get(name, math.nan)
            return math.nan if value is None else float(value)

        face = get("face") > 0
        return cls(
            t_seconds=t_seconds,
            face=face,
            ear_left=get("ear_left"),
            ear_right=get("ear_right"),
            yaw=get("yaw"),
            pitch=get("pitch"),
            roll=get("roll"),
            gaze_horizontal=get("gaze_horizontal"),
            eye_visibility=eye_visibility if face else 0.0,
            mouth_visibility=mouth_visibility if face else 0.0,
            shoulder_y=shoulder_y,
        )


@dataclass(frozen=True)
class DriverCalibration:
    """The driver's own reference, learned from the calibration period (medians of alert, forward frames).

    The open-eye levels come only from frames where the eyes were visible; they are None when the driver wore
    sunglasses for the whole calibration. Head and gaze centres still come from every tracked frame, so head
    monitoring and the head-based response check work without the eyes. Recalibrate after taking sunglasses off.
    """

    open_ear_left: float | None
    open_ear_right: float | None
    gaze_centre: float
    yaw_centre: float
    pitch_centre: float
    roll_centre: float

    @property
    def eyes_calibrated(self) -> bool:
        return self.open_ear_left is not None and self.open_ear_right is not None

    @classmethod
    def from_frames(cls, frames: Sequence[FrameSignals], eye_visibility_cutoff: float) -> DriverCalibration | None:
        """None when the calibration never tracked a face."""
        tracked = [frame for frame in frames if frame.face]
        if not tracked:
            return None
        visible = [frame for frame in tracked if frame.eye_visibility >= eye_visibility_cutoff]

        def median(values: list[float]) -> float:
            usable = sorted(value for value in values if math.isfinite(value))
            if not usable:
                return math.nan
            middle = len(usable) // 2
            return usable[middle] if len(usable) % 2 else (usable[middle - 1] + usable[middle]) / 2

        def open_level(values: list[float]) -> float | None:
            value = median(values)
            return value if math.isfinite(value) and value > 0 else None

        def centre(values: list[float]) -> float:
            value = median(values)
            return value if math.isfinite(value) else 0.0

        open_left = open_level([frame.ear_left for frame in visible])
        open_right = open_level([frame.ear_right for frame in visible])
        both = open_left is not None and open_right is not None
        return cls(
            open_ear_left=open_left if both else None,
            open_ear_right=open_right if both else None,
            gaze_centre=centre([frame.gaze_horizontal for frame in visible]),
            yaw_centre=centre([frame.yaw for frame in tracked]),
            pitch_centre=centre([frame.pitch for frame in tracked]),
            roll_centre=centre([frame.roll for frame in tracked]),
        )
