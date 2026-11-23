"""Head control: nods, a dropped head, and a slumped head, relative to the driver's calibrated pose.

Pitch, yaw and roll come from MediaPipe's facial transformation matrix (``features.geometry.head_pose``); down is
negative pitch. A nod is a pitch drop of at least ``nod_drop_degrees`` (15) that recovers within
``nod_recovery_seconds`` (2) and lasts at least ``SignalThresholds.nod_min_seconds``, so landmark jitter is not a
nod. ``value`` is the hold time in seconds for ``head_drop`` and ``slumped``, otherwise nods in the last 60 s.

States: ``stable``, ``nodding`` (``nodding_min_per_minute`` nods or more in 60 s), ``head_drop`` (pitch 25 degrees
or more below calibration, held 2 s), ``slumped`` (roll 30 degrees or more, held 2 s), ``unavailable``. A face
that MediaPipe loses while the head drops is unavailable, not a drop: steep head-down poses can lose tracking.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace

from ..spec import INDICATOR_SPEC, HeadConfig
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

STABLE = "stable"
NODDING = "nodding"
HEAD_DROP = "head_drop"
SLUMPED = "slumped"
LOST_CONTROL_STATES = frozenset({HEAD_DROP, SLUMPED})
NOD_WINDOW_SECONDS = 60.0


def relative_pose(frame: FrameSignals, calibration: DriverCalibration) -> tuple[float, float]:
    """(pitch, roll) in degrees relative to calibration."""
    return frame.pitch - calibration.pitch_centre, frame.roll - calibration.roll_centre


class HeadIndicator:
    def __init__(self, config: HeadConfig = INDICATOR_SPEC.head) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._nods: deque[float] = deque()
        self._dip_start: float | None = None
        self._drop_since: float | None = None
        self._slump_since: float | None = None
        self._last_valid_t: float | None = None
        self._frame_seconds = 0.0
        self._last: IndicatorReading | None = None

    @property
    def nods_last_minute(self) -> int:
        return len(self._nods)

    def push(self, frame: FrameSignals, calibration: DriverCalibration | None) -> IndicatorReading:
        t = frame.t_seconds
        if calibration is None:
            return self._remember(
                unavailable_reading(IndicatorName.HEAD, t, "calibrating", ("Learning this driver's head position.",))
            )
        if not frame.face or not (finite(frame.pitch) and finite(frame.roll)):
            return self._gap(t)
        self._frame_seconds = 0.0 if self._last_valid_t is None else t - self._last_valid_t
        if self._last_valid_t is not None and not within_gap(self._frame_seconds, self.config.bridge_gap_seconds):
            self._clear_holds()
            self._frame_seconds = 0.0
        self._last_valid_t = t
        pitch, roll = relative_pose(frame, calibration)
        self._track_nod(t, pitch)
        config = self.config
        if pitch > -config.head_drop_degrees:
            self._drop_since = None
        elif self._drop_since is None:
            self._drop_since = t
        if abs(roll) < config.slump_roll_degrees:
            self._slump_since = None
        elif self._slump_since is None:
            self._slump_since = t
        return self._remember(self._classify(t, pitch, roll))

    def _classify(self, t: float, pitch: float, roll: float) -> IndicatorReading:
        config = self.config
        # Holds count each frame as one frame interval, like eye closure.
        drop_held = 0.0 if self._drop_since is None else t - self._drop_since + self._frame_seconds
        slump_held = 0.0 if self._slump_since is None else t - self._slump_since + self._frame_seconds
        severity = 0.0
        if drop_held >= config.hold_seconds and drop_held >= slump_held:
            state, value = HEAD_DROP, drop_held
            evidence = f"Head dropped {-pitch:.0f} degrees below the driver's normal position for {drop_held:.1f} s."
            severity = 0.6 + 0.4 * clamp01(value / 5.0)
        elif slump_held >= config.hold_seconds:
            state, value = SLUMPED, slump_held
            evidence = f"Head tilted {abs(roll):.0f} degrees to the side for {slump_held:.1f} s."
            severity = 0.6 + 0.4 * clamp01(value / 5.0)
        elif self.nods_last_minute >= config.nodding_min_per_minute:
            state, value = NODDING, float(self.nods_last_minute)
            evidence = f"{self.nods_last_minute} head nods in the last minute."
            severity = 0.5
        else:
            state, value = STABLE, float(self.nods_last_minute)
            evidence = "Head position is steady."
        return IndicatorReading(
            name=IndicatorName.HEAD,
            t_seconds=t,
            available=True,
            state=state,
            value=value,
            severity=severity,
            confidence=0.9,
            evidence=(evidence,),
            unavailable_reason=None,
        )

    def _track_nod(self, t: float, pitch: float) -> None:
        config = self.config
        if pitch <= -config.nod_drop_degrees:
            if self._dip_start is None:
                self._dip_start = t
        elif self._dip_start is not None:
            dip = t - self._dip_start
            if config.nod_min_seconds <= dip <= config.nod_recovery_seconds:
                self._nods.append(t)
            self._dip_start = None
        while self._nods and self._nods[0] < t - NOD_WINDOW_SECONDS:
            self._nods.popleft()

    def _gap(self, t: float) -> IndicatorReading:
        last = self._last
        if (
            last is not None
            and last.available
            and self._last_valid_t is not None
            and within_gap(t - self._last_valid_t, self.config.bridge_gap_seconds)
        ):
            return replace(last, t_seconds=t, confidence=last.confidence * 0.5)
        self._clear_holds()
        return self._remember(
            unavailable_reading(
                IndicatorName.HEAD,
                t,
                "face_not_tracked",
                ("The face is not tracked, so head position is not measured.",),
            )
        )

    def _clear_holds(self) -> None:
        self._dip_start = None
        self._drop_since = None
        self._slump_since = None

    def _remember(self, reading: IndicatorReading) -> IndicatorReading:
        self._last = reading
        return reading
