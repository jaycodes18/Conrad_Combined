"""Did the driver respond to the in-car alert? A visual check, because the driver cannot tap while moving.

``product.json`` sets ``driver_input_while_moving`` to false, so a response is something the camera sees. When
an in-car alert is raised at t0, ``open_window(t0)`` starts a ``response_window_seconds`` (5 s) check. The driver
has responded when, held for ``hold_seconds`` (0.5 s):

* eye-based half: both eyes reach ``open_fraction`` (70%) of their calibrated open level, and the gaze is back
  at centre or the head is back from a drop;
* head-based half, used when the eyes cannot be seen (sunglasses): the head is back within
  ``head_return_degrees`` of its calibrated pitch and roll.

``value`` is the response latency in seconds for ``responded`` (from t0 to the start of the held response; the
Key Metrics latency), and the seconds elapsed for ``waiting``. States: ``idle``, ``waiting``, ``responded``,
``no_response``, ``unavailable`` (the face was not trackable for most of the window, or is not tracked now).
"""

from __future__ import annotations

from ..spec import INDICATOR_SPEC, ResponsivenessConfig
from .base import (
    DriverCalibration,
    FrameSignals,
    IndicatorName,
    IndicatorReading,
    finite,
    unavailable_reading,
)
from .eyes import eye_input_problem
from .head import relative_pose

IDLE = "idle"
WAITING = "waiting"
RESPONDED = "responded"
NO_RESPONSE = "no_response"
UNTRACKED = "unavailable"


class ResponsivenessIndicator:
    def __init__(
        self, config: ResponsivenessConfig = INDICATOR_SPEC.responsiveness, eye_visibility_cutoff: float | None = None
    ) -> None:
        self.config = config
        self.eye_visibility_cutoff = (
            INDICATOR_SPEC.eyes.eye_visibility_cutoff if eye_visibility_cutoff is None else eye_visibility_cutoff
        )
        self.reset()

    def reset(self) -> None:
        self.state = IDLE
        self.window_start: float | None = None
        self.latency_seconds: float | None = None
        self._hold_since: float | None = None
        self._tracked = 0
        self._untracked = 0

    def open_window(self, t0: float) -> bool:
        """Start a check at t0. Ignored (returns False) while a check is already waiting."""
        if self.state == WAITING:
            return False
        self.state = WAITING
        self.window_start = t0
        self.latency_seconds = None
        self._hold_since = None
        self._tracked = self._untracked = 0
        return True

    def push(self, frame: FrameSignals, calibration: DriverCalibration | None) -> IndicatorReading:
        t = frame.t_seconds
        if calibration is None:
            return unavailable_reading(
                IndicatorName.RESPONSIVENESS, t, "calibrating", ("Waiting for calibration before checking responses.",)
            )
        tracked = frame.face and finite(frame.pitch) and finite(frame.roll)
        if self.state in {WAITING, NO_RESPONSE}:
            if tracked:
                self._tracked += 1
                self._watch(t, frame, calibration)
            else:
                self._untracked += 1
                self._hold_since = None
            self._expire(t)
        if not tracked:
            return unavailable_reading(
                IndicatorName.RESPONSIVENESS,
                t,
                "face_not_tracked",
                ("The face is not tracked, so a response cannot be seen.",),
            )
        return self._reading(t)

    def _watch(self, t: float, frame: FrameSignals, calibration: DriverCalibration) -> None:
        if self._responding(frame, calibration):
            if self._hold_since is None:
                self._hold_since = t
            if t - self._hold_since >= self.config.hold_seconds and self.window_start is not None:
                self.state = RESPONDED
                self.latency_seconds = max(0.0, self._hold_since - self.window_start)
        else:
            self._hold_since = None

    def _expire(self, t: float) -> None:
        if self.state != WAITING or self.window_start is None:
            return
        if t - self.window_start < self.config.response_window_seconds:
            return
        seen = self._tracked + self._untracked
        if seen and self._untracked / seen > self.config.max_untracked_fraction:
            self.state = UNTRACKED
        else:
            self.state = NO_RESPONSE

    def _responding(self, frame: FrameSignals, calibration: DriverCalibration) -> bool:
        config = self.config
        pitch, roll = relative_pose(frame, calibration)
        head_back = pitch > -config.head_return_degrees and abs(roll) < config.head_return_degrees
        open_left, open_right = calibration.open_ear_left, calibration.open_ear_right
        if eye_input_problem(frame, self.eye_visibility_cutoff) is not None or open_left is None or open_right is None:
            return head_back  # head-based half only: the eyes cannot be seen, or were never calibrated
        eyes_open = (
            frame.ear_left >= config.open_fraction * open_left and frame.ear_right >= config.open_fraction * open_right
        )
        gaze_back = (
            finite(frame.gaze_horizontal)
            and abs(frame.gaze_horizontal - calibration.gaze_centre) <= config.gaze_centre_ratio
        )
        return eyes_open and (gaze_back or head_back)

    def _reading(self, t: float) -> IndicatorReading:
        state = self.state
        if state == UNTRACKED:
            return unavailable_reading(
                IndicatorName.RESPONSIVENESS,
                t,
                "face_not_tracked_during_check",
                ("The face was not trackable for most of the response check.",),
            )
        if state == RESPONDED and self.latency_seconds is not None:
            value: float | None = self.latency_seconds
            severity, evidence = 0.0, f"The driver responded {self.latency_seconds:.1f} s after the alert."
        elif state == WAITING and self.window_start is not None:
            value = t - self.window_start
            severity, evidence = 0.3, f"Checking for a response: {value:.1f} s since the alert."
        elif state == NO_RESPONSE:
            value = None
            severity = 0.9
            evidence = f"No visible response within {self.config.response_window_seconds:.0f} s of the alert."
        else:
            value, severity, evidence = None, 0.0, "No response check is running."
        return IndicatorReading(
            name=IndicatorName.RESPONSIVENESS,
            t_seconds=t,
            available=True,
            state=state,
            value=value,
            severity=severity,
            confidence=0.8,
            evidence=(evidence,),
            unavailable_reason=None,
        )
