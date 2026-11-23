"""Eyelid closure: how long the eyes have been shut, PERCLOS over 60 s, and the last blink's duration.

``value`` is the current continuous closure in seconds, counting each frame as one frame interval. Closure counts
only when the face is tracked, eye visibility is above the occlusion cutoff, and both eyes agree: one eye shut with
the other open is a wink or a covered eye, not closure. Each eye is compared with its own calibrated open level, so
camera distance cancels out. A closure episode survives up to ``bridge_gap_seconds`` (0.3 s) of lost frames; a
longer gap ends it as unavailable and never extends it. PERCLOS is P80, the share of frames at least 80% closed
(Wierwille et al., NHTSA DOT HS 808 247, 1994).

States: ``open``, ``blinking`` (closed under 2 s), ``closure_warning`` (2 s or more), ``closure_critical``
(5 s or more, the team's line for a clear indication), ``unavailable``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import replace

from ..spec import INDICATOR_SPEC, EyesConfig
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

OPEN = "open"
BLINKING = "blinking"
CLOSURE_WARNING = "closure_warning"
CLOSURE_CRITICAL = "closure_critical"
CLOSED_STATES = frozenset({CLOSURE_WARNING, CLOSURE_CRITICAL})


def eye_input_problem(frame: FrameSignals, cutoff: float) -> str | None:
    """Why this frame cannot support an eye reading, or None when it can."""
    if not frame.face:
        return "face_not_tracked"
    if frame.eye_visibility < cutoff:
        return "eyes_not_visible"
    if not (finite(frame.ear_left) and finite(frame.ear_right)):
        return "eye_landmarks_missing"
    return None


class EyesIndicator:
    """One per driving session. ``push`` once per frame, in time order."""

    def __init__(self, config: EyesConfig = INDICATOR_SPEC.eyes) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._episode_start: float | None = None
        self._frame_seconds = 0.0
        self._last_valid_t: float | None = None
        self._last: IndicatorReading | None = None
        self._perclos_samples: deque[tuple[float, bool]] = deque()
        self.perclos: float | None = None
        self.last_blink_seconds: float | None = None

    def push(self, frame: FrameSignals, calibration: DriverCalibration | None) -> IndicatorReading:
        t = frame.t_seconds
        if calibration is None:
            return self._remember(
                unavailable_reading(
                    IndicatorName.EYES, t, "calibrating", ("Learning this driver's open-eye level during calibration.",)
                )
            )
        problem = eye_input_problem(frame, self.config.eye_visibility_cutoff)
        if problem is not None:
            return self._gap(t, problem)
        if calibration.open_ear_left is None or calibration.open_ear_right is None:
            return self._remember(
                unavailable_reading(
                    IndicatorName.EYES,
                    t,
                    "eyes_not_calibrated",
                    ("The eyes were covered during calibration; recalibrate to measure closure.",),
                )
            )
        self._frame_seconds = 0.0 if self._last_valid_t is None else t - self._last_valid_t
        if self._last_valid_t is not None and not within_gap(self._frame_seconds, self.config.bridge_gap_seconds):
            self._episode_start = None  # the gap was too long to bridge: a new frame starts fresh
            self._frame_seconds = 0.0
        self._last_valid_t = t
        closure_left = 1.0 - frame.ear_left / calibration.open_ear_left
        closure_right = 1.0 - frame.ear_right / calibration.open_ear_right
        self._update_perclos(t, (closure_left + closure_right) / 2 >= self.config.perclos_fraction)
        closed_left = closure_left >= self.config.closed_fraction
        closed_right = closure_right >= self.config.closed_fraction
        disagree = abs(closure_left - closure_right) > self.config.eye_agreement_tolerance
        confidence = clamp01(min(1.0, frame.eye_visibility) * (1.0 - min(1.0, abs(closure_left - closure_right))))
        if closed_left and closed_right and not disagree:
            return self._remember(self._closed(t, confidence))
        evidence = []
        if closed_left != closed_right and disagree:
            evidence.append("One eye shut and the other open: treated as a wink or a covered eye, not closure.")
        self._end_episode(t)
        return self._remember(self._open(t, confidence, evidence))

    # ---- states ----------------------------------------------------------------------------------------
    def _closed(self, t: float, confidence: float) -> IndicatorReading:
        if self._episode_start is None:
            self._episode_start = t
        # Each frame stands for one frame interval, so 50 closed frames at 10 fps are 5.0 s, not 4.9 s.
        duration = t - self._episode_start + self._frame_seconds
        config = self.config
        if duration >= config.closure_critical_seconds:
            state = CLOSURE_CRITICAL
        elif duration >= config.closure_warning_seconds:
            state = CLOSURE_WARNING
        else:
            state = BLINKING
        evidence = [f"Eyes closed for {duration:.1f} s."]
        if self.perclos is not None:
            evidence.append(f"Eyes at least 80% closed {self.perclos:.0%} of the last 60 s.")
        return IndicatorReading(
            name=IndicatorName.EYES,
            t_seconds=t,
            available=True,
            state=state,
            value=duration,
            severity=max(clamp01(duration / config.closure_critical_seconds), self._perclos_severity()),
            confidence=confidence,
            evidence=tuple(evidence),
            unavailable_reason=None,
        )

    def _open(self, t: float, confidence: float, evidence: list[str]) -> IndicatorReading:
        if self.perclos is not None:
            evidence.append(f"Eyes at least 80% closed {self.perclos:.0%} of the last 60 s.")
        if self.last_blink_seconds is not None:
            evidence.append(f"Last closure lasted {self.last_blink_seconds:.2f} s.")
        return IndicatorReading(
            name=IndicatorName.EYES,
            t_seconds=t,
            available=True,
            state=OPEN,
            value=0.0,
            severity=self._perclos_severity(),
            confidence=confidence,
            evidence=tuple(evidence),
            unavailable_reason=None,
        )

    def _gap(self, t: float, reason: str) -> IndicatorReading:
        last = self._last
        bridging = (
            self._episode_start is not None
            and self._last_valid_t is not None
            and last is not None
            and last.available
            and within_gap(t - self._last_valid_t, self.config.bridge_gap_seconds)
        )
        if bridging and last is not None:
            return replace(
                last,
                t_seconds=t,
                confidence=last.confidence * 0.5,
                evidence=(*last.evidence, "Bridging a short gap in eye tracking."),
            )
        self._episode_start = None
        messages = {
            "face_not_tracked": "The face is not tracked, so eye closure is not measured.",
            "eyes_not_visible": "The eyes are not visible (sunglasses, glare or a hand), so closure is not measured.",
            "eye_landmarks_missing": "Eye landmarks are missing in this frame.",
        }
        return self._remember(unavailable_reading(IndicatorName.EYES, t, reason, (messages.get(reason, reason),)))

    # ---- helpers ---------------------------------------------------------------------------------------
    def _end_episode(self, t: float) -> None:
        if self._episode_start is not None:
            self.last_blink_seconds = t - self._episode_start
        self._episode_start = None

    def _update_perclos(self, t: float, closed80: bool) -> None:
        samples = self._perclos_samples
        samples.append((t, closed80))
        cutoff = t - self.config.perclos_window_seconds
        while samples and samples[0][0] < cutoff:
            samples.popleft()
        self.perclos = sum(flag for _t, flag in samples) / len(samples)

    def _perclos_severity(self) -> float:
        # PERCLOS of 0.15 or more is the usual drowsy line for P80; it alone never reaches an alert on its own.
        return 0.0 if self.perclos is None else clamp01(self.perclos / 0.15) * 0.4

    def _remember(self, reading: IndicatorReading) -> IndicatorReading:
        self._last = reading
        return reading
