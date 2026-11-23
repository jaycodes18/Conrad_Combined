"""Frame-by-frame monitoring for gaze, eyelid closure, and lost visibility.

The drowsiness model still scores 60-second windows. This tracker is the short-horizon layer the
driver screen needs: how long the gaze has been off centre, how long the eyes have been shut, and
whether the camera is actually usable. Missing frames are a camera fault, never evidence of
impairment. Thresholds are configurable so a research pilot can use different cutoffs from the
shipped 2-second eyes-off rule without changing that rule.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..errors import ConfigurationError
from ..spec import FEATURE_SPEC, FeatureSpec


class MonitoringState(StrEnum):
    """Whether the camera currently supports a behavior reading."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class MonitorConfig:
    """Durations are seconds. Closure uses the eye's own open level when calibration has one."""

    off_forward_gaze_seconds: float = 3.0
    prolonged_closure_seconds: float = 1.5
    invalid_after_seconds: float = 1.0
    recovery_seconds: float = 1.0
    rearm_seconds: float = 1.0
    closed_ear: float = 0.18
    closed_fraction_of_open: float = 0.5
    yaw_off_degrees: float = 25.0
    gaze_off_ratio: float = 0.15
    profile_name: str = "research_brief"

    def validate(self) -> None:
        for name, value in (
            ("off_forward_gaze_seconds", self.off_forward_gaze_seconds),
            ("prolonged_closure_seconds", self.prolonged_closure_seconds),
            ("invalid_after_seconds", self.invalid_after_seconds),
            ("recovery_seconds", self.recovery_seconds),
            ("rearm_seconds", self.rearm_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ConfigurationError(f"{name} must be a positive number of seconds")
        if not 0 < self.closed_fraction_of_open < 1:
            raise ConfigurationError("closed_fraction_of_open must be between 0 and 1")

    @staticmethod
    def research_brief() -> MonitorConfig:
        """November 2026 stationary-pilot cutoffs. Not a claim that the pilot was run."""
        config = MonitorConfig()
        config.validate()
        return config

    @staticmethod
    def product(spec: FeatureSpec = FEATURE_SPEC) -> MonitorConfig:
        """Match the shipped eyes-off duration. Closure stays a measured indicator, not a second model."""
        config = MonitorConfig(
            off_forward_gaze_seconds=float(spec.live_alerts.eyes_off_seconds),
            prolonged_closure_seconds=float(spec.thresholds.microsleep_seconds),
            yaw_off_degrees=float(spec.thresholds.yaw_off_degrees),
            gaze_off_ratio=float(spec.thresholds.gaze_off_ratio),
            profile_name="product",
        )
        config.validate()
        return config


@dataclass(frozen=True)
class MonitorEvent:
    kind: str
    t_seconds: float
    duration_seconds: float
    message: str

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "t": self.t_seconds,
            "score": self.duration_seconds,
            "message": self.message,
        }


@dataclass
class MonitorSnapshot:
    """One frame's monitoring reading. `warning_delay_seconds` is detector time, not hardware latency."""

    t_seconds: float
    state: MonitoringState
    face_visible: bool
    eyes_closed: bool
    looking_away: bool
    gaze_for_seconds: float
    closure_for_seconds: float
    invalid_for_seconds: float
    warning_delay_seconds: float
    events: list[MonitorEvent] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "t": self.t_seconds,
            "monitoring": self.state.value,
            "face_visible": self.face_visible,
            "eyes_closed": self.eyes_closed,
            "looking_away": self.looking_away,
            "gaze_for_seconds": self.gaze_for_seconds,
            "closure_for_seconds": self.closure_for_seconds,
            "invalid_for_seconds": self.invalid_for_seconds,
            "warning_delay_seconds": self.warning_delay_seconds,
        }


class TemporalTracker:
    """Holds short timers. Call `push` once per sample with a monotonic timestamp."""

    def __init__(self, config: MonitorConfig | None = None) -> None:
        self.config = config or MonitorConfig.research_brief()
        self.config.validate()
        self.state = MonitoringState.AVAILABLE
        self._invalid_since: float | None = None
        self._valid_since: float | None = None
        self._gaze_since: float | None = None
        self._closure_since: float | None = None
        self._forward_open_since: float | None = None
        self._gaze_armed = True
        self._closure_armed = True
        self._last_t: float | None = None

    def reset(self) -> None:
        self.state = MonitoringState.AVAILABLE
        self._invalid_since = None
        self._valid_since = None
        self._gaze_since = None
        self._closure_since = None
        self._forward_open_since = None
        self._gaze_armed = True
        self._closure_armed = True
        self._last_t = None

    def push(
        self,
        t_seconds: float,
        *,
        face: float,
        ear: float,
        yaw: float,
        gaze_horizontal: float,
        centre_yaw: float = 0.0,
        centre_gaze: float = 0.0,
        open_ear: float | None = None,
        camera_connected: bool = True,
        eye_visible: bool = True,
    ) -> MonitorSnapshot:
        if self._last_t is not None and t_seconds <= self._last_t:
            return self._fault(
                t_seconds,
                "timestamp did not increase; this frame is dropped and is not a behavior event",
            )
        self._last_t = t_seconds
        invalid = _input_invalid(face, ear, camera_connected, eye_visible)
        if invalid:
            return self._invalid(t_seconds, face_visible=face > 0)
        return self._valid(
            t_seconds,
            ear=ear,
            yaw=yaw,
            gaze_horizontal=gaze_horizontal,
            centre_yaw=centre_yaw,
            centre_gaze=centre_gaze,
            open_ear=open_ear,
        )

    def _fault(self, t_seconds: float, message: str) -> MonitorSnapshot:
        event = MonitorEvent("camera_fault", t_seconds, 0.0, message)
        return MonitorSnapshot(
            t_seconds=t_seconds,
            state=self.state,
            face_visible=False,
            eyes_closed=False,
            looking_away=False,
            gaze_for_seconds=0.0,
            closure_for_seconds=0.0,
            invalid_for_seconds=0.0,
            warning_delay_seconds=0.0,
            events=[event],
        )

    def _invalid(self, t_seconds: float, *, face_visible: bool) -> MonitorSnapshot:
        self._gaze_since = None
        self._closure_since = None
        self._forward_open_since = None
        self._valid_since = None
        self._invalid_since = t_seconds if self._invalid_since is None else self._invalid_since
        invalid_for = t_seconds - self._invalid_since
        events: list[MonitorEvent] = []
        if invalid_for >= self.config.invalid_after_seconds and self.state is MonitoringState.AVAILABLE:
            self.state = MonitoringState.UNAVAILABLE
            events.append(
                MonitorEvent(
                    "monitoring_unavailable",
                    t_seconds,
                    invalid_for,
                    "Camera or face is not usable. This is not a driver-behavior event.",
                )
            )
        return MonitorSnapshot(
            t_seconds=t_seconds,
            state=self.state,
            face_visible=face_visible,
            eyes_closed=False,
            looking_away=False,
            gaze_for_seconds=0.0,
            closure_for_seconds=0.0,
            invalid_for_seconds=invalid_for,
            warning_delay_seconds=0.0,
            events=events,
        )

    def _valid(
        self,
        t_seconds: float,
        *,
        ear: float,
        yaw: float,
        gaze_horizontal: float,
        centre_yaw: float,
        centre_gaze: float,
        open_ear: float | None,
    ) -> MonitorSnapshot:
        self._invalid_since = None
        events: list[MonitorEvent] = []
        if self.state is MonitoringState.UNAVAILABLE:
            self._valid_since = t_seconds if self._valid_since is None else self._valid_since
            if t_seconds - self._valid_since >= self.config.recovery_seconds:
                self.state = MonitoringState.AVAILABLE
                self._valid_since = None
                events.append(
                    MonitorEvent(
                        "monitoring_resumed",
                        t_seconds,
                        self.config.recovery_seconds,
                        "Face is visible again. Behavior timing starts over.",
                    )
                )
            else:
                return self._quiet(t_seconds, events)
        else:
            self._valid_since = None

        closed = _eyes_closed(ear, open_ear, self.config)
        away = _looking_away(yaw, gaze_horizontal, centre_yaw, centre_gaze, self.config) and not closed
        forward_open = not closed and not away
        self._arm(t_seconds, forward_open)
        gaze_for, closure_for, behavior = self._behavior(t_seconds, closed=closed, away=away)
        events.extend(behavior)
        return MonitorSnapshot(
            t_seconds=t_seconds,
            state=self.state,
            face_visible=True,
            eyes_closed=closed,
            looking_away=away,
            gaze_for_seconds=gaze_for,
            closure_for_seconds=closure_for,
            invalid_for_seconds=0.0,
            warning_delay_seconds=0.0,
            events=events,
        )

    def _quiet(self, t_seconds: float, events: list[MonitorEvent]) -> MonitorSnapshot:
        return MonitorSnapshot(
            t_seconds=t_seconds,
            state=self.state,
            face_visible=True,
            eyes_closed=False,
            looking_away=False,
            gaze_for_seconds=0.0,
            closure_for_seconds=0.0,
            invalid_for_seconds=0.0,
            warning_delay_seconds=0.0,
            events=events,
        )

    def _arm(self, t_seconds: float, forward_open: bool) -> None:
        if not forward_open:
            self._forward_open_since = None
            return
        self._forward_open_since = t_seconds if self._forward_open_since is None else self._forward_open_since
        if t_seconds - self._forward_open_since >= self.config.rearm_seconds:
            self._gaze_armed = True
            self._closure_armed = True

    def _behavior(self, t_seconds: float, *, closed: bool, away: bool) -> tuple[float, float, list[MonitorEvent]]:
        events: list[MonitorEvent] = []
        if closed:
            self._gaze_since = None
            self._closure_since = t_seconds if self._closure_since is None else self._closure_since
            closure_for = t_seconds - self._closure_since
            if closure_for >= self.config.prolonged_closure_seconds and self._closure_armed:
                self._closure_armed = False
                self._forward_open_since = None
                events.append(
                    MonitorEvent(
                        "research_closure",
                        t_seconds,
                        closure_for,
                        "Eyelids stayed shut past the configured closure threshold.",
                    )
                )
            return 0.0, closure_for, events
        self._closure_since = None
        if away:
            self._gaze_since = t_seconds if self._gaze_since is None else self._gaze_since
            gaze_for = t_seconds - self._gaze_since
            if gaze_for >= self.config.off_forward_gaze_seconds and self._gaze_armed:
                self._gaze_armed = False
                self._forward_open_since = None
                events.append(
                    MonitorEvent(
                        "research_gaze",
                        t_seconds,
                        gaze_for,
                        "Gaze stayed off the calibrated centre past the configured threshold.",
                    )
                )
            return gaze_for, 0.0, events
        self._gaze_since = None
        return 0.0, 0.0, events


def _input_invalid(face: float, ear: float, camera_connected: bool, eye_visible: bool) -> bool:
    if not camera_connected or not eye_visible or face <= 0:
        return True
    return not math.isfinite(ear)


def _eyes_closed(ear: float, open_ear: float | None, config: MonitorConfig) -> bool:
    if open_ear is not None and math.isfinite(open_ear) and open_ear > 0:
        return ear < open_ear * config.closed_fraction_of_open
    return ear < config.closed_ear


def _looking_away(
    yaw: float,
    gaze_horizontal: float,
    centre_yaw: float,
    centre_gaze: float,
    config: MonitorConfig,
) -> bool:
    yaw_off = yaw - centre_yaw if math.isfinite(yaw) else 0.0
    gaze_off = gaze_horizontal - centre_gaze if math.isfinite(gaze_horizontal) else 0.0
    return abs(yaw_off) > config.yaw_off_degrees or abs(gaze_off) > config.gaze_off_ratio
