"""The six indicator submodels run together on one driving session, plus the eyes-off-road timer.

``IndicatorSuite`` owns the calibration: while ``calibrating`` is true it keeps the frames; on the first frame
after, it learns the driver's ``DriverCalibration`` and the ocular baseline from them. Each frame then yields one
reading per indicator. The eyes-off timer is the NHTSA visual-manual rule (78 FR 24818, 2013): the gaze or head
off the driver's calibrated centre for more than ``LiveAlertRules.eyes_off_seconds`` (2 s). It is a fixed rule,
not a model; Studio's live alert and the fusion both read it from here.
"""

from __future__ import annotations

import math

import numpy as np

from ..spec import FEATURE_SPEC, INDICATOR_SPEC, FeatureSpec, IndicatorSpec
from ..studio.audio_health import AudioHealthReport
from .base import DriverCalibration, FrameSignals, IndicatorName, IndicatorReading, within_gap
from .breathing import BreathingIndicator
from .eyes import EyesIndicator
from .head import HeadIndicator
from .ocular import OcularIndicator
from .responsiveness import ResponsivenessIndicator
from .sensor_quality import CameraHealth, SensorQualityIndicator

MAX_CALIBRATION_FRAMES = 6000  # 10 minutes at 10 fps; calibration is 2 minutes


def looking_away(frame: FrameSignals, calibration: DriverCalibration, spec: FeatureSpec = FEATURE_SPEC) -> bool:
    """Gaze or head yaw off the calibrated centre by more than the eyes-off thresholds."""
    if not frame.face:
        return False
    thresholds = spec.thresholds
    gaze = frame.gaze_horizontal - calibration.gaze_centre
    yaw = frame.yaw - calibration.yaw_centre
    return bool(
        (math.isfinite(gaze) and abs(gaze) > thresholds.gaze_off_ratio)
        or (math.isfinite(yaw) and abs(yaw) > thresholds.yaw_off_degrees)
    )


class EyesOffTimer:
    """Seconds the driver has continuously looked away; 0 when looking ahead.

    A frame without a usable face inside ``bridge_seconds`` of the last away frame keeps the count, the same
    0.3 s bridge the eye and head indicators use, so one dropped frame does not restart the 2 s rule.
    """

    def __init__(self, bridge_seconds: float = INDICATOR_SPEC.eyes.bridge_gap_seconds) -> None:
        self.bridge_seconds = bridge_seconds
        self.since: float | None = None
        self._last_away: float | None = None

    def push(self, t_seconds: float, away: bool, *, tracked: bool = True) -> float:
        if away:
            if self.since is None:
                self.since = t_seconds
            self._last_away = t_seconds
            return t_seconds - self.since
        bridging = (
            not tracked
            and self.since is not None
            and self._last_away is not None
            and within_gap(t_seconds - self._last_away, self.bridge_seconds)
        )
        if bridging and self.since is not None:
            return t_seconds - self.since
        self.reset()
        return 0.0

    def reset(self) -> None:
        self.since = None
        self._last_away = None


class IndicatorSuite:
    def __init__(self, spec: IndicatorSpec = INDICATOR_SPEC, *, audio_allowed: bool = False) -> None:
        cutoff = spec.eyes.eye_visibility_cutoff
        self.spec = spec
        self.eyes = EyesIndicator(spec.eyes)
        self.ocular = OcularIndicator(spec.ocular, eye_visibility_cutoff=cutoff)
        self.head = HeadIndicator(spec.head)
        self.breathing = BreathingIndicator(spec.breathing, audio_allowed=audio_allowed)
        self.responsiveness = ResponsivenessIndicator(spec.responsiveness, eye_visibility_cutoff=cutoff)
        self.sensor_quality = SensorQualityIndicator(spec.sensor_quality, eye_visibility_cutoff=cutoff)
        self.eyes_off = EyesOffTimer()
        self.calibration: DriverCalibration | None = None
        self._calibration_frames: list[FrameSignals] = []
        self.eyes_off_seconds = 0.0

    def reset(self) -> None:
        """Start over, including calibration (Studio's Recalibrate)."""
        for indicator in (self.eyes, self.ocular, self.head, self.breathing, self.responsiveness):
            indicator.reset()
        self.ocular.baseline = None
        self.eyes_off.reset()
        self.calibration = None
        self._calibration_frames = []
        self.eyes_off_seconds = 0.0

    def set_calibration(self, calibration: DriverCalibration, frames: list[FrameSignals]) -> None:
        """Use a calibration learned elsewhere (a saved driver profile, or Studio's own baseline)."""
        self.calibration = calibration
        self.ocular.fit_baseline(frames, calibration)
        self._calibration_frames = []

    def push_frame(
        self, frame: FrameSignals, camera: CameraHealth, *, calibrating: bool
    ) -> dict[IndicatorName, IndicatorReading]:
        if calibrating:
            if len(self._calibration_frames) < MAX_CALIBRATION_FRAMES:
                self._calibration_frames.append(frame)
        elif self.calibration is None and self._calibration_frames:
            learned = DriverCalibration.from_frames(self._calibration_frames, self.spec.eyes.eye_visibility_cutoff)
            if learned is not None:
                self.set_calibration(learned, self._calibration_frames)
            self._calibration_frames = []
        calibration = None if calibrating else self.calibration
        readings = {
            IndicatorName.EYES: self.eyes.push(frame, calibration),
            IndicatorName.OCULAR: self.ocular.push(frame, calibration),
            IndicatorName.HEAD: self.head.push(frame, calibration),
            IndicatorName.BREATHING: self.breathing.push_frame(frame),
            IndicatorName.RESPONSIVENESS: self.responsiveness.push(frame, calibration),
        }
        readings[IndicatorName.SENSOR_QUALITY] = self.sensor_quality.push(frame, camera, readings)
        away = (
            calibration is not None and camera.connected and not camera.frame_fault and looking_away(frame, calibration)
        )
        self.eyes_off_seconds = self.eyes_off.push(frame.t_seconds, away, tracked=frame.face)
        return readings

    def push_audio(
        self, t_seconds: float, samples: np.ndarray, health: AudioHealthReport, *, driving: bool
    ) -> IndicatorReading:
        return self.breathing.push_audio(t_seconds, samples, health, driving=driving)
