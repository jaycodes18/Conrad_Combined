"""Indicator submodels: short-horizon readings of eyes, eye movement, head, breathing, response and sensors.

This is product ML, not Studio code: Studio, the contract fixtures and the risk model all consume it. Each
submodel is its own module, keeps its own short history, reads its thresholds from ``spec.INDICATOR_SPEC``,
and emits one ``IndicatorReading`` per push. Missing, occluded or noisy input is ``available=False`` with
severity 0, never a high reading.
"""

from __future__ import annotations

from .base import (
    BEHAVIOUR_INDICATORS,
    DriverCalibration,
    FrameSignals,
    IndicatorName,
    IndicatorReading,
    unavailable_reading,
)
from .breathing import BreathingIndicator, BreathingUnavailable
from .eyes import EyesIndicator
from .head import HeadIndicator
from .ocular import OcularIndicator
from .responsiveness import ResponsivenessIndicator
from .sensor_quality import CameraHealth, Coverage, SensorQualityIndicator, Situation
from .suite import EyesOffTimer, IndicatorSuite, looking_away

__all__ = [
    "BEHAVIOUR_INDICATORS",
    "BreathingIndicator",
    "BreathingUnavailable",
    "CameraHealth",
    "Coverage",
    "DriverCalibration",
    "EyesIndicator",
    "EyesOffTimer",
    "FrameSignals",
    "HeadIndicator",
    "IndicatorName",
    "IndicatorReading",
    "IndicatorSuite",
    "OcularIndicator",
    "ResponsivenessIndicator",
    "SensorQualityIndicator",
    "Situation",
    "looking_away",
    "unavailable_reading",
]
