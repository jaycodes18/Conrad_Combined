"""What the sensors can support right now: per-indicator coverage and the named situations behind it.

It combines what already exists: face found, eye and mouth visibility, luminance, blur, frame-gap faults and
microphone health. It does not gate the other indicators (each gates itself on its own inputs with the same
cutoffs); it names why coverage is what it is, so the product keeps working with sunglasses instead of failing:
eyes, ocular and the eye-based half of responsiveness go unavailable, and head, breathing and the head-based
half of responsiveness keep running.

``coverage`` maps each behaviour indicator to ``available``, ``degraded`` or ``unavailable``. ``value`` is the
share of behaviour indicators available (degraded counts half). States: ``ok``, ``degraded``, ``fault`` (the
camera itself is faulted). The reading is always available: it describes the sensors, it does not use them.

Situations are a closed set. The team's list plus three this change needed to name every condition in the
regression suite: ``image_blur``, ``fisheye_distortion`` and ``lower_face_occluded`` (a hand or mask over the
mouth; the eyes still work).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from ..spec import FEATURE_SPEC, INDICATOR_SPEC, SensorQualityConfig
from .base import BEHAVIOUR_INDICATORS, FrameSignals, IndicatorName, IndicatorReading, clamp01
from .breathing import CABIN_NOISE_REASONS, BreathingUnavailable


class Coverage(StrEnum):
    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"


class Situation(StrEnum):
    SUNGLASSES_OR_EYE_OCCLUSION = "sunglasses_or_eye_occlusion"
    CLEAR_GLASSES_GLARE = "clear_glasses_glare"
    LOW_LIGHT = "low_light"
    IR_MODE = "ir_mode"
    BACKLIT = "backlit"
    FACE_OUT_OF_FRAME = "face_out_of_frame"
    CAMERA_FAULT = "camera_fault"
    MIC_BLOCKED_BY_POLICY = "mic_blocked_by_policy"
    MIC_FAULT = "mic_fault"
    CABIN_TOO_NOISY = "cabin_too_noisy"
    IMAGE_BLUR = "image_blur"
    FISHEYE_DISTORTION = "fisheye_distortion"
    LOWER_FACE_OCCLUDED = "lower_face_occluded"


# Situations that are reported but are the default product state, so they do not by themselves make
# monitoring "degraded": the microphone is off unless a written legal review turns it on.
EXPECTED_SITUATIONS = frozenset({Situation.MIC_BLOCKED_BY_POLICY})

# Which camera indicators each situation weakens while they still produce readings.
_DEGRADES: dict[Situation, tuple[IndicatorName, ...]] = {
    Situation.CLEAR_GLASSES_GLARE: (IndicatorName.EYES, IndicatorName.OCULAR),
    Situation.LOW_LIGHT: (IndicatorName.EYES, IndicatorName.OCULAR),
    Situation.IR_MODE: (IndicatorName.OCULAR,),
    Situation.BACKLIT: (IndicatorName.EYES, IndicatorName.OCULAR),
    Situation.IMAGE_BLUR: (IndicatorName.EYES, IndicatorName.OCULAR),
    Situation.FISHEYE_DISTORTION: (IndicatorName.OCULAR, IndicatorName.HEAD),
    Situation.SUNGLASSES_OR_EYE_OCCLUSION: (IndicatorName.RESPONSIVENESS,),
}

_SENTENCES: dict[Situation, str] = {
    Situation.SUNGLASSES_OR_EYE_OCCLUSION: "The eyes are covered (sunglasses or a hand); head monitoring continues.",
    Situation.CLEAR_GLASSES_GLARE: "Glare on glasses hides part of the eyes.",
    Situation.LOW_LIGHT: "The cabin is too dark for a clear view of the eyes.",
    Situation.IR_MODE: "The camera is in infrared mode.",
    Situation.BACKLIT: "Bright light behind the driver darkens the face.",
    Situation.FACE_OUT_OF_FRAME: "The face is out of frame.",
    Situation.CAMERA_FAULT: "The camera is disconnected or dropping frames.",
    Situation.MIC_BLOCKED_BY_POLICY: "The microphone is off by policy.",
    Situation.MIC_FAULT: "The microphone signal is not usable.",
    Situation.CABIN_TOO_NOISY: "The cabin is too noisy to hear breathing.",
    Situation.IMAGE_BLUR: "The image is blurred.",
    Situation.FISHEYE_DISTORTION: "A fisheye lens distorts the face.",
    Situation.LOWER_FACE_OCCLUDED: "The lower face is covered; eye monitoring continues.",
}


@dataclass(frozen=True)
class CameraHealth:
    """Image statistics for one frame. Lumas are 0..1; ``sharpness`` is the variance of the Laplacian."""

    connected: bool = True
    frame_fault: bool = False
    mean_luma: float | None = None
    face_luma: float | None = None
    sharpness: float | None = None
    glare_fraction: float = 0.0
    grayscale: bool = False
    fisheye: bool = False

    @classmethod
    def from_rgb(
        cls,
        rgb: np.ndarray | None,
        points: np.ndarray | None = None,
        *,
        frame_fault: bool = False,
        fisheye: bool = False,
    ) -> CameraHealth:
        """Statistics of an RGB frame and, when landmarks are given, of the face and eye regions."""
        if rgb is None:
            return cls(frame_fault=frame_fault, fisheye=fisheye)
        import cv2

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        channels = rgb.astype(np.int16)
        grayscale = (
            float(np.mean(np.abs(channels[..., 0] - channels[..., 1]))) < 2.0
            and float(np.mean(np.abs(channels[..., 1] - channels[..., 2]))) < 2.0
        )
        face_luma = None
        glare = 0.0
        if points is not None and len(points) > max(FEATURE_SPEC.landmarks.left_eye):
            face_luma = _region_mean(gray, points, range(len(points)))
            landmarks = FEATURE_SPEC.landmarks
            eye_values = [_region_pixels(gray, points, ring) for ring in (landmarks.left_eye, landmarks.right_eye)]
            pixels = np.concatenate([values for values in eye_values if values.size])
            glare = float(np.mean(pixels >= 250)) if pixels.size else 0.0
        return cls(
            connected=True,
            frame_fault=frame_fault,
            mean_luma=float(np.mean(gray)) / 255.0,
            face_luma=face_luma,
            sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            glare_fraction=glare,
            grayscale=grayscale,
            fisheye=fisheye,
        )


def _region_pixels(gray: np.ndarray, points: np.ndarray, indices: Any) -> np.ndarray:
    selected = np.asarray([points[index] for index in indices], dtype=float)
    height, width = gray.shape[:2]
    x0, y0 = np.clip(selected.min(axis=0).astype(int), 0, [width - 1, height - 1])
    x1, y1 = np.clip(selected.max(axis=0).astype(int) + 1, 1, [width, height])
    return gray[y0:y1, x0:x1].reshape(-1)


def _region_mean(gray: np.ndarray, points: np.ndarray, indices: Any) -> float | None:
    pixels = _region_pixels(gray, points, indices)
    return float(np.mean(pixels)) / 255.0 if pixels.size else None


def camera_situations(
    frame: FrameSignals, camera: CameraHealth, config: SensorQualityConfig, eye_visibility_cutoff: float
) -> list[Situation]:
    if not camera.connected or camera.frame_fault:
        return [Situation.CAMERA_FAULT]
    found: list[Situation] = []
    if not frame.face:
        found.append(Situation.FACE_OUT_OF_FRAME)
    else:
        if frame.eye_visibility < eye_visibility_cutoff:
            found.append(Situation.SUNGLASSES_OR_EYE_OCCLUSION)
        if frame.mouth_visibility < config.mouth_visibility_cutoff:
            found.append(Situation.LOWER_FACE_OCCLUDED)
        if camera.glare_fraction >= config.glare_fraction:
            found.append(Situation.CLEAR_GLASSES_GLARE)
    if camera.grayscale:
        found.append(Situation.IR_MODE)
    if camera.mean_luma is not None and camera.mean_luma < config.low_light_luma:
        found.append(Situation.LOW_LIGHT)
    if (
        camera.mean_luma is not None
        and camera.face_luma is not None
        and camera.mean_luma - camera.face_luma >= config.backlit_luma_delta
    ):
        found.append(Situation.BACKLIT)
    if camera.sharpness is not None and camera.sharpness < config.blur_sharpness:
        found.append(Situation.IMAGE_BLUR)
    if camera.fisheye:
        found.append(Situation.FISHEYE_DISTORTION)
    return found


def microphone_situation(breathing: IndicatorReading) -> Situation | None:
    reason = breathing.unavailable_reason
    if reason == BreathingUnavailable.MIC_BLOCKED_BY_POLICY:
        return Situation.MIC_BLOCKED_BY_POLICY
    if reason == BreathingUnavailable.MIC_FAULT:
        return Situation.MIC_FAULT
    if reason is not None and BreathingUnavailable(reason) in CABIN_NOISE_REASONS:
        return Situation.CABIN_TOO_NOISY
    return None


class SensorQualityIndicator:
    def __init__(
        self, config: SensorQualityConfig = INDICATOR_SPEC.sensor_quality, eye_visibility_cutoff: float | None = None
    ) -> None:
        self.config = config
        self.eye_visibility_cutoff = (
            INDICATOR_SPEC.eyes.eye_visibility_cutoff if eye_visibility_cutoff is None else eye_visibility_cutoff
        )
        self.coverage: dict[IndicatorName, Coverage] = dict.fromkeys(BEHAVIOUR_INDICATORS, Coverage.UNAVAILABLE)
        self.situations: tuple[Situation, ...] = ()

    def push(
        self,
        frame: FrameSignals,
        camera: CameraHealth,
        readings: Mapping[IndicatorName, IndicatorReading],
    ) -> IndicatorReading:
        """``readings`` are this frame's readings of the five behaviour indicators."""
        situations = camera_situations(frame, camera, self.config, self.eye_visibility_cutoff)
        breathing = readings.get(IndicatorName.BREATHING)
        if breathing is not None and (mic := microphone_situation(breathing)) is not None:
            situations.append(mic)
        weakened = {name for situation in situations for name in _DEGRADES.get(situation, ())}
        coverage: dict[IndicatorName, Coverage] = {}
        for name in BEHAVIOUR_INDICATORS:
            reading = readings.get(name)
            if reading is None or not reading.available:
                coverage[name] = Coverage.UNAVAILABLE
            elif name in weakened or reading.confidence < self.config.degraded_confidence:
                coverage[name] = Coverage.DEGRADED
            else:
                coverage[name] = Coverage.AVAILABLE
        self.coverage = coverage
        self.situations = tuple(situations)
        share = sum(
            1.0 if value is Coverage.AVAILABLE else 0.5 if value is Coverage.DEGRADED else 0.0
            for value in coverage.values()
        ) / len(coverage)
        if Situation.CAMERA_FAULT in situations:
            state = "fault"
        elif set(situations) - EXPECTED_SITUATIONS:
            state = "degraded"
        else:
            state = "ok"
        evidence = tuple(_SENTENCES[situation] for situation in situations) or ("All sensors usable.",)
        return IndicatorReading(
            name=IndicatorName.SENSOR_QUALITY,
            t_seconds=frame.t_seconds,
            available=True,
            state=state,
            value=clamp01(share),
            severity=0.0,
            confidence=1.0,
            evidence=evidence,
            unavailable_reason=None,
        )

    def degraded_situations(self) -> tuple[Situation, ...]:
        """The situations that count as degraded monitoring (everything but the default mic-off state)."""
        return tuple(situation for situation in self.situations if situation not in EXPECTED_SITUATIONS)
