"""Sensor quality: named situations, per-indicator coverage, and what keeps running when the eyes are hidden."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import CLOSED_EAR, make_frame

from driven_ml.indicators import CameraHealth, Coverage, IndicatorName, IndicatorSuite, Situation

CLEAR = CameraHealth(mean_luma=0.5, face_luma=0.5, sharpness=100.0)


def calibrated_suite() -> IndicatorSuite:
    suite = IndicatorSuite()
    for index in range(1200):
        suite.push_frame(make_frame(index / 10), CLEAR, calibrating=True)
    suite.push_frame(make_frame(120.0), CLEAR, calibrating=False)
    return suite


def test_sunglasses_keep_head_and_the_head_based_response_running() -> None:
    suite = calibrated_suite()
    readings = suite.push_frame(make_frame(120.1, eye_visibility=0.1, ear=CLOSED_EAR), CLEAR, calibrating=False)
    quality = suite.sensor_quality
    assert Situation.SUNGLASSES_OR_EYE_OCCLUSION in quality.situations
    assert quality.coverage[IndicatorName.EYES] is Coverage.UNAVAILABLE
    assert quality.coverage[IndicatorName.OCULAR] is Coverage.UNAVAILABLE
    assert quality.coverage[IndicatorName.HEAD] is Coverage.AVAILABLE
    assert quality.coverage[IndicatorName.RESPONSIVENESS] is Coverage.DEGRADED
    assert readings[IndicatorName.SENSOR_QUALITY].state == "degraded"
    assert readings[IndicatorName.EYES].severity == 0.0


@pytest.mark.parametrize(
    ("camera", "situation"),
    [
        (CameraHealth(mean_luma=0.05, face_luma=0.05, sharpness=100.0), Situation.LOW_LIGHT),
        (CameraHealth(mean_luma=0.85, face_luma=0.3, sharpness=100.0), Situation.BACKLIT),
        (CameraHealth(mean_luma=0.4, face_luma=0.4, sharpness=100.0, grayscale=True), Situation.IR_MODE),
        (
            CameraHealth(mean_luma=0.5, face_luma=0.5, sharpness=100.0, glare_fraction=0.2),
            Situation.CLEAR_GLASSES_GLARE,
        ),
        (CameraHealth(mean_luma=0.5, face_luma=0.5, sharpness=3.0), Situation.IMAGE_BLUR),
        (CameraHealth(mean_luma=0.5, face_luma=0.5, sharpness=100.0, fisheye=True), Situation.FISHEYE_DISTORTION),
    ],
)
def test_camera_conditions_are_named(camera: CameraHealth, situation: Situation) -> None:
    suite = calibrated_suite()
    suite.push_frame(make_frame(120.1), camera, calibrating=False)
    assert situation in suite.sensor_quality.degraded_situations()


def test_camera_fault_face_out_of_frame_and_lower_face() -> None:
    suite = calibrated_suite()
    readings = suite.push_frame(make_frame(120.1), CameraHealth(connected=False), calibrating=False)
    assert suite.sensor_quality.situations == (Situation.CAMERA_FAULT, Situation.MIC_BLOCKED_BY_POLICY)
    assert readings[IndicatorName.SENSOR_QUALITY].state == "fault"
    suite.push_frame(make_frame(120.2, face=False), CLEAR, calibrating=False)
    assert Situation.FACE_OUT_OF_FRAME in suite.sensor_quality.situations
    suite.push_frame(make_frame(120.3, mouth_visibility=0.1), CLEAR, calibrating=False)
    assert Situation.LOWER_FACE_OCCLUDED in suite.sensor_quality.situations
    assert suite.sensor_quality.coverage[IndicatorName.EYES] is Coverage.AVAILABLE


def test_the_default_mic_off_state_is_named_but_not_degraded() -> None:
    suite = calibrated_suite()
    readings = suite.push_frame(make_frame(120.1), CLEAR, calibrating=False)
    assert suite.sensor_quality.situations == (Situation.MIC_BLOCKED_BY_POLICY,)
    assert suite.sensor_quality.degraded_situations() == ()
    assert readings[IndicatorName.SENSOR_QUALITY].state == "ok"
    assert suite.sensor_quality.coverage[IndicatorName.BREATHING] is Coverage.UNAVAILABLE


def test_camera_health_from_an_image() -> None:
    dark_gray = np.full((120, 160, 3), 10, dtype=np.uint8)
    health = CameraHealth.from_rgb(dark_gray)
    assert health.grayscale
    assert health.mean_luma is not None
    assert health.mean_luma < 0.12
    assert health.sharpness == pytest.approx(0.0)
    rng = np.random.default_rng(0)
    colour = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    assert not CameraHealth.from_rgb(colour).grayscale
    assert CameraHealth.from_rgb(None).mean_luma is None
