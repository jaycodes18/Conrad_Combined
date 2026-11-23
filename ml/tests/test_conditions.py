"""OpenCV preparation and the condition test's verdicts, checked without downloading a face model."""

from __future__ import annotations

import math

import numpy as np
import pytest

from driven_ml.features.preprocess import (
    Preprocess,
    crop_region,
    distort_fisheye,
    enhance_low_light,
    mean_luminance,
    parse_region,
    undistort_fisheye,
)
from driven_ml.studio.conditions import (
    CONDITIONS,
    DEGRADED,
    FAILS,
    WORKS,
    _measure,
    _Tally,
)


def gradient_image(height: int = 240, width: int = 320, scale: float = 1.0) -> np.ndarray:
    xs = np.linspace(0, 255, width)[None, :, None]
    ys = np.linspace(0, 255, height)[:, None, None]
    image = (xs * 0.6 + ys * 0.4) * np.ones((1, 1, 3)) * scale
    return np.clip(image, 0, 255).astype(np.uint8)


def test_low_light_enhancement_brightens_dark_frames_and_leaves_bright_ones_alone() -> None:
    dark = gradient_image(scale=0.15)
    bright = gradient_image()
    assert mean_luminance(enhance_low_light(dark)) > 2 * mean_luminance(dark)
    assert np.array_equal(enhance_low_light(bright), bright)
    assert np.array_equal(dark, gradient_image(scale=0.15)), "the input must not be modified"


def test_region_crop_and_validation() -> None:
    image = gradient_image(200, 400)
    crop = crop_region(image, (0.5, 0.25, 1.0, 0.75))
    assert crop.shape == (100, 200, 3)
    assert np.array_equal(crop, image[50:150, 200:400])
    assert parse_region("0.5, 0.1, 1, 0.9") == (0.5, 0.1, 1.0, 0.9)
    with pytest.raises(ValueError, match="four"):
        parse_region("0.5,0.1,1")
    for bad in ("0.6,0.1,0.5,0.9", "0,0,1.2,1"):
        with pytest.raises(ValueError, match="fractions must satisfy"):
            parse_region(bad)


def test_fisheye_correction_inverts_the_fisheye_model_near_the_centre() -> None:
    image = gradient_image(360, 480)
    round_trip = undistort_fisheye(distort_fisheye(image, 210.0), 210.0)
    centre = (slice(120, 240), slice(160, 320))
    error = np.abs(round_trip[centre].astype(float) - image[centre].astype(float))
    assert error.mean() < 3.0
    with pytest.raises(ValueError, match="field of view"):
        undistort_fisheye(image, 300.0)


def test_preprocess_order_and_default() -> None:
    image = gradient_image(240, 320, scale=0.2)
    assert not Preprocess().active
    assert np.array_equal(Preprocess().apply(image), image)
    prepared = Preprocess(enhance_low_light=True, region=(0.0, 0.0, 0.5, 0.5)).apply(image)
    assert prepared.shape == (120, 160, 3)
    assert mean_luminance(prepared) > mean_luminance(crop_region(image, (0.0, 0.0, 0.5, 0.5)))


def tally(
    found: int,
    total: int,
    contrast: float,
    mouth: float = 10.0,
    closed: float | None = None,
) -> _Tally:
    result = _Tally(found=found, total=total)
    result.contrast = [contrast] * found
    result.mouth = [mouth] * found
    result.ear_open = [0.30] * found
    result.ear_closed = [closed] * 3 if closed is not None else []
    result.yaw = [0.0] * found
    return result


def test_verdicts() -> None:
    clean = (10.0, 10.0)
    assert _measure(tally(10, 10, 10.0), clean, closed_supplied=False).verdict == WORKS
    assert _measure(tally(5, 10, 10.0), clean, closed_supplied=False).verdict == FAILS  # below 60% face
    assert _measure(tally(9, 10, 10.0), clean, closed_supplied=False).verdict == DEGRADED  # 90% face
    hidden_eyes = _measure(tally(10, 10, 1.0, closed=0.05), clean, closed_supplied=True)
    assert hidden_eyes.verdict == FAILS
    assert math.isnan(hidden_eyes.ear_open)
    assert math.isnan(hidden_eyes.ear_closed)
    assert math.isnan(hidden_eyes.closure_separation)
    assert _measure(tally(10, 10, 4.0), clean, closed_supplied=False).verdict == DEGRADED
    assert _measure(tally(10, 10, 10.0, mouth=1.0), clean, closed_supplied=False).verdict == DEGRADED
    blinks_lost = _measure(tally(10, 10, 10.0, closed=0.20), clean, closed_supplied=True)
    assert blinks_lost.verdict == FAILS
    assert math.isclose(blinks_lost.closure_separation, 1 - 0.20 / 0.30)
    assert _measure(tally(10, 10, 10.0, closed=0.05), clean, closed_supplied=True).verdict == WORKS


def test_every_condition_transform_keeps_an_rgb_image() -> None:
    image = gradient_image(240, 240)
    points = np.column_stack([np.linspace(60, 180, 478), np.linspace(60, 200, 478)])
    for condition in CONDITIONS:
        context = {"points": points}
        out = condition.transform(image, np.random.default_rng(0), context)
        assert out.dtype == np.uint8, condition.key
        assert out.ndim == 3, condition.key
        assert out.shape[2] == 3, condition.key
        if condition.fix is not None:
            fixed = condition.fix(context).apply(out)
            assert fixed.dtype == np.uint8, condition.key
            assert fixed.size > 0, condition.key
