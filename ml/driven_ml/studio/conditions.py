"""Condition test: which in-car conditions the face pipeline survives, measured, not assumed.

Each condition is an OpenCV transform of real face frames: your webcam, a video, or photos. Every
transformed frame goes through MediaPipe Face Landmarker and ``frame_features``, and is scored on four
things the model depends on:

* detection: the share of frames with a face. Under ``min_face_fraction`` (0.6), windows are not scored
  at all.
* eye visibility: luminance contrast of the eye region against the clean frame. Closure is measured from
  the eyes; when the eyes are not visible (sunglasses, darkness), MediaPipe still places eye landmarks, but
  it is guessing, so closures cannot be seen.
* closure separation: when eyes-closed frames are supplied, ``1 - EAR_closed / EAR_open``. The features
  count a blink at 0.5 and PERCLOS at 0.8, so separation must clear those with margin.
* head-angle noise: the spread of yaw across frames that should be identical. The eyes-off rule fires at
  25 degrees from the driver's centre.

Each condition is run twice: as captured, and with the OpenCV fix in ``features.preprocess`` that targets
it, where one exists. Synthetic transforms approximate reality; they are not a substitute for recording
in the car. Run: ``python -m driven_ml.studio.conditions --source 0``.

``FEATURE_CONDITIONS`` are the same conditions one level up, as their effect on the frame signals and camera
statistics the indicators read (eye and mouth visibility, landmark noise, luminance, sharpness, glare, IR,
fisheye). They let the regression suite run the whole indicator, fusion and escalation stack on a scripted
synthetic session under each condition without MediaPipe. They encode what the image transforms above
measure, not what a real cabin does.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..features.geometry import frame_features
from ..features.preprocess import Image, Preprocess, distort_fisheye
from ..indicators import CameraHealth, Situation
from ..spec import FEATURE_SPEC, INDICATOR_SPEC, FeatureSpec

logger = logging.getLogger(__name__)

WORKS, DEGRADED, FAILS = "works", "degraded", "fails"
MIN_DETECTION = 0.95
FAIL_DETECTION = FEATURE_SPEC.thresholds.min_face_fraction
MIN_EYE_VISIBILITY = 0.5
FAIL_EYE_VISIBILITY = INDICATOR_SPEC.eyes.eye_visibility_cutoff  # the indicators' occlusion cutoff
MIN_SEPARATION = 0.65  # blinks count at 0.5 closure, PERCLOS at 0.8: keep a margin over 0.5
FAIL_SEPARATION = 0.5
MAX_YAW_SPREAD_DEGREES = 5.0
EYE_PATCH_PAD = 0.6
WIDE_FRAME = (1280, 720)
TINY_FACE_SCALE = 0.22
SIDE_SQUEEZE = 0.6
FISHEYE_FOV = 210.0
FISHEYE_FACE_SCALE = 0.33
LEFT_EYE_RING = (362, 385, 387, 263, 373, 380)
RIGHT_EYE_RING = (33, 160, 158, 133, 153, 144)
# Full outlines from the MediaPipe face mesh, in order around each region.
EYE_OUTLINE_LEFT = (
    362,
    382,
    381,
    380,
    374,
    373,
    390,
    249,
    263,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
)
EYE_OUTLINE_RIGHT = (
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    173,
    157,
    158,
    159,
    160,
    161,
    246,
)
MOUTH_OUTLINE = (
    61,
    146,
    91,
    181,
    84,
    17,
    314,
    405,
    321,
    375,
    291,
    409,
    270,
    269,
    267,
    0,
    37,
    39,
    40,
    185,
)
MIN_MOUTH_VISIBILITY = 0.4
NOSE_TIP, CHIN = 1, 152


@dataclass(frozen=True)
class Sample:
    rgb: Image
    eyes_closed: bool


@dataclass(frozen=True)
class Condition:
    key: str
    name: str
    meaning: str
    transform: Callable[[Image, np.random.Generator, dict[str, Any]], Image]
    fix: Callable[[dict[str, Any]], Preprocess] | None = None  # built per frame: a crop depends on where the face is
    fix_note: str = ""


@dataclass(frozen=True)
class Measurement:
    detection: float
    eye_visibility: float
    mouth_visibility: float
    ear_open: float
    ear_closed: float
    closure_separation: float
    yaw_spread_degrees: float
    verdict: str
    reasons: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ConditionResult:
    key: str
    name: str
    meaning: str
    as_captured: Measurement
    with_fix: Measurement | None
    fix_note: str


# ---- transforms ---------------------------------------------------------------------------------------
def _noise(image: np.ndarray, rng: np.random.Generator, sigma: float) -> Image:
    noisy = image.astype(np.float32) + rng.normal(0.0, sigma, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _scale_light(image: Image, factor: float) -> np.ndarray:
    return image.astype(np.float32) * factor


def _eye_boxes(context: dict[str, Any]) -> list[tuple[int, int, int, int]]:
    points = context.get("points")
    if points is None:
        return []
    boxes = []
    for ring in (LEFT_EYE_RING, RIGHT_EYE_RING):
        xs, ys = points[list(ring), 0], points[list(ring), 1]
        pad_x = (xs.max() - xs.min()) * EYE_PATCH_PAD
        pad_y = (xs.max() - xs.min()) * EYE_PATCH_PAD * 0.8
        boxes.append(
            (
                int(xs.min() - pad_x),
                int(ys.min() - pad_y),
                int(xs.max() + pad_x),
                int(ys.max() + pad_y),
            )
        )
    return boxes


def _sunglasses(image: Image, rng: np.random.Generator, context: dict[str, Any]) -> Image:
    import cv2

    out = image.copy()
    for x0, y0, x1, y1 in _eye_boxes(context):
        cv2.ellipse(
            out,
            ((x0 + x1) // 2, (y0 + y1) // 2),
            ((x1 - x0) // 2, (y1 - y0) // 2),
            0,
            0,
            360,
            (18, 18, 22),
            -1,
        )
    return _noise(out, rng, 2.0)


def _glare(image: Image, rng: np.random.Generator, context: dict[str, Any]) -> Image:
    import cv2

    overlay = image.copy()
    for x0, y0, x1, y1 in _eye_boxes(context):
        cx, cy = (x0 + x1) // 2 + (x1 - x0) // 6, (y0 + y1) // 2 - (y1 - y0) // 6
        cv2.ellipse(
            overlay,
            (cx, cy),
            ((x1 - x0) // 3, (y1 - y0) // 3),
            25,
            0,
            360,
            (255, 255, 255),
            -1,
        )
    return np.asarray(cv2.addWeighted(overlay, 0.75, image, 0.25, 0), dtype=np.uint8)


def _place_in_wide_frame(image: Image, scale: float) -> tuple[Image, tuple[float, float, float, float]]:
    """The face as a cabin camera at the mirror sees it: small, on the right of a wide frame (US driver seat)."""
    import cv2

    width, height = WIDE_FRAME
    canvas = np.full((height, width, 3), 38, dtype=np.uint8)
    target_h = int(height * scale * 2.2)
    target_w = int(image.shape[1] * target_h / image.shape[0])
    small = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    left, top = int(width * 0.66) - target_w // 2, int(height * 0.42) - target_h // 2
    canvas[top : top + target_h, left : left + target_w] = small
    margin = 0.08  # the seat region is set once per car, close around where the head sits
    region = (
        max(0.0, (left - target_w * margin) / width),
        max(0.0, (top - target_h * margin) / height),
        min(1.0, (left + target_w * (1 + margin)) / width),
        min(1.0, (top + target_h * (1 + margin)) / height),
    )
    return canvas, region


def _tiny(image: Image, rng: np.random.Generator, context: dict[str, Any]) -> Image:
    canvas, context["region"] = _place_in_wide_frame(image, TINY_FACE_SCALE)
    return _noise(canvas, rng, 3.0)


def _fisheye(image: Image, rng: np.random.Generator, context: dict[str, Any]) -> Image:
    canvas, context["region"] = _place_in_wide_frame(image, FISHEYE_FACE_SCALE)
    return _noise(distort_fisheye(canvas, FISHEYE_FOV), rng, 3.0)


def _side_angle(image: Image, rng: np.random.Generator, context: dict[str, Any]) -> Image:
    """A planar approximation of seeing the face from the mirror, about 30-40 degrees off-axis."""
    import cv2

    height, width = image.shape[:2]
    source = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)
    squeeze = width * (1 - SIDE_SQUEEZE)
    target = np.array(
        [
            [squeeze, height * 0.08],
            [width, 0],
            [width, height],
            [squeeze, height * 0.92],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(source, target)
    warped = cv2.warpPerspective(image, matrix, (width, height), borderValue=(38, 38, 38))
    return _noise(np.asarray(warped, dtype=np.uint8), rng, 2.0)


def _ir_night(image: Image, rng: np.random.Generator, context: dict[str, Any]) -> Image:
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) * 0.55 + 20
    soft = np.asarray(cv2.GaussianBlur(gray, (3, 3), 0), dtype=np.float32)
    return _noise(np.stack([soft] * 3, axis=-1), rng, 6.0)


def _backlit(image: Image, rng: np.random.Generator, context: dict[str, Any]) -> Image:
    width = image.shape[1]
    out = _scale_light(image, 0.3)
    ramp = np.clip((np.abs(np.linspace(-1, 1, width)) - 0.45) / 0.55, 0, 1)[None, :, None]
    out = out * (1 - ramp) + 255 * ramp
    return _noise(out, rng, 4.0)


def _side_sun(image: Image, rng: np.random.Generator, context: dict[str, Any]) -> Image:
    width = image.shape[1]
    gain = np.linspace(2.0, 0.5, width)[None, :, None]
    return _noise(_scale_light(image, 1.0) * gain, rng, 3.0)


def _motion_blur(image: Image, rng: np.random.Generator, context: dict[str, Any]) -> Image:
    import cv2

    kernel = np.zeros((15, 15), np.float32)
    kernel[7, :] = 1.0 / 15
    return np.asarray(cv2.filter2D(image, -1, kernel), dtype=np.uint8)


def _low_resolution(image: Image, rng: np.random.Generator, context: dict[str, Any]) -> Image:
    import cv2

    height, width = image.shape[:2]
    small = cv2.resize(image, (max(1, width // 5), max(1, height // 5)), interpolation=cv2.INTER_AREA)
    return np.asarray(
        cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR),
        dtype=np.uint8,
    )


def _jpeg(image: Image, rng: np.random.Generator, context: dict[str, Any]) -> Image:
    import cv2

    ok, encoded = cv2.imencode(
        ".jpg",
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), 8],
    )
    if not ok:
        return image.copy()
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        return image.copy()
    return np.asarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), dtype=np.uint8)


def _hand(image: Image, rng: np.random.Generator, context: dict[str, Any]) -> Image:
    """A hand (or mask) over the face from just below the nose to under the chin."""
    out = image.copy()
    points = context.get("points")
    if points is None:
        return out
    nose_y, chin_y = points[NOSE_TIP, 1], points[CHIN, 1]
    left, right = points[:, 0].min(), points[:, 0].max()
    pad = (right - left) * 0.12
    top = int(nose_y + (chin_y - nose_y) * 0.15)
    out[top : int(chin_y + pad), max(0, int(left - pad)) : int(right + pad)] = (
        178,
        132,
        110,
    )
    return _noise(out, rng, 2.0)


def _light_fix(context: dict[str, Any]) -> Preprocess:
    return Preprocess(enhance_low_light=True)


def _seat_fix(context: dict[str, Any]) -> Preprocess:
    return Preprocess(region=context["region"])


def _fisheye_fix(context: dict[str, Any]) -> Preprocess:
    return Preprocess(fisheye_fov_degrees=FISHEYE_FOV, region=context["region"])


CONDITIONS: tuple[Condition, ...] = (
    Condition(
        "clean",
        "Clean daylight, phone in front",
        "the reference every row is compared with",
        lambda image, rng, context: image.copy(),
    ),
    Condition(
        "dim",
        "Dim cabin (dusk, overcast)",
        "brightness at 35% with sensor noise",
        lambda image, rng, context: _noise(_scale_light(image, 0.35), rng, 6.0),
        _light_fix,
        "CLAHE low-light enhancement",
    ),
    Condition(
        "night",
        "Night, no infrared light",
        "brightness at 8%, heavy noise: a phone camera at night",
        lambda image, rng, context: _noise(_scale_light(image, 0.08), rng, 7.0),
        _light_fix,
        "CLAHE low-light enhancement",
    ),
    Condition(
        "ir",
        "Night with an infrared cabin camera",
        "monochrome, soft, noisy: what an IR dashcam records",
        _ir_night,
        _light_fix,
        "CLAHE low-light enhancement",
    ),
    Condition(
        "backlit",
        "Backlit (bright windows behind)",
        "face at 30% with blown-out window edges",
        _backlit,
        _light_fix,
        "CLAHE low-light enhancement",
    ),
    Condition(
        "side_sun",
        "Hard sun on one side of the face",
        "one side overexposed, the other in shadow",
        _side_sun,
    ),
    Condition(
        "blur",
        "Motion blur (bumps, vibration)",
        "15-pixel horizontal blur",
        _motion_blur,
    ),
    Condition(
        "low_res",
        "Low resolution",
        "one fifth of the resolution, scaled back up",
        _low_resolution,
    ),
    Condition(
        "jpeg",
        "Heavy stream compression",
        "JPEG quality 8: a weak Wi-Fi dashcam stream",
        _jpeg,
    ),
    Condition(
        "tiny",
        "Cabin camera at the mirror: small face",
        "the face fills about a fifth of a 1280x720 frame",
        _tiny,
        _seat_fix,
        "crop to the driver's seat region",
    ),
    Condition(
        "fisheye",
        "210-degree fisheye cabin camera",
        "VIOFO RNF210-style lens, face off-centre",
        _fisheye,
        _fisheye_fix,
        "fisheye correction, then the driver's seat region",
    ),
    Condition(
        "side_angle",
        "Seen from the side (mirror mount)",
        "planar approximation of a 30-40 degree view",
        _side_angle,
    ),
    Condition("sunglasses", "Dark sunglasses", "both eyes covered by dark lenses", _sunglasses),
    Condition("glare", "Glare on clear glasses", "bright reflections over both lenses", _glare),
    Condition(
        "hand",
        "Hand or mask over the lower face",
        "covered from below the nose to the chin",
        _hand,
    ),
)


# ---- measurement --------------------------------------------------------------------------------------
class ImageLandmarker:
    """MediaPipe Face Landmarker in IMAGE mode: every call is independent, so frames can come in any order."""

    def __init__(self, models_dir: Path) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        from ..features.extraction import download_face_model

        self._mp = mp
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(download_face_model(models_dir))),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)

    def close(self) -> None:
        self._detector.close()

    def detect(self, rgb: Image, spec: FeatureSpec = FEATURE_SPEC) -> tuple[dict[str, float], np.ndarray] | None:
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._detector.detect(image)
        if not result.face_landmarks:
            return None
        height, width = rgb.shape[:2]
        points = np.array([[p.x * width, p.y * height] for p in result.face_landmarks[0]])
        blendshapes = (
            {c.category_name: c.score for c in result.face_blendshapes[0]} if result.face_blendshapes else None
        )
        matrix = result.facial_transformation_matrixes[0] if result.facial_transformation_matrixes else None
        return frame_features(points, blendshapes, matrix, spec), points


def _region_contrast(rgb: Image, points: np.ndarray, rings: Sequence[Sequence[int]], grow: int) -> float:
    """Luminance spread inside the landmark outlines (slightly grown): what an eye or mouth change shows up in."""
    import cv2

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = np.zeros(gray.shape, np.uint8)
    for ring in rings:
        polygon = np.round(points[list(ring)]).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 255)
    if grow:
        mask = np.asarray(cv2.dilate(mask, np.ones((grow, grow), np.uint8)), dtype=np.uint8)
    values = gray[mask > 0]
    return float(values.std()) if values.size >= 12 else 0.0


def _eye_contrast(rgb: Image, points: np.ndarray) -> float:
    return _region_contrast(rgb, points, (EYE_OUTLINE_LEFT, EYE_OUTLINE_RIGHT), grow=5)


def _mouth_contrast(rgb: Image, points: np.ndarray) -> float:
    return _region_contrast(rgb, points, (MOUTH_OUTLINE,), grow=9)


@dataclass
class _Tally:
    found: int = 0
    total: int = 0
    contrast: list[float] = field(default_factory=list)
    mouth: list[float] = field(default_factory=list)
    ear_open: list[float] = field(default_factory=list)
    ear_closed: list[float] = field(default_factory=list)
    yaw: list[float] = field(default_factory=list)


def _measure(tally: _Tally, clean: tuple[float, float], closed_supplied: bool) -> Measurement:
    clean_contrast, clean_mouth = clean
    detection = tally.found / tally.total if tally.total else 0.0
    visibility = (float(np.median(tally.contrast)) / clean_contrast) if tally.contrast and clean_contrast else 0.0
    mouth = (float(np.median(tally.mouth)) / clean_mouth) if tally.mouth and clean_mouth else 0.0
    ear_open = float(np.median(tally.ear_open)) if tally.ear_open else math.nan
    ear_closed = float(np.median(tally.ear_closed)) if tally.ear_closed else math.nan
    if visibility < FAIL_EYE_VISIBILITY:
        ear_open = math.nan
        ear_closed = math.nan
    separation = (
        1.0 - ear_closed / ear_open
        if ear_open and math.isfinite(ear_open) and tally.ear_closed and math.isfinite(ear_closed)
        else math.nan
    )
    yaws = [value for value in tally.yaw if math.isfinite(value)]
    yaw_spread = float(np.std(yaws)) if len(yaws) > 1 else math.nan
    reasons: list[str] = []
    verdict = WORKS
    if detection < FAIL_DETECTION:
        verdict, reasons = FAILS, [f"face found in {detection:.0%} of frames: windows go unscored below 60%"]
    else:
        if detection < MIN_DETECTION:
            verdict = DEGRADED
            reasons.append(f"face found in {detection:.0%} of frames")
        if visibility < FAIL_EYE_VISIBILITY:
            verdict = FAILS
            reasons.append(f"eyes not visible ({visibility:.0%} of clean contrast): closures cannot be seen")
        elif visibility < MIN_EYE_VISIBILITY:
            verdict = DEGRADED if verdict == WORKS else verdict
            reasons.append(f"eye contrast {visibility:.0%} of clean")
        if mouth < MIN_MOUTH_VISIBILITY:
            verdict = DEGRADED if verdict == WORKS else verdict
            reasons.append(f"mouth not visible ({mouth:.0%} of clean contrast): yawns cannot be seen")
        if closed_supplied and math.isfinite(separation):
            if separation < FAIL_SEPARATION:
                verdict = FAILS
                reasons.append(f"closed eyes only {separation:.0%} below open: blinks would not count")
            elif separation < MIN_SEPARATION:
                verdict = DEGRADED if verdict == WORKS else verdict
                reasons.append(f"closure separation {separation:.0%}")
        if math.isfinite(yaw_spread) and yaw_spread > MAX_YAW_SPREAD_DEGREES:
            verdict = DEGRADED if verdict == WORKS else verdict
            reasons.append(f"head angle jitters by {yaw_spread:.1f} degrees")
    return Measurement(
        detection,
        visibility,
        mouth,
        ear_open,
        ear_closed,
        separation,
        yaw_spread,
        verdict,
        reasons,
    )


def assess(
    samples: Sequence[Sample],
    landmarker: ImageLandmarker,
    conditions: Sequence[Condition] = CONDITIONS,
    seed: int = 7,
    progress: Callable[[str], None] | None = None,
) -> list[ConditionResult]:
    """Run every condition over every sample, as captured and with its fix."""
    bases: list[tuple[Sample, dict[str, Any]]] = []
    eye_clean, mouth_clean = [], []
    for sample in samples:
        found = landmarker.detect(sample.rgb)
        if found is None:
            continue
        points = found[1]
        bases.append((sample, {"points": points}))
        if not sample.eyes_closed:
            eye_clean.append(_eye_contrast(sample.rgb, points))
            mouth_clean.append(_mouth_contrast(sample.rgb, points))
    if not bases:
        raise ValueError("no face found in the clean frames: face the camera in normal light and try again")
    clean = (
        float(np.median(eye_clean)) if eye_clean else 1.0,
        float(np.median(mouth_clean)) if mouth_clean else 1.0,
    )
    closed_supplied = any(sample.eyes_closed for sample, _ in bases)
    results = []
    for condition in conditions:
        if progress:
            progress(condition.name)
        rng = np.random.default_rng(seed)
        tallies = {"raw": _Tally(), "fixed": _Tally()}
        for sample, context in bases:
            transformed = condition.transform(sample.rgb, rng, context)
            variants = {"raw": transformed}
            if condition.fix is not None:
                variants["fixed"] = condition.fix(context).apply(transformed)
            for name, image in variants.items():
                tally = tallies[name]
                tally.total += 1
                found = landmarker.detect(image)
                if found is None:
                    continue
                features, points = found
                tally.found += 1
                (tally.ear_closed if sample.eyes_closed else tally.ear_open).append(features["ear"])
                if not sample.eyes_closed:
                    tally.contrast.append(_eye_contrast(image, points))
                    tally.mouth.append(_mouth_contrast(image, points))
                    tally.yaw.append(features["yaw"])
        results.append(
            ConditionResult(
                key=condition.key,
                name=condition.name,
                meaning=condition.meaning,
                as_captured=_measure(tallies["raw"], clean, closed_supplied),
                with_fix=_measure(tallies["fixed"], clean, closed_supplied) if condition.fix else None,
                fix_note=condition.fix_note,
            )
        )
    return results


def results_table(results: Sequence[ConditionResult]) -> str:
    """A plain-text table for the terminal."""
    lines = [f"{'Condition':44} {'Face':>6} {'Eyes':>6} {'Mouth':>6} {'Sep.':>6} {'Verdict':>9}   With fix"]
    for result in results:
        raw = result.as_captured
        separation = "–" if not math.isfinite(raw.closure_separation) else f"{raw.closure_separation:.0%}"
        fixed = (
            "no fix available"
            if result.with_fix is None
            else (
                f"{result.with_fix.verdict} ({result.with_fix.detection:.0%} face, "
                f"{result.with_fix.eye_visibility:.0%} eyes; {result.fix_note})"
            )
        )
        lines.append(
            f"{result.name:44} {raw.detection:>6.0%} {raw.eye_visibility:>6.0%} {raw.mouth_visibility:>6.0%} "
            f"{separation:>6} {raw.verdict:>9}   {fixed}"
        )
        for reason in raw.reasons:
            lines.append(f"{'':46}- {reason}")
    return "\n".join(lines)


# ---- capture ------------------------------------------------------------------------------------------
def capture_from_camera(source: str, seconds_open: float, seconds_closed: float, every: int = 3) -> list[Sample]:
    """Prompted capture: eyes open and looking ahead, then eyes closed. Frames stay in memory."""
    import cv2

    capture = cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)
    if not capture.isOpened():
        raise OSError(f"cannot open camera {source}")
    samples: list[Sample] = []
    try:
        for closed, seconds, prompt in (
            (False, seconds_open, "Look at the camera with your eyes OPEN"),
            (True, seconds_closed, "Now CLOSE your eyes and keep your head still"),
        ):
            if seconds <= 0:
                continue
            print(f"\n{prompt} in 3 seconds...", flush=True)
            time.sleep(3)
            print("recording", flush=True)
            started, index = time.monotonic(), 0
            while time.monotonic() - started < seconds:
                ok, bgr = capture.read()
                if not ok:
                    raise OSError("the camera stopped sending frames")
                index += 1
                if index % every == 0:
                    samples.append(
                        Sample(
                            np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), np.uint8),
                            closed,
                        )
                    )
    finally:
        capture.release()
    return samples


def load_images(paths: Sequence[str], eyes_closed: bool) -> list[Sample]:
    import cv2

    samples = []
    for path in paths:
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise OSError(f"cannot read image {path}")
        samples.append(Sample(np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), np.uint8), eyes_closed))
    return samples


def load_video(path: str, every_seconds: float, eyes_closed: bool, limit: int = 60) -> list[Sample]:
    import cv2

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise OSError(f"cannot open video {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps * every_seconds))
    samples: list[Sample] = []
    index = 0
    try:
        while len(samples) < limit:
            ok, bgr = capture.read()
            if not ok:
                break
            if index % step == 0:
                samples.append(
                    Sample(
                        np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), np.uint8),
                        eyes_closed,
                    )
                )
            index += 1
    finally:
        capture.release()
    return samples


# ---- the same conditions as frame-signal effects, for the scripted regression suite --------------------------
@dataclass(frozen=True)
class FeatureCondition:
    """A condition's effect on the indicators' inputs. ``eyes_visible`` is whether eye readings should survive it."""

    key: str
    name: str
    situation: Situation
    eyes_visible: bool
    camera: CameraHealth
    eye_visibility: float
    mouth_visibility: float = 1.0
    ear_noise: float = 0.0


_CLEAR = CameraHealth(mean_luma=0.5, face_luma=0.5, sharpness=100.0)
FEATURE_CONDITIONS: tuple[FeatureCondition, ...] = (
    FeatureCondition(
        "sunglasses",
        "Dark sunglasses",
        Situation.SUNGLASSES_OR_EYE_OCCLUSION,
        eyes_visible=False,
        camera=_CLEAR,
        eye_visibility=0.1,
    ),
    FeatureCondition(
        "ir",
        "Infrared cabin camera",
        Situation.IR_MODE,
        eyes_visible=True,
        camera=CameraHealth(mean_luma=0.35, face_luma=0.4, sharpness=40.0, grayscale=True),
        eye_visibility=0.8,
        ear_noise=0.008,
    ),
    FeatureCondition(
        "low_light",
        "Dim cabin, no infrared",
        Situation.LOW_LIGHT,
        eyes_visible=True,
        camera=CameraHealth(mean_luma=0.08, face_luma=0.09, sharpness=25.0),
        eye_visibility=0.55,
        ear_noise=0.012,
    ),
    FeatureCondition(
        "backlit",
        "Backlit by bright windows",
        Situation.BACKLIT,
        eyes_visible=True,
        camera=CameraHealth(mean_luma=0.8, face_luma=0.3, sharpness=60.0),
        eye_visibility=0.5,
        ear_noise=0.01,
    ),
    FeatureCondition(
        "glare",
        "Glare on clear glasses",
        Situation.CLEAR_GLASSES_GLARE,
        eyes_visible=False,
        camera=CameraHealth(mean_luma=0.55, face_luma=0.55, sharpness=90.0, glare_fraction=0.15),
        eye_visibility=0.15,
    ),
    FeatureCondition(
        "blur",
        "Motion blur",
        Situation.IMAGE_BLUR,
        eyes_visible=True,
        camera=CameraHealth(mean_luma=0.5, face_luma=0.5, sharpness=5.0),
        eye_visibility=0.6,
        ear_noise=0.015,
    ),
    FeatureCondition(
        "fisheye",
        "210-degree fisheye lens",
        Situation.FISHEYE_DISTORTION,
        eyes_visible=True,
        camera=replace(_CLEAR, fisheye=True),
        eye_visibility=0.6,
        ear_noise=0.01,
    ),
    FeatureCondition(
        "hand",
        "Hand over the lower face",
        Situation.LOWER_FACE_OCCLUDED,
        eyes_visible=True,
        camera=_CLEAR,
        eye_visibility=1.0,
        mouth_visibility=0.05,
    ),
)
FEATURE_CONDITION_KEYS: dict[str, FeatureCondition] = {condition.key: condition for condition in FEATURE_CONDITIONS}


def apply_feature_condition(frames: Any, condition: FeatureCondition, seed: int = 0) -> Any:
    """A copy of a frame table (``features.synthetic.scripted_synthetic_session``) under ``condition``.

    Where the eyes are hidden, EAR reads a plausible open value, because MediaPipe still places eyelids it cannot
    see; the indicators must reject it on visibility, not on the value.
    """
    rng = np.random.default_rng(seed)
    out = frames.copy()
    out["eye_visibility"] = np.minimum(out["eye_visibility"], condition.eye_visibility)
    out["mouth_visibility"] = np.minimum(out["mouth_visibility"], condition.mouth_visibility)
    eye_columns = ["ear", "ear_left", "ear_right"]
    if not condition.eyes_visible:
        out[eye_columns] = float(np.nanmedian(frames["ear"].to_numpy(dtype=float)))
    elif condition.ear_noise > 0:
        for column in eye_columns:
            out[column] = out[column] + rng.normal(0.0, condition.ear_noise, len(out))
    return out


def to_json(results: Sequence[ConditionResult]) -> list[dict[str, Any]]:
    return [asdict(result) for result in results]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m driven_ml.studio.conditions",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        default="",
        help="camera index or stream URL for a prompted capture (e.g. 0)",
    )
    parser.add_argument("--open-seconds", type=float, default=5.0)
    parser.add_argument(
        "--closed-seconds",
        type=float,
        default=4.0,
        help="0 to skip the eyes-closed capture",
    )
    parser.add_argument("--image", nargs="*", default=[], help="photos with eyes open")
    parser.add_argument(
        "--closed-image",
        nargs="*",
        default=[],
        help="photos of the same people with eyes closed",
    )
    parser.add_argument("--video", default="", help="a video with eyes open, sampled every second")
    parser.add_argument("--models-dir", default="driven_work/studio/mediapipe")
    parser.add_argument("--json", default="", help="also write the results to this JSON file")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    samples: list[Sample] = []
    if args.source:
        samples += capture_from_camera(args.source, args.open_seconds, args.closed_seconds)
    samples += load_images(args.image, eyes_closed=False) + load_images(args.closed_image, eyes_closed=True)
    if args.video:
        samples += load_video(args.video, every_seconds=1.0, eyes_closed=False)
    if not samples:
        parser.error("give --source, --image or --video")
    landmarker = ImageLandmarker(Path(args.models_dir))
    try:
        results = assess(samples, landmarker, progress=lambda name: logger.info("testing: %s", name))
    finally:
        landmarker.close()
    print(results_table(results))
    if args.json:
        Path(args.json).write_text(json.dumps(to_json(results), indent=1, allow_nan=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
