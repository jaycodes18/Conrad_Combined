"""Per-frame signals from MediaPipe face landmarks: eye and mouth openness, head pose and gaze.

These are pure numpy functions with no I/O. The on-device port is checked against them to 1e-6 with the
frame fixtures, so every formula here is the contract.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ..spec import FEATURE_SPEC, FRAME_FEATURES, FeatureSpec

_EPSILON = 1e-9

Points = npt.NDArray[np.float64]


@dataclass(frozen=True)
class HeadPose:
    yaw_degrees: float
    pitch_degrees: float
    roll_degrees: float


def _distance(points: Points, a: int, b: int) -> float:
    return float(np.hypot(points[a, 0] - points[b, 0], points[a, 1] - points[b, 1]))


def eye_aspect_ratio(points: Points, indices: Sequence[int]) -> float:
    """Soukupová and Čech (2016): vertical eyelid gaps over the eye's width. Falls toward 0 as the eye closes."""
    p1, p2, p3, p4, p5, p6 = indices
    return (_distance(points, p2, p6) + _distance(points, p3, p5)) / (2 * _distance(points, p1, p4) + _EPSILON)


def mouth_aspect_ratio(points: Points, indices: Sequence[int]) -> float:
    left, right, upper, lower = indices
    return _distance(points, upper, lower) / (_distance(points, left, right) + _EPSILON)


def head_pose(matrix: npt.ArrayLike) -> HeadPose:
    """Yaw, pitch and roll in degrees from the rotation part of MediaPipe's 4x4 facial transformation matrix."""
    rotation = np.asarray(matrix, dtype=float).reshape(4, 4)[:3, :3]
    yaw = math.asin(min(1.0, max(-1.0, -rotation[2, 0])))
    pitch = math.atan2(rotation[2, 1], rotation[2, 2])
    roll = math.atan2(rotation[1, 0], rotation[0, 0])
    return HeadPose(math.degrees(yaw), math.degrees(pitch), math.degrees(roll))


def _iris_position(points: Points, iris: int, corners: tuple[int, int]) -> float:
    """Where the iris sits between the eye's corners: 0 at the first corner, 1 at the second."""
    first, second = corners
    span = points[second, 0] - points[first, 0]
    return float((points[iris, 0] - points[first, 0]) / (span or _EPSILON))


def horizontal_gaze(points: Points, spec: FeatureSpec = FEATURE_SPEC) -> float:
    """Mean iris position of both eyes, centred on 0; both corner pairs run left to right in the image."""
    landmarks = spec.landmarks
    right = _iris_position(points, landmarks.right_iris_center, landmarks.right_eye_corners)
    left = _iris_position(points, landmarks.left_iris_center, landmarks.left_eye_corners)
    return (right + left) / 2 - 0.5


def _blendshape(blendshapes: Mapping[str, float] | None, name: str) -> float:
    if not blendshapes or name not in blendshapes:
        return math.nan
    return float(blendshapes[name])


def _look_down(blendshapes: Mapping[str, float] | None) -> float:
    if not blendshapes:
        return math.nan
    values = [float(blendshapes[name]) for name in ("eyeLookDownLeft", "eyeLookDownRight") if name in blendshapes]
    return sum(values) / len(values) if values else math.nan


def frame_features(
    points_px: npt.ArrayLike,
    blendshapes: Mapping[str, float] | None = None,
    transform_matrix: npt.ArrayLike | None = None,
    spec: FeatureSpec = FEATURE_SPEC,
) -> dict[str, float]:
    """The 13 ``FRAME_FEATURES`` for one frame with a face, from landmarks in pixels."""
    points = np.asarray(points_px, dtype=float)
    landmarks = spec.landmarks
    ear_left = eye_aspect_ratio(points, landmarks.left_eye)
    ear_right = eye_aspect_ratio(points, landmarks.right_eye)
    pose = head_pose(transform_matrix) if transform_matrix is not None else HeadPose(math.nan, math.nan, math.nan)
    features = {
        "ear_left": ear_left,
        "ear_right": ear_right,
        "ear": (ear_left + ear_right) / 2,
        "mar": mouth_aspect_ratio(points, landmarks.mouth),
        "yaw": pose.yaw_degrees,
        "pitch": pose.pitch_degrees,
        "roll": pose.roll_degrees,
        "gaze_horizontal": horizontal_gaze(points, spec),
        "blink_left": _blendshape(blendshapes, "eyeBlinkLeft"),
        "blink_right": _blendshape(blendshapes, "eyeBlinkRight"),
        "jaw_open": _blendshape(blendshapes, "jawOpen"),
        "look_down": _look_down(blendshapes),
        "face": 1.0,
    }
    return {name: features[name] for name in FRAME_FEATURES}


def no_face_frame() -> dict[str, float]:
    """A frame where no face was found: ``face`` is 0 and every signal is NaN."""
    return {name: 0.0 if name == "face" else math.nan for name in FRAME_FEATURES}
