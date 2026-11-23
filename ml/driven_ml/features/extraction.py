"""Run MediaPipe Face Landmarker over a video and turn each sampled frame into frame features.

MediaPipe is the pre-trained network (the "transfer learning" part): Driven never trains it. Videos are
sampled down to the spec's frame rate by timestamp, whatever the source rate, so training sees what the phone
sees. ``mediapipe`` and ``cv2`` are
imported inside the functions that need them, so the rest of the pipeline runs without them.
"""

from __future__ import annotations

import logging
import math
import os
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from ..errors import DataError
from ..spec import CLIP_DRIVE, FEATURE_SPEC, FRAME_FEATURES, FeatureSpec
from .geometry import frame_features, no_face_frame
from .preprocess import Preprocess

logger = logging.getLogger(__name__)

FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
FACE_MODEL_FILE = "face_landmarker.task"
_DEFAULT_SOURCE_FPS = 30.0


class FaceDetector(Protocol):
    def detect_for_video(self, image: Any, timestamp_ms: int) -> Any: ...

    def close(self) -> None: ...


@dataclass
class LandmarkerSession:
    detector: FaceDetector
    to_image: Callable[[np.ndarray], Any]


def download_face_model(directory: Path) -> Path:
    """Download the Face Landmarker model once and reuse it after that."""
    path = directory / FACE_MODEL_FILE
    if path.is_file():
        return path
    directory.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    logger.info("downloading %s", FACE_LANDMARKER_URL)
    try:
        urllib.request.urlretrieve(FACE_LANDMARKER_URL, temp_path)
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return path


@contextmanager
def open_face_landmarker(model_dir: Path) -> Iterator[LandmarkerSession]:
    """A VIDEO-mode Face Landmarker for one face, with blendshapes and transformation matrices."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(download_face_model(model_dir))),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    detector = vision.FaceLandmarker.create_from_options(options)
    try:
        yield LandmarkerSession(
            detector=detector,
            to_image=lambda rgb: mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
        )
    finally:
        detector.close()


def _features_from_result(result: Any, width: int, height: int, spec: FeatureSpec) -> dict[str, float]:
    if not result.face_landmarks:
        return no_face_frame()
    points = np.array([[landmark.x * width, landmark.y * height] for landmark in result.face_landmarks[0]])
    blendshapes = (
        {category.category_name: category.score for category in result.face_blendshapes[0]}
        if result.face_blendshapes
        else None
    )
    matrices = result.facial_transformation_matrixes
    matrix = matrices[0] if matrices else None
    return frame_features(points, blendshapes, matrix, spec)


def extract_video_features(
    path: Path,
    *,
    subject_id: str,
    session_id: str,
    label: int,
    session: LandmarkerSession,
    clip: str = CLIP_DRIVE,
    spec: FeatureSpec = FEATURE_SPEC,
    preprocess: Preprocess | None = None,
) -> pd.DataFrame:
    """One row per sampled frame: subject, session, clip, label, t in seconds, then the 13 frame features.

    ``preprocess`` applies the OpenCV preparation in ``features.preprocess`` (low-light enhancement, driver
    crop, fisheye correction) before MediaPipe. Use the same setting the deployed camera uses: training on
    prepared frames and deploying on raw ones (or the reverse) shifts the features.
    """
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"cannot open video {path}")
    rows = []
    try:
        source_fps = capture.get(cv2.CAP_PROP_FPS) or _DEFAULT_SOURCE_FPS
        if source_fps < spec.target_fps:
            raise DataError(f"{path} runs at {source_fps:g} fps; at least {spec.target_fps} fps is needed")
        index = 0
        last_slot = -1
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            # Keep the first frame of each 1/target_fps slot, so any source rate (25, 29.97, 30, 60 fps) gives
            # exactly target_fps samples per second; for 30 and 60 fps this is every 3rd or 6th frame.
            slot = math.floor(index * spec.target_fps / source_fps + 1e-9)
            if slot > last_slot:
                last_slot = slot
                timestamp_ms = int(index / source_fps * 1000)
                rgb = np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=np.uint8)
                if preprocess is not None and preprocess.active:
                    rgb = preprocess.apply(rgb)
                result = session.detector.detect_for_video(session.to_image(rgb), timestamp_ms)
                height, width = rgb.shape[:2]
                rows.append(
                    {
                        "subject": subject_id,
                        "session": session_id,
                        "clip": clip,
                        "label": label,
                        "t": timestamp_ms / 1000,
                        **_features_from_result(result, width, height, spec),
                    }
                )
            index += 1
    finally:
        capture.release()
    return pd.DataFrame(rows, columns=["subject", "session", "clip", "label", "t", *FRAME_FEATURES])
