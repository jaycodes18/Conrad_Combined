"""Where live frames come from: a webcam, a dashcam's RTSP or HTTP stream, a video file, or synthetic data.

Every source yields `SourceFrame`s sampled at the feature spec's target fps (10 fps). Cameras and
streams are sampled by wall-clock time; files by their own timestamps, played back in real time
(or faster) so the dashboard looks exactly like a live drive.

VIOFO and other Novatek dashcams: join the camera's Wi-Fi (or put it in Wi-Fi Station Mode on the same
network) and use `rtsp://192.168.1.254`. Some models only serve HTTP live view, at
`http://192.168.1.254:8192`. The camera streams one channel at a time, so select the cabin camera
in the VIOFO app first. `start_novatek_live_view` sends the documented live-view-start command (2015).
"""

from __future__ import annotations

import os
import time
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ..spec import FEATURE_SPEC, FeatureSpec

SYNTHETIC = "synthetic"
NOVATEK_DEFAULT_IP = "192.168.1.254"
NOVATEK_LIVE_VIEW_START = 2015


@dataclass(frozen=True)
class SourceFrame:
    t: float  # seconds since the source started
    rgb: NDArray[np.uint8] | None  # None for synthetic frames
    features: dict[str, float] | None  # set only by the synthetic source


def start_novatek_live_view(ip: str = NOVATEK_DEFAULT_IP, timeout_seconds: float = 3.0) -> str:
    """Ask a Novatek-based dashcam (VIOFO and others) to start live view. Returns the camera's XML reply."""
    url = f"http://{ip}/?custom=1&cmd={NOVATEK_LIVE_VIEW_START}&par=1"
    with urllib.request.urlopen(url, timeout=timeout_seconds) as response:
        return str(response.read().decode("utf-8", "replace"))


def describe(source: str) -> str:
    if source == SYNTHETIC:
        return "synthetic driver (no camera): the pipeline's scripted drowsiness session"
    if source.isdigit():
        return f"webcam {source}"
    if source.startswith("rtsp://"):
        return f"RTSP stream {source}"
    if source.startswith(("http://", "https://")):
        return f"HTTP stream {source}"
    return f"video file {source}"


def iter_camera(source: str, spec: FeatureSpec = FEATURE_SPEC, speed: float = 1.0) -> Iterator[SourceFrame]:
    """Frames from a webcam index, an RTSP/HTTP URL or a video file, sampled at `spec.target_fps`."""
    import cv2

    if source.startswith("rtsp://"):
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    is_file = not source.isdigit() and "://" not in source
    capture = _open_capture(source, cv2, is_file)
    step = 1.0 / spec.target_fps
    started = time.monotonic()
    next_sample = 0.0
    try:
        while True:
            ok, bgr = capture.read()
            if not ok:
                if is_file:
                    return
                raise OSError(f"{describe(source)} stopped sending frames")
            if is_file:
                t = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                if t + 1e-9 < next_sample:
                    continue
                wait = t / max(speed, 0.01) - (time.monotonic() - started)
                if wait > 0:
                    time.sleep(wait)
            else:
                t = time.monotonic() - started
                if t + 1e-9 < next_sample:
                    continue
            next_sample += step
            while next_sample <= t:
                next_sample += step
            rgb: NDArray[np.uint8] = np.asarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=np.uint8)
            yield SourceFrame(t=t, rgb=rgb, features=None)
    finally:
        capture.release()


def _open_capture(source: str, cv2: Any, is_file: bool) -> Any:
    capture = cv2.VideoCapture(int(source)) if source.isdigit() else cv2.VideoCapture(source)
    if not capture.isOpened():
        raise OSError(f"cannot open {describe(source)}")
    if not is_file:
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def iter_synthetic(seed: int = 20, speed: float = 4.0, spec: FeatureSpec = FEATURE_SPEC) -> Iterator[SourceFrame]:
    """Replay the scripted indicator session (calibration, then closure, gaze, nods, head drops, sunglasses)."""
    frames = _scripted_frames(seed)
    started = time.monotonic()
    for record in frames:
        t = float(record["t"])
        wait = t / max(speed, 0.01) - (time.monotonic() - started)
        if wait > 0:
            time.sleep(wait)
        features = {key: float(value) for key, value in record.items() if key not in {"subject", "session", "label"}}
        yield SourceFrame(t=t, rgb=None, features=features)


def _scripted_frames(seed: int) -> list[dict[str, Any]]:
    """The indicator demo session: calibration, then each scripted indicator event (features.synthetic)."""
    from ..features.synthetic import scripted_synthetic_session

    frames, _events = scripted_synthetic_session(seed)
    numeric = frames.select_dtypes(include=["number"]).drop(columns=["shoulder_y"])
    return [{str(key): value for key, value in row.items()} for row in numeric.to_dict(orient="records")]
