"""Camera adapters and a hardware compatibility registry.

Phone, file, and network sources already enter through `sources.iter_camera`. This module is the
contract those sources must meet, plus a registry of devices that have not been bench-tested.
A registry row is not a claim that the device works.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from ..spec import FEATURE_SPEC, FeatureSpec


class CameraKind(StrEnum):
    PHONE = "phone"
    USB = "usb"
    RTSP = "rtsp"
    HTTP = "http"
    FILE = "file"
    SYNTHETIC = "synthetic"


class HardwareStatus(StrEnum):
    CODE_PATH = "implemented_not_verified"
    NOT_TESTED = "not_tested"
    DOCUMENTED_ONLY = "documented_only"
    UNSUPPORTED = "unsupported"


class CameraError(Exception):
    """A camera failed. This is not a driver-behavior event."""

    def __init__(self, source: str, detail: str) -> None:
        self.source = source
        self.detail = detail
        super().__init__(f"camera {source}: {detail}")


@dataclass(frozen=True)
class NormalizedFrame:
    """What every adapter must hand the detection engine, independent of the device."""

    t_seconds: float
    width: int
    height: int
    fps: float
    orientation: str
    connected: bool
    dropped: bool
    error: str | None
    rgb: NDArray[np.uint8] | None = None


@dataclass(frozen=True)
class HardwareDevice:
    key: str
    name: str
    kind: CameraKind
    status: HardwareStatus
    connection: str
    inward_facing: str
    tested_on: str
    limitations: str

    def to_json(self) -> dict[str, str]:
        return {
            "key": self.key,
            "name": self.name,
            "kind": self.kind.value,
            "status": self.status.value,
            "connection": self.connection,
            "inward_facing": self.inward_facing,
            "tested_on": self.tested_on,
            "limitations": self.limitations,
        }


DEVICES: tuple[HardwareDevice, ...] = (
    HardwareDevice(
        "phone_or_webcam",
        "Phone or computer webcam",
        CameraKind.PHONE,
        HardwareStatus.CODE_PATH,
        "OpenCV camera index, usually 0",
        "driver-facing when the phone is mounted on the dash",
        "not bench-tested in this session",
        "Needs camera permission for the process that runs Python. Frames stay on this computer.",
    ),
    HardwareDevice(
        "usb_uvc",
        "USB / UVC camera",
        CameraKind.USB,
        HardwareStatus.CODE_PATH,
        "OpenCV camera index",
        "only if the lens is aimed at the driver",
        "not bench-tested in this session",
        "Same code path as a webcam. No UVC device was attached for this change.",
    ),
    HardwareDevice(
        "rtsp_generic",
        "RTSP or IP camera",
        CameraKind.RTSP,
        HardwareStatus.CODE_PATH,
        "rtsp:// URL, TCP transport by default",
        "not assumed",
        "not bench-tested in this session",
        "Many dashcams have no inward camera and no open stream. Verify the SDK before claiming support.",
    ),
    HardwareDevice(
        "http_stream",
        "HTTP live view",
        CameraKind.HTTP,
        HardwareStatus.CODE_PATH,
        "http:// or https:// URL",
        "not assumed",
        "not bench-tested in this session",
        "Novatek cameras may need command 2015 before the stream starts. That command was not sent to a device.",
    ),
    HardwareDevice(
        "viofo_a229_plus",
        "VIOFO A229 Plus",
        CameraKind.RTSP,
        HardwareStatus.DOCUMENTED_ONLY,
        "camera Wi-Fi, rtsp://192.168.1.254, one channel at a time",
        "front and rear face the road; cabin view needs the RF200 or a 3CH interior camera",
        "not tested",
        "Do not claim dashcam compatibility. Reviewers report RTSP failures at high resolution.",
    ),
    HardwareDevice(
        "file_replay",
        "Video file",
        CameraKind.FILE,
        HardwareStatus.CODE_PATH,
        "local path",
        "depends on the recording",
        "not bench-tested in this session",
        "Sampled by file timestamp at the feature-spec fps. Below 10 fps is refused by the extractor.",
    ),
    HardwareDevice(
        "synthetic",
        "Synthetic driver",
        CameraKind.SYNTHETIC,
        HardwareStatus.CODE_PATH,
        "no camera",
        "not a camera",
        "unit-tested as signals, not as video",
        "Scripted features only. Not a substitute for a person in a car.",
    ),
)


class CameraAdapter(Protocol):
    """Hardware-specific open/read/close. The detection engine consumes `NormalizedFrame` only."""

    kind: CameraKind
    source: str

    def open(self) -> None: ...

    def read(self) -> NormalizedFrame: ...

    def close(self) -> None: ...


def classify_source(source: str) -> CameraKind:
    if source == "synthetic":
        return CameraKind.SYNTHETIC
    if source.isdigit():
        return CameraKind.PHONE if source == "0" else CameraKind.USB
    if source.startswith("rtsp://"):
        return CameraKind.RTSP
    if source.startswith(("http://", "https://")):
        return CameraKind.HTTP
    return CameraKind.FILE


def expected_step_seconds(spec: FeatureSpec = FEATURE_SPEC) -> float:
    return 1.0 / spec.target_fps


def frame_gap_dropped(previous_t: float | None, t_seconds: float, step_seconds: float) -> bool:
    """A gap larger than 1.5 sample periods counts as a dropped frame, not as driver behavior."""
    if previous_t is None:
        return False
    return t_seconds - previous_t > step_seconds * 1.5


def timestamp_advances(previous_t: float | None, t_seconds: float) -> bool:
    return previous_t is None or t_seconds > previous_t


def normalize_rgb(
    rgb: NDArray[np.uint8],
    t_seconds: float,
    fps: float,
    *,
    dropped: bool,
    error: str | None,
) -> NormalizedFrame:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise CameraError("frame", "expected an RGB image")
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    return NormalizedFrame(
        t_seconds=t_seconds,
        width=width,
        height=height,
        fps=fps,
        orientation="upright",
        connected=error is None,
        dropped=dropped,
        error=error,
        rgb=rgb,
    )


def registry_json() -> list[dict[str, str]]:
    return [device.to_json() for device in DEVICES]
