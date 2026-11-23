"""OpenCV image preparation before MediaPipe: low-light enhancement, a driver crop, and fisheye correction.

Each step targets a condition the condition test (``driven_ml.studio.conditions``) showed MediaPipe
handles badly on its own:

* ``enhance_low_light``: CLAHE on the L channel of LAB (Zuiderveld, "Contrast Limited Adaptive Histogram
  Equalization," Graphics Gems IV, 1994) plus a gamma lift, applied only when the frame is dark. It helps a
  dim cabin; it cannot create detail that the sensor never captured.
* ``crop_region``: a fixed region of interest (fractions of the frame) around the driver's seat. A cabin
  camera at the mirror sees the whole cabin, so the driver's face is small; MediaPipe's detector works on a
  downscaled frame and loses small faces. Cropping first gives the face more pixels.
* ``undistort_fisheye``: an equidistant fisheye model built from the lens's field of view (no calibration
  board needed), remapped to a rectilinear view. For VIOFO's 210-degree RNF210 cabin camera. A calibrated
  K and D matrix from ``cv2.fisheye.calibrate`` is better; the FOV model is the no-equipment approximation.

All three return a new array and never modify the input. Frames stay in memory (rule L6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import numpy.typing as npt

DARK_MEAN_LUMINANCE = 90.0  # 0-255; brighter frames are left alone
CLAHE_CLIP_LIMIT = 2.5
CLAHE_TILES = 8
MAX_RECTILINEAR_HALF_ANGLE_DEGREES = 75.0  # a rectilinear view cannot show 90 degrees off-axis

Image = npt.NDArray[np.uint8]


@dataclass(frozen=True)
class Preprocess:
    """What to do to each RGB frame before face landmarks. The default does nothing."""

    enhance_low_light: bool = False
    region: tuple[float, float, float, float] | None = None  # x0, y0, x1, y1 as fractions of width and height
    fisheye_fov_degrees: float | None = None

    @property
    def active(self) -> bool:
        return self.enhance_low_light or self.region is not None or self.fisheye_fov_degrees is not None

    def apply(self, rgb: Image) -> Image:
        """Fisheye correction first (it is a property of the lens), then the crop, then the light fix."""
        image = rgb
        if self.fisheye_fov_degrees is not None:
            image = undistort_fisheye(image, self.fisheye_fov_degrees)
        if self.region is not None:
            image = crop_region(image, self.region)
        if self.enhance_low_light:
            image = enhance_low_light(image)
        return image


def mean_luminance(rgb: Image) -> float:
    import cv2

    return float(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).mean())


def enhance_low_light(rgb: Image, *, dark_below: float = DARK_MEAN_LUMINANCE) -> Image:
    """CLAHE on lightness plus a gamma lift, only for frames darker than ``dark_below``."""
    import cv2

    brightness = mean_luminance(rgb)
    if brightness >= dark_below:
        return rgb.copy()
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lightness, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(CLAHE_TILES, CLAHE_TILES))
    lightness = clahe.apply(lightness)
    target = 0.45  # aim the mean lightness near the middle of the range
    current = max(float(lightness.mean()) / 255.0, 1e-3)
    gamma = float(np.clip(math.log(target) / math.log(current), 0.4, 1.0))
    table = np.clip(((np.arange(256) / 255.0) ** gamma) * 255.0, 0, 255).astype(np.uint8)
    lightness = cv2.LUT(lightness, table)
    enhanced = cv2.cvtColor(cv2.merge((lightness, a, b)), cv2.COLOR_LAB2RGB)
    return np.asarray(enhanced, dtype=np.uint8)


def parse_region(text: str) -> tuple[float, float, float, float]:
    """``"0.5,0.1,1.0,0.9"`` to a validated (x0, y0, x1, y1) in fractions."""
    parts = [float(part) for part in text.split(",")]
    if len(parts) != 4:
        raise ValueError("region needs four comma-separated fractions: x0,y0,x1,y1")
    x0, y0, x1, y1 = parts
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError("region fractions must satisfy 0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1")
    return x0, y0, x1, y1


def crop_region(rgb: Image, region: tuple[float, float, float, float]) -> Image:
    height, width = rgb.shape[:2]
    x0, y0, x1, y1 = region
    left, top = round(x0 * width), round(y0 * height)
    right, bottom = max(left + 1, round(x1 * width)), max(top + 1, round(y1 * height))
    return np.ascontiguousarray(rgb[top:bottom, left:right])


@lru_cache(maxsize=8)
def _undistort_maps(width: int, height: int, fov_degrees: float) -> tuple[np.ndarray, np.ndarray]:
    """Remap tables from an equidistant fisheye (r = f * theta) to a rectilinear view (r = f' * tan(theta))."""
    half_diagonal = math.hypot(width, height) / 2.0
    f_fisheye = half_diagonal / math.radians(fov_degrees / 2.0)
    max_angle = math.radians(min(fov_degrees / 2.0, MAX_RECTILINEAR_HALF_ANGLE_DEGREES))
    f_rect = half_diagonal / math.tan(max_angle)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    xs, ys = np.meshgrid(
        np.arange(width, dtype=np.float64) - cx,
        np.arange(height, dtype=np.float64) - cy,
    )
    r_rect = np.hypot(xs, ys)
    theta = np.arctan(r_rect / f_rect)
    scale = np.divide(f_fisheye * theta, r_rect, out=np.ones_like(r_rect), where=r_rect > 0)
    map_x = (xs * scale + cx).astype(np.float32)
    map_y = (ys * scale + cy).astype(np.float32)
    return map_x, map_y


def undistort_fisheye(rgb: Image, fov_degrees: float) -> Image:
    import cv2

    if not 60.0 <= fov_degrees <= 240.0:
        raise ValueError("fisheye field of view must be between 60 and 240 degrees")
    height, width = rgb.shape[:2]
    map_x, map_y = _undistort_maps(width, height, float(fov_degrees))
    return np.asarray(
        cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT),
        dtype=np.uint8,
    )


def distort_fisheye(rgb: Image, fov_degrees: float) -> Image:
    """The inverse of ``undistort_fisheye``: what a rectilinear scene looks like through the fisheye lens.

    Used by the condition test to simulate the RNF210; not used on live frames.
    """
    import cv2

    height, width = rgb.shape[:2]
    half_diagonal = math.hypot(width, height) / 2.0
    f_fisheye = half_diagonal / math.radians(fov_degrees / 2.0)
    max_angle = math.radians(min(fov_degrees / 2.0, MAX_RECTILINEAR_HALF_ANGLE_DEGREES))
    f_rect = half_diagonal / math.tan(max_angle)
    cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    xs, ys = np.meshgrid(
        np.arange(width, dtype=np.float64) - cx,
        np.arange(height, dtype=np.float64) - cy,
    )
    r_fish = np.hypot(xs, ys)
    theta = np.minimum(r_fish / f_fisheye, math.radians(89.0))
    scale = np.divide(f_rect * np.tan(theta), r_fish, out=np.ones_like(r_fish), where=r_fish > 0)
    map_x = (xs * scale + cx).astype(np.float32)
    map_y = (ys * scale + cy).astype(np.float32)
    return np.asarray(
        cv2.remap(rgb, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT),
        dtype=np.uint8,
    )
