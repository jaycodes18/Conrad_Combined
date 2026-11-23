"""Frame features: MediaPipe landmarks to per-frame signals, real video extraction and synthetic dev people."""

from __future__ import annotations

from .frames import build_frame_table
from .geometry import frame_features, head_pose, no_face_frame
from .synthetic import simulate_sessions

__all__ = [
    "build_frame_table",
    "frame_features",
    "head_pose",
    "no_face_frame",
    "simulate_sessions",
]
