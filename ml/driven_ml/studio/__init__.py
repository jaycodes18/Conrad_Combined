"""Driven Studio: watch the drowsiness model work on a live camera and watch it train.

Run `python -m driven_ml.studio` and open http://127.0.0.1:8765. Live mode runs the exact package
functions the pipeline uses (frame_features, summarize_window, predict_probability) on a webcam, a
dashcam stream or a video file. Training mode runs the pipeline stage by stage and charts each result.
Nothing here is used by the pipeline itself, and nothing in the studio sends a text or stores video.
"""

from __future__ import annotations

__all__ = ["main"]


def main() -> int:
    from .server import main as run

    return run()
