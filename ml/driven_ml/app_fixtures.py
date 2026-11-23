"""Fixtures that pin down the on-device port: frame, window, replay and model parity cases.

The phone re-implements frame features, windows, the baseline, the forward pass and the alert loop. These
files hold inputs and the reference outputs from this package, so a port is correct only when it reproduces
every one of them. The shapes here are the contract between this pipeline and the app.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features.geometry import frame_features, no_face_frame
from .features.synthetic import simulate_sessions
from .modeling.artifact import ModelArtifact
from .spec import (
    BASELINE_DELTA_PREFIX,
    FEATURE_SPEC,
    WINDOW_FEATURES,
    WINDOW_INPUT_SIGNALS,
)
from .staging.packaging import PARITY_FILE, parity_document
from .staging.replay import alert_times_for_session, scripted_synthetic_session
from .windows import slide_windows
from .workspace import Workspace, copy_text_atomic, write_json

logger = logging.getLogger(__name__)

FRAME_CASES = 8
FACE_LOSS_GAPS = 12
FACE_LOSS_FRAMES = 31
NO_POSE_START_SECONDS = 150  # a window start, so one window has a face in every frame but no pitch at all
MODEL_FIXTURE_FILE = "model.json"
FEATURE_SPEC_FIXTURE_FILE = "feature_spec.json"
FRAME_FIXTURE_FILE = "frame_fixtures.json"
WINDOW_FIXTURE_FILE = "window_fixtures.json"
REPLAY_FIXTURE_FILE = "replay_fixture.json"
FIXTURE_FILES: tuple[str, ...] = (
    MODEL_FIXTURE_FILE,
    FEATURE_SPEC_FIXTURE_FILE,
    FRAME_FIXTURE_FILE,
    WINDOW_FIXTURE_FILE,
    REPLAY_FIXTURE_FILE,
    PARITY_FILE,
)
BLENDSHAPE_NAMES: tuple[str, ...] = (
    "eyeBlinkLeft",
    "eyeBlinkRight",
    "jawOpen",
    "eyeLookDownLeft",
    "eyeLookDownRight",
)
APP_FIXTURES_DIR = Path("test") / "fixtures"
APP_DEV_MODELS_DIR = Path("dev-models")


@dataclass(frozen=True)
class FixtureExport:
    directory: Path
    files: tuple[Path, ...]


def _rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Rz(roll) · Ry(yaw) · Rx(pitch), in degrees, inside a 4x4 identity; ``head_pose`` inverts it."""
    y, p, r = (math.radians(angle) for angle in (yaw, pitch, roll))
    rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    matrix = np.eye(4)
    matrix[:3, :3] = rz @ ry @ rx
    return matrix


def frame_cases(seed: int = 3) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    cases = []
    for _ in range(FRAME_CASES):
        points = rng.uniform(0, 640, (478, 2))
        blendshapes = {name: float(rng.uniform()) for name in BLENDSHAPE_NAMES}
        matrix = _rotation_matrix(*rng.uniform(-40, 40, 3))
        cases.append(
            {
                "points": points,
                "blendshapes": blendshapes,
                "matrix_row_major": matrix.ravel(),
                "expected": frame_features(points, blendshapes, matrix),
            }
        )
    return cases


def _signal_columns(frames: pd.DataFrame) -> dict[str, list[float]]:
    return {name: frames[name].tolist() for name in ("t", *WINDOW_INPUT_SIGNALS)}


def window_case(seed: int = 5) -> dict[str, Any]:
    """Four drowsy minutes with 12 face-loss gaps and one minute with a face but no head pose: windows, the
    phone's baseline and the deltas. The no-pose minute forces ``pitch_std`` and ``head_nods`` to null.
    """
    frames = simulate_sessions(subject_count=1, minutes=4, seed=seed)
    frames = frames.loc[frames["label"] == 2].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    gap = no_face_frame()
    for start in rng.integers(0, len(frames) - 40, FACE_LOSS_GAPS):
        frames.loc[start : start + FACE_LOSS_FRAMES - 1, list(gap)] = list(gap.values())
    no_pose = (frames["t"] >= NO_POSE_START_SECONDS) & (
        frames["t"] < NO_POSE_START_SECONDS + FEATURE_SPEC.window_seconds
    )
    frames.loc[no_pose, "pitch"] = np.nan
    windows = slide_windows(frames)
    features = windows[list(WINDOW_FEATURES)]
    # No label filter: the phone's baseline is whatever it recorded during calibration.
    baseline = features.loc[windows["t0"] + FEATURE_SPEC.window_seconds <= FEATURE_SPEC.baseline_seconds].mean()
    deltas = features - baseline
    return {
        "frames": _signal_columns(frames),
        "windows": windows[["t0", *WINDOW_FEATURES]].to_dict(orient="records"),
        "baseline": baseline.to_dict(),
        "deltas": {f"{BASELINE_DELTA_PREFIX}{name}": deltas[name].tolist() for name in WINDOW_FEATURES},
    }


def replay_case(artifact: ModelArtifact, seed: int) -> dict[str, Any]:
    frames, script = scripted_synthetic_session(seed)
    return {
        "frames": _signal_columns(frames),
        "script": script.to_dict(orient="records"),
        "alert_times": alert_times_for_session(frames, artifact),
    }


def model_case(artifact: ModelArtifact, inputs: np.ndarray) -> dict[str, Any]:
    return parity_document(artifact, inputs)


def export_fixtures(artifact: ModelArtifact, parity_inputs: np.ndarray, directory: Path, seed: int) -> FixtureExport:
    documents: dict[str, Any] = {
        MODEL_FIXTURE_FILE: artifact.to_json(),
        FEATURE_SPEC_FIXTURE_FILE: FEATURE_SPEC.to_json(),
        FRAME_FIXTURE_FILE: frame_cases(),
        WINDOW_FIXTURE_FILE: window_case(),
        REPLAY_FIXTURE_FILE: replay_case(artifact, seed),
        PARITY_FILE: model_case(artifact, parity_inputs),
    }
    files = tuple(write_json(directory / name, documents[name], indent=None) for name in FIXTURE_FILES)
    logger.info("app fixtures: %d files in %s", len(files), directory)
    return FixtureExport(directory=directory, files=files)


def copy_fixtures_into_app(export: FixtureExport, app_dir: Path, *, include_dev_model: bool) -> list[Path]:
    """Copy fixtures into ``test/fixtures``; in dev also give the Vite dev server a model. Never ``public/``."""
    written = [copy_text_atomic(export.directory / name, app_dir / APP_FIXTURES_DIR / name) for name in FIXTURE_FILES]
    if include_dev_model:
        for name in (MODEL_FIXTURE_FILE, FEATURE_SPEC_FIXTURE_FILE):
            written.append(copy_text_atomic(export.directory / name, app_dir / APP_DEV_MODELS_DIR / name))
    logger.info("app fixtures: copied %d files into %s", len(written), app_dir)
    return written


def fixtures_directory(workspace: Workspace, short_sha: str) -> Path:
    return workspace.staging_dir(short_sha) / "app_fixtures"
