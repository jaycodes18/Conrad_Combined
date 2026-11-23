"""Where every stage reads and writes, plus atomic JSON and parquet helpers.

Dev writes under ``<root>/dev`` so synthetic output can never be mistaken for a real run. Recordings are shared
by all environments because they are the team's consented source data, not pipeline output. Every write goes
to a temp file first and is moved into place with ``os.replace``, so a crash never leaves a half-written file.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import tempfile
from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .errors import MissingArtifactError
from .settings import Environment, RunConfig


class Workspace:
    """Paths for one run. Call ``ensure()`` before writing."""

    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.root = config.workspace_root
        self.base = self.root / "dev" if config.environment is Environment.DEV else self.root
        self.recordings_dir = self.root / "recordings"
        self.raw_dir = self.base / "raw"
        self.features_dir = self.base / "features"
        self.windows_dir = self.base / "windows"
        self.models_dir = self.base / "models"
        self.reports_dir = self.base / "reports"
        self.staging_root = self.base / "staging"
        self.production_dir = self.base / "production"

    def ensure(self) -> Workspace:
        for directory in (
            self.recordings_dir,
            self.raw_dir,
            self.features_dir,
            self.windows_dir,
            self.models_dir,
            self.reports_dir,
            self.staging_root,
            self.production_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def frames_path(self) -> Path:
        return self.features_dir / "frames.parquet"

    @property
    def scripts_path(self) -> Path:
        """Scripts of team drive clips, saved beside the frames so windows can be labeled in any environment."""
        return self.features_dir / "scripts.parquet"

    @property
    def windows_path(self) -> Path:
        return self.windows_dir / "windows.parquet"

    @property
    def candidate_pointer(self) -> Path:
        """Text file holding the file name of the current candidate model."""
        return self.models_dir / "CANDIDATE"

    @property
    def test_run_log(self) -> Path:
        return self.reports_dir / "test_runs.json"

    @property
    def consent_path(self) -> Path:
        return self.recordings_dir / "consent.csv"

    @property
    def extra_strings_path(self) -> Path:
        return self.root / "app_strings.json"

    def candidate_path(self, short_sha: str) -> Path:
        return self.models_dir / f"candidate-{short_sha}.json"

    def staging_dir(self, short_sha: str) -> Path:
        directory = self.staging_root / short_sha
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def test_report_path(self, short_sha: str) -> Path:
        return self.reports_dir / f"test-{short_sha}.json"

    def economics_report_path(self, short_sha: str) -> Path:
        return self.reports_dir / f"economics-{short_sha}.json"


def to_jsonable(value: Any) -> Any:
    """Convert dataclasses, enums, numpy values and paths into plain JSON types; NaN and inf become None."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = dataclasses.asdict(value)
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [to_jsonable(item) for item in value]
    return _scalar_to_jsonable(value)


def _scalar_to_jsonable(value: Any) -> Any:
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def write_text_atomic(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return path


def copy_text_atomic(source: Path, destination: Path) -> Path:
    return write_text_atomic(destination, source.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any, *, indent: int | None = 1) -> Path:
    return write_text_atomic(path, json.dumps(to_jsonable(payload), indent=indent, allow_nan=False))


def read_json(path: Path, *, produced_by: str) -> Any:
    if not path.is_file():
        raise MissingArtifactError(str(path), produced_by)
    return json.loads(path.read_text(encoding="utf-8"))


def write_parquet(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.tmp")
    try:
        frame.to_parquet(temp_path, index=False)
        os.replace(temp_path, path)
    except BaseException:
        temp_path.unlink(missing_ok=True)
        raise
    return path


def read_parquet(path: Path, *, produced_by: str) -> pd.DataFrame:
    if not path.is_file():
        raise MissingArtifactError(str(path), produced_by)
    return pd.read_parquet(path)
