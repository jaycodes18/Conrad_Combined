"""Build the frame table: synthetic people in dev, or landmarks from licensed, labeled video.

Public datasets use one folder per person, ``raw/<dataset>/<person>/{0,5,10}.<ext>`` (alert, low vigilance,
drowsy). Team recordings follow ``ml/RECORDING_PROTOCOL.md``: ``recordings/<person>/<session>/`` holds
``calibration.<ext>``, ``drive.<ext>`` and ``script.csv``. Team recordings are read only for people with
recorded consent (L5), and datasets the track may not use are never opened (L9).
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..data import DATASETS, TEAM_DATASET_KEY, is_allowed, load_consented_subjects
from ..errors import DataError
from ..settings import DataSource, RunConfig
from ..spec import (
    CLIP_CALIBRATION,
    CLIP_DRIVE,
    SCRIPT_COLUMNS,
    SCRIPT_EVENTS,
    UNLABELED,
)
from ..workspace import Workspace, write_parquet
from .extraction import LandmarkerSession, extract_video_features, open_face_landmarker
from .synthetic import simulate_sessions

logger = logging.getLogger(__name__)

LABEL_BY_FILE_STEM: dict[str, int] = {"0": 0, "5": 1, "10": 2}
VIDEO_SUFFIXES = frozenset({".mp4", ".mov", ".avi", ".mkv"})
SCRIPT_FILE = "script.csv"
SCRIPT_TABLE_COLUMNS: tuple[str, ...] = ("session", *SCRIPT_COLUMNS)

LandmarkerFactory = Callable[[Path], AbstractContextManager[LandmarkerSession]]


@dataclass(frozen=True)
class LabeledVideo:
    path: Path
    dataset_key: str
    subject_id: str
    session_id: str
    label: int
    clip: str = CLIP_DRIVE


@dataclass(frozen=True)
class TeamSession:
    calibration: LabeledVideo
    drive: LabeledVideo
    script_path: Path


def _find_video(directory: Path, stem: str) -> Path | None:
    matches = [path for path in sorted(directory.glob(f"{stem}.*")) if path.suffix.lower() in VIDEO_SUFFIXES]
    return matches[0] if matches else None


def iter_team_sessions(workspace: Workspace) -> Iterator[TeamSession]:
    """Consented team sessions with both clips and a script; anything incomplete is skipped with a warning."""
    consented = load_consented_subjects(workspace.consent_path)
    refused: set[str] = set()
    for session_dir in sorted(path for path in workspace.recordings_dir.glob("*/*") if path.is_dir()):
        person = session_dir.parent.name
        if person not in consented:
            if person not in refused:
                logger.warning(
                    "skipping recordings/%s: no recorded consent in consent.csv (L5)",
                    person,
                )
                refused.add(person)
            continue
        calibration, drive = _find_video(session_dir, CLIP_CALIBRATION), _find_video(session_dir, CLIP_DRIVE)
        script_path = session_dir / SCRIPT_FILE
        if calibration is None or drive is None or not script_path.is_file():
            logger.warning(
                "skipping recordings/%s/%s: it needs calibration and drive videos and %s",
                person,
                session_dir.name,
                SCRIPT_FILE,
            )
            continue
        subject_id = f"{TEAM_DATASET_KEY}:{person}"
        session_id = f"{subject_id}:{session_dir.name}"
        yield TeamSession(
            calibration=LabeledVideo(
                calibration,
                TEAM_DATASET_KEY,
                subject_id,
                session_id,
                0,
                CLIP_CALIBRATION,
            ),
            drive=LabeledVideo(drive, TEAM_DATASET_KEY, subject_id, session_id, UNLABELED, CLIP_DRIVE),
            script_path=script_path,
        )


def _public_videos(key: str, workspace: Workspace) -> Iterator[LabeledVideo]:
    for path in sorted((workspace.raw_dir / key).glob("*/*")):
        if path.suffix.lower() not in VIDEO_SUFFIXES or path.stem not in LABEL_BY_FILE_STEM:
            continue
        subject_id = f"{key}:{path.parent.name}"
        yield LabeledVideo(
            path,
            key,
            subject_id,
            f"{subject_id}:{path.stem}",
            LABEL_BY_FILE_STEM[path.stem],
        )


def iter_labeled_videos(config: RunConfig, workspace: Workspace) -> Iterator[LabeledVideo]:
    """Every video to extract, in registry order, skipping datasets the track may not use."""
    for key in DATASETS:
        if not is_allowed(key, config.track):
            continue
        if key == TEAM_DATASET_KEY:
            for session in iter_team_sessions(workspace):
                yield session.calibration
                yield session.drive
        else:
            yield from _public_videos(key, workspace)


def read_script(path: Path, session_id: str) -> pd.DataFrame:
    """A session's script, checked: start before end, and only the protocol's event names."""
    script = pd.read_csv(path)
    missing = [column for column in SCRIPT_COLUMNS if column not in script.columns]
    if missing:
        raise DataError(f"{path} is missing columns: {', '.join(missing)}")
    unknown = sorted(set(script["event"]) - SCRIPT_EVENTS)
    if unknown:
        raise DataError(f"{path} has unknown events {unknown}; use one of {sorted(SCRIPT_EVENTS)}")
    if (script["end_s"] <= script["start_s"]).any():
        raise DataError(f"{path} has an event that does not end after it starts")
    return script[list(SCRIPT_COLUMNS)].assign(session=session_id)[list(SCRIPT_TABLE_COLUMNS)]


def read_scripts(workspace: Workspace) -> pd.DataFrame:
    if not workspace.scripts_path.is_file():
        return pd.DataFrame(columns=list(SCRIPT_TABLE_COLUMNS))
    return pd.read_parquet(workspace.scripts_path)


def _team_scripts(config: RunConfig, workspace: Workspace) -> pd.DataFrame:
    if not is_allowed(TEAM_DATASET_KEY, config.track):
        return pd.DataFrame(columns=list(SCRIPT_TABLE_COLUMNS))
    scripts = [read_script(session.script_path, session.drive.session_id) for session in iter_team_sessions(workspace)]
    return pd.concat(scripts, ignore_index=True) if scripts else pd.DataFrame(columns=list(SCRIPT_TABLE_COLUMNS))


def _extract_real_frames(
    config: RunConfig, workspace: Workspace, landmarker_factory: LandmarkerFactory
) -> pd.DataFrame:
    videos = list(iter_labeled_videos(config, workspace))
    if not videos:
        raise DataError(
            f"no labeled videos for the {config.track} track: put videos in "
            f"{workspace.raw_dir}/<dataset>/<person>/{{0,5,10}}.mp4 or consented team sessions in "
            f"{workspace.recordings_dir}/<person>/<session>/ (see ml/RECORDING_PROTOCOL.md)"
        )
    tables = []
    for number, video in enumerate(videos, start=1):
        # VIDEO mode needs increasing timestamps and tracks the face between frames, so every video gets a
        # fresh landmarker; the model file is downloaded once and reused.
        with landmarker_factory(workspace.raw_dir) as session:
            try:
                table = extract_video_features(
                    video.path,
                    subject_id=video.subject_id,
                    session_id=video.session_id,
                    label=video.label,
                    session=session,
                    clip=video.clip,
                )
            except OSError as exc:
                raise DataError(f"cannot read {video.path}: {exc}") from exc
        logger.info(
            "features: [%d/%d] %s -> %d frames",
            number,
            len(videos),
            video.path,
            len(table),
        )
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def build_frame_table(
    config: RunConfig,
    workspace: Workspace,
    landmarker_factory: LandmarkerFactory = open_face_landmarker,
) -> pd.DataFrame:
    """Write ``frames.parquet`` and, for team sessions, ``scripts.parquet`` beside it."""
    if config.data_source is DataSource.SYNTHETIC:
        frames = simulate_sessions(seed=config.seed)
        scripts = pd.DataFrame(columns=list(SCRIPT_TABLE_COLUMNS))
    elif config.reuse_features and workspace.frames_path.is_file():
        logger.info("features: reusing %s", workspace.frames_path)
        return pd.read_parquet(workspace.frames_path)
    else:
        frames = _extract_real_frames(config, workspace, landmarker_factory)
        scripts = _team_scripts(config, workspace)
    write_parquet(workspace.frames_path, frames)
    write_parquet(workspace.scripts_path, scripts)
    logger.info(
        "features: %d frames from %d people, %d scripted team sessions",
        len(frames),
        frames["subject"].nunique(),
        scripts["session"].nunique(),
    )
    return frames
