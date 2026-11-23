"""Replay scripted sessions as the phone scores them, and count caught events and false alerts.

A scripted session is a parked-car recording with acted drowsiness at known times (L11). Dev replays a
synthetic scripted person outside training. Real runs replay only held-out people (replay people and locked
test people), each drive scored against its own calibration clip.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..data import load_consented_subjects
from ..features.frames import read_scripts
from ..features.synthetic import act_event, simulate_sessions
from ..modeling.alert_simulation import alert_window_starts
from ..modeling.artifact import ModelArtifact, predict_probability
from ..modeling.store import load_candidate
from ..scoring import WithinRecording, within_recording
from ..settings import DataSource, RunConfig
from ..spec import (
    ACTED_DROWSINESS,
    BASELINE_DELTA_PREFIX,
    CLIP_CALIBRATION,
    CLIP_DRIVE,
    FEATURE_SPEC,
    SCRIPT_COLUMNS,
    UNLABELED,
    WINDOW_FEATURES,
    FeatureSpec,
)
from ..splits import is_locked_test_subject, replay_subjects
from ..windows import DROWSY_LABEL, script_labels, slide_windows
from ..workspace import Workspace, read_parquet, write_text_atomic

logger = logging.getLogger(__name__)

SCRIPTED_EVENT_STARTS = range(240, 660, 150)
SCRIPTED_EVENT_SECONDS = 40
REPLAY_FILE = "replay.csv"


GROUP_REPLAY = "replay"
GROUP_TEST = "test"


@dataclass(frozen=True)
class ReplayResult:
    """One replayed session. ``group`` says how its person is held out: a replay person or a locked test person."""

    session: str
    person: str
    group: str
    events: int
    detected: int
    false_alerts: int
    alert_times: tuple[float, ...]
    within_recording: WithinRecording


def scripted_synthetic_session(seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A 12-minute alert session with three 40-second acted drowsiness events: long eye closures, head drop."""
    frames = simulate_sessions(subject_count=1, minutes=12, seed=seed + 1)
    frames = frames.loc[frames["label"] == 0].reset_index(drop=True)
    for start in SCRIPTED_EVENT_STARTS:
        act_event(frames, start, SCRIPTED_EVENT_SECONDS, ACTED_DROWSINESS)
    events = pd.DataFrame(
        [
            {
                "start_s": float(start),
                "end_s": float(start + SCRIPTED_EVENT_SECONDS),
                "event": ACTED_DROWSINESS,
            }
            for start in SCRIPTED_EVENT_STARTS
        ],
        columns=list(SCRIPT_COLUMNS),
    )
    frames["session"] = "SCRIPT_1"
    frames["subject"] = "SCRIPT"
    return frames, events


def score_session(
    frames: pd.DataFrame,
    artifact: ModelArtifact,
    spec: FeatureSpec = FEATURE_SPEC,
    *,
    calibration: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Windows of one recording and their probabilities, with the baseline the phone would have.

    With a calibration clip the baseline is the mean of its windows; without one it is the session's own first
    ``baseline_seconds``.
    """
    windows = slide_windows(frames.assign(label=0, clip=CLIP_DRIVE), spec)
    if windows.empty:
        return windows, np.empty(0)
    features = windows[list(WINDOW_FEATURES)]
    if calibration is not None:
        baseline = slide_windows(calibration.assign(label=0, clip=CLIP_CALIBRATION), spec)[list(WINDOW_FEATURES)]
    else:
        baseline = features.loc[windows["t0"] + spec.window_seconds <= spec.baseline_seconds]
    deltas = (features - baseline.mean()).add_prefix(BASELINE_DELTA_PREFIX)
    scored = pd.concat([windows, deltas], axis=1)
    return windows, predict_probability(artifact, scored[artifact.inputs])


def alert_times_for_session(
    frames: pd.DataFrame,
    artifact: ModelArtifact,
    spec: FeatureSpec = FEATURE_SPEC,
    *,
    calibration: pd.DataFrame | None = None,
) -> list[float]:
    """When the phone speaks an alert, in seconds: each window that fires is spoken as the window closes."""
    windows, probabilities = score_session(frames, artifact, spec, calibration=calibration)
    if windows.empty:
        return []
    fired = alert_window_starts(windows["t0"], probabilities, artifact.threshold, artifact.cooldown_seconds)
    return [start + spec.window_seconds for start in fired]


def replay_session(
    name: str,
    frames: pd.DataFrame,
    script: pd.DataFrame,
    artifact: ModelArtifact,
    *,
    calibration: pd.DataFrame | None = None,
    person: str = "",
    group: str = GROUP_REPLAY,
) -> ReplayResult:
    """Score one scripted session. An acted drowsiness event is caught when an alert is spoken after it starts
    and before its last window closes; an alert exactly at the start comes from a window that ended before the
    event. Alerts near any scripted event (yawns, nods and looking away included) are not false alerts.
    """
    windows, probabilities = score_session(frames, artifact, calibration=calibration)
    fired = alert_window_starts(windows["t0"], probabilities, artifact.threshold, artifact.cooldown_seconds)
    alert_times = [start + FEATURE_SPEC.window_seconds for start in fired]

    def spans(events: pd.DataFrame) -> list[tuple[float, float]]:
        return [
            (float(start), float(end) + FEATURE_SPEC.window_seconds)
            for start, end in zip(events["start_s"], events["end_s"], strict=True)
        ]

    drowsiness = spans(script.loc[script["event"] == ACTED_DROWSINESS])
    every_event = spans(script)
    detected = sum(any(start < time <= end for time in alert_times) for start, end in drowsiness)
    false_alerts = sum(not any(start < time <= end for start, end in every_event) for time in alert_times)

    labels = script_labels(windows["t0"], script) if len(windows) else np.empty(0, dtype=int)
    labeled = labels != UNLABELED
    scored = pd.DataFrame(
        {
            "session": name,
            "clip": CLIP_DRIVE,
            "y": (labels[labeled] == DROWSY_LABEL).astype(int),
        }
    )
    return ReplayResult(
        session=name,
        person=person or name,
        group=group,
        events=len(drowsiness),
        detected=detected,
        false_alerts=false_alerts,
        alert_times=tuple(alert_times),
        within_recording=within_recording(scored, probabilities[labeled]),
    )


@dataclass(frozen=True)
class _ReplaySession:
    name: str
    person: str
    group: str
    drive: pd.DataFrame
    script: pd.DataFrame
    calibration: pd.DataFrame


def _team_replay_sessions(config: RunConfig, workspace: Workspace) -> list[_ReplaySession]:
    """Scripted drowsiness sessions of held-out people only: replay people and locked test people."""
    frames = read_parquet(workspace.frames_path, produced_by="features")
    scripts = read_scripts(workspace)
    consented = load_consented_subjects(workspace.consent_path)
    replay_people = replay_subjects(config, workspace)
    sessions = []
    for session, script in scripts.groupby("session", sort=True):
        subject = str(session).rsplit(":", 1)[0]
        person = subject.split(":", 1)[1]
        if person not in consented:
            logger.warning("replay: skipping %s, no recorded consent (L5)", session)
            continue
        if is_locked_test_subject(subject):
            group = GROUP_TEST
        elif subject in replay_people:
            group = GROUP_REPLAY
        else:
            continue
        if not (script["event"] == ACTED_DROWSINESS).any():
            continue
        recording = frames.loc[frames["session"] == session]
        sessions.append(
            _ReplaySession(
                name=str(session),
                person=person,
                group=group,
                drive=recording.loc[recording["clip"] == CLIP_DRIVE],
                script=script.reset_index(drop=True),
                calibration=recording.loc[recording["clip"] == CLIP_CALIBRATION],
            )
        )
    return sessions


def replay_recordings(config: RunConfig, workspace: Workspace) -> list[ReplayResult]:
    artifact = load_candidate(workspace)
    if config.data_source is DataSource.SYNTHETIC:
        frames, script = scripted_synthetic_session(config.seed)
        results = [replay_session("synthetic/SCRIPT_1", frames, script, artifact, person="SCRIPT")]
    else:
        results = [
            replay_session(
                session.name,
                session.drive,
                session.script,
                artifact,
                calibration=session.calibration,
                person=session.person,
                group=session.group,
            )
            for session in _team_replay_sessions(config, workspace)
        ]
    table = pd.DataFrame(
        [
            {
                "session": result.session,
                "person": result.person,
                "group": result.group,
                "events": result.events,
                "detected": result.detected,
                "false_alerts": result.false_alerts,
                "alert_times": ";".join(f"{time:g}" for time in result.alert_times),
                "within_recording_auc": result.within_recording.auc,
            }
            for result in results
        ],
        columns=[
            "session",
            "person",
            "group",
            "events",
            "detected",
            "false_alerts",
            "alert_times",
            "within_recording_auc",
        ],
    )
    write_text_atomic(
        workspace.staging_dir(artifact.short_sha) / REPLAY_FILE,
        table.to_csv(index=False),
    )
    for result in results:
        logger.info(
            "replay: %s (%s person) caught %d/%d scripted drowsiness events, %d false alerts, "
            "within-recording AUC %.3f",
            result.session,
            result.group,
            result.detected,
            result.events,
            result.false_alerts,
            result.within_recording.auc,
        )
    if not results:
        logger.warning("replay: no scripted sessions of held-out people to replay")
    return results


def detection_rate(results: list[ReplayResult]) -> float:
    events = sum(result.events for result in results)
    return sum(result.detected for result in results) / events if events else 0.0
