"""Turn frames into 60-second windows and compare each with the driver's own alert baseline.

Features are relative to the eye's own open level, the window's median head pose and gaze, so camera mounting
does not move them (feature spec v4). A driver's baseline comes from their alert video (public data) or their
calibration clip (team sessions), and every window also carries its change from it. Windows hop with a plain
loop, not ``rolling``, because the phone uses the same loop and the window fixtures check the port against it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt
import pandas as pd

from .data import TEAM_DATASET_KEY, dataset_key_of
from .features.frames import read_scripts
from .spec import (
    ACTED_EYES_OFF,
    ALERT_MARGIN_SECONDS,
    BASELINE_DELTA_PREFIX,
    CLIP_CALIBRATION,
    CLIP_DRIVE,
    FEATURE_SPEC,
    UNLABELED,
    WINDOW_FEATURES,
    WINDOW_INPUT_SIGNALS,
    FeatureSpec,
)
from .workspace import Workspace, read_parquet, write_parquet

logger = logging.getLogger(__name__)

ALERT_LABEL = 0
_CLOSURE_FEATURES: tuple[str, ...] = (
    "closure_mean",
    "closure_std",
    "perclos",
    "blink_rate",
    "blink_duration",
    "microsleeps",
)
DROWSY_LABEL = 2
BASELINE_SCOPE_PERSON = "person"
BASELINE_SCOPE_SESSION = "session"
WINDOW_KEY_COLUMNS: tuple[str, ...] = ("subject", "session", "clip", "label", "t0")


def run_lengths(mask: np.ndarray) -> np.ndarray:
    """Lengths of each run of True, in order."""
    padded = np.concatenate(([0], np.asarray(mask, dtype=np.int8), [0]))
    edges = np.diff(padded)
    return np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)


def _mean(values: np.ndarray) -> float:
    return float(values.mean()) if values.size else float("nan")


def _finite_std(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(finite.std()) if finite.size else float("nan")


def _finite_median(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def _closure_features(ear: np.ndarray, minutes: float, spec: FeatureSpec) -> dict[str, float]:
    """Eye closure relative to this window's open level, so a nearer or farther camera changes nothing."""
    thresholds, fps = spec.thresholds, spec.target_fps
    finite = ear[np.isfinite(ear)]
    open_level = float(np.percentile(finite, thresholds.open_eye_percentile)) if finite.size else float("nan")
    if not np.isfinite(open_level) or open_level <= 0:
        return dict.fromkeys(_CLOSURE_FEATURES, float("nan"))
    closure = np.clip(1 - ear / open_level, 0, 1)
    closed_runs = run_lengths(closure >= thresholds.closed_fraction)
    return {
        "closure_mean": float(np.nanmean(closure)),
        "closure_std": float(np.nanstd(closure)),
        "perclos": _mean(closure >= thresholds.perclos_fraction),
        "blink_rate": len(closed_runs) / minutes,
        "blink_duration": float(closed_runs.mean()) / fps if closed_runs.size else 0.0,
        "microsleeps": int(np.count_nonzero(closed_runs >= thresholds.microsleep_seconds * fps)),
    }


def _head_nods(pitch: np.ndarray, spec: FeatureSpec) -> float:
    """Dips of the head below the window's own median pitch that last like a nod, not a glance at the floor."""
    thresholds, fps = spec.thresholds, spec.target_fps
    centre = _finite_median(pitch)
    if not np.isfinite(centre):
        return float("nan")
    runs = run_lengths(pitch < centre - thresholds.nod_degrees)
    lasting = (runs >= thresholds.nod_min_seconds * fps) & (runs <= thresholds.nod_max_seconds * fps)
    return int(np.count_nonzero(lasting))


def _summarize_signals(signals: Mapping[str, np.ndarray], spec: FeatureSpec) -> dict[str, float]:
    thresholds = spec.thresholds
    fps = spec.target_fps
    has_face = signals["face"] > 0
    minutes = len(has_face) / fps / 60
    yawn_runs = run_lengths(signals["mar"][has_face] > thresholds.yawn_mar)
    gaze = signals["gaze_horizontal"][has_face]
    yaw = signals["yaw"][has_face]
    pitch = signals["pitch"][has_face]
    looking_away = (np.abs(gaze - _finite_median(gaze)) > thresholds.gaze_off_ratio) | (
        np.abs(yaw - _finite_median(yaw)) > thresholds.yaw_off_degrees
    )
    summary = {
        **_closure_features(signals["ear"][has_face], minutes, spec),
        "yawns": int(np.count_nonzero(yawn_runs >= thresholds.yawn_min_seconds * fps)),
        "pitch_std": _finite_std(pitch),
        "yaw_std": _finite_std(yaw),
        "head_nods": _head_nods(pitch, spec),
        "gaze_off": _mean(looking_away),
        "face_fraction": _mean(has_face),
    }
    return {name: summary[name] for name in WINDOW_FEATURES}


def summarize_window(frames: pd.DataFrame, spec: FeatureSpec = FEATURE_SPEC) -> dict[str, float]:
    """The 12 ``WINDOW_FEATURES`` of one window. Face statistics use face frames; rates use the full length."""
    return _summarize_signals(
        {name: frames[name].to_numpy(dtype=float) for name in WINDOW_INPUT_SIGNALS},
        spec,
    )


def slide_windows(frames: pd.DataFrame, spec: FeatureSpec = FEATURE_SPEC) -> pd.DataFrame:
    """Windows per (subject, session, clip): a team session's calibration and drive clips never share one."""
    rows = []
    size, hop = spec.frames_per_window, spec.frames_per_hop
    clipped = frames if "clip" in frames.columns else frames.assign(clip=CLIP_DRIVE)
    for (subject, session, clip), group in clipped.groupby(["subject", "session", "clip"], sort=False):
        ordered = group.sort_values("t", kind="stable")
        signals = {name: ordered[name].to_numpy(dtype=float) for name in WINDOW_INPUT_SIGNALS}
        times = ordered["t"].to_numpy(dtype=float)
        label = int(ordered["label"].iloc[0])
        for start in range(0, len(ordered) - size + 1, hop):
            window = {name: values[start : start + size] for name, values in signals.items()}
            if np.mean(window["face"] > 0) < spec.thresholds.min_face_fraction:
                continue
            rows.append(
                {
                    "subject": subject,
                    "session": session,
                    "clip": clip,
                    "label": label,
                    "t0": float(times[start]),
                    **_summarize_signals(window, spec),
                }
            )
    return pd.DataFrame(rows, columns=[*WINDOW_KEY_COLUMNS, *WINDOW_FEATURES])


def script_labels(t0: npt.ArrayLike, script: pd.DataFrame, spec: FeatureSpec = FEATURE_SPEC) -> np.ndarray:
    """Label drive windows from a script.

    2: the window overlaps an acted drowsiness, yawn or nod. 0: it is at least 30 s from every event.
    ``UNLABELED`` (dropped): it overlaps looking away, since eyes-off is a fixed rule and not the model's job, or
    it is part event and part not.
    """
    starts = np.asarray(t0, dtype=float)
    ends = starts + spec.window_seconds
    overlaps_drowsy = np.zeros(len(starts), dtype=bool)
    overlaps_eyes_off = np.zeros(len(starts), dtype=bool)
    near_event = np.zeros(len(starts), dtype=bool)
    for event_start, event_end, event in zip(script["start_s"], script["end_s"], script["event"], strict=True):
        overlaps = (starts < event_end) & (ends > event_start)
        if event == ACTED_EYES_OFF:
            overlaps_eyes_off |= overlaps
        else:
            overlaps_drowsy |= overlaps
        near_event |= (starts < event_end + ALERT_MARGIN_SECONDS) & (ends + ALERT_MARGIN_SECONDS > event_start)
    labels = np.where(overlaps_drowsy, DROWSY_LABEL, np.where(near_event, UNLABELED, ALERT_LABEL))
    return np.where(overlaps_eyes_off, UNLABELED, labels)


def label_scripted_windows(windows: pd.DataFrame, scripts: pd.DataFrame) -> pd.DataFrame:
    """Give team drive windows their script labels and drop the ones the script leaves unlabeled."""
    drive = (windows["clip"] == CLIP_DRIVE) & (windows["label"] == UNLABELED)
    labeled = windows.copy()
    for session, script in scripts.groupby("session", sort=False):
        rows = drive & (labeled["session"] == session)
        labeled.loc[rows, "label"] = script_labels(labeled.loc[rows, "t0"], script)
    unscripted = drive & ~labeled["session"].isin(scripts["session"])
    if unscripted.any():
        logger.warning("windows: dropped %d drive windows with no script", int(unscripted.sum()))
    return labeled.loc[labeled["label"] != UNLABELED].reset_index(drop=True)


def is_baseline_window(windows: pd.DataFrame, spec: FeatureSpec = FEATURE_SPEC) -> pd.Series:
    return (windows["label"] == ALERT_LABEL) & (windows["t0"] + spec.window_seconds <= spec.baseline_seconds)


def personal_baselines(windows: pd.DataFrame, spec: FeatureSpec = FEATURE_SPEC) -> pd.DataFrame:
    """Each person's mean window features over their baseline windows, indexed by subject."""
    baseline_windows = windows.loc[is_baseline_window(windows, spec)]
    return baseline_windows.groupby("subject")[list(WINDOW_FEATURES)].mean()


def _with_deltas(windows: pd.DataFrame, baseline: pd.DataFrame, scope: str) -> pd.DataFrame:
    deltas = pd.DataFrame(
        windows[list(WINDOW_FEATURES)].to_numpy() - baseline[list(WINDOW_FEATURES)].to_numpy(),
        columns=[f"{BASELINE_DELTA_PREFIX}{name}" for name in WINDOW_FEATURES],
    )
    return pd.concat([windows.reset_index(drop=True).assign(baseline_scope=scope), deltas], axis=1)


def _person_scope(windows: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Public datasets: the person's alert video, first ``baseline_seconds``, is the baseline for all their data."""
    baselines = personal_baselines(windows, spec)
    has_baseline = windows["subject"].isin(baselines.index)
    kept = windows.loc[~is_baseline_window(windows, spec) & has_baseline]
    without_baseline = windows.loc[~has_baseline, "subject"].nunique()
    if without_baseline:
        logger.warning(
            "windows: dropped %d people with no alert baseline windows",
            without_baseline,
        )
    return _with_deltas(kept, baselines.loc[kept["subject"]], BASELINE_SCOPE_PERSON)


def _session_scope(windows: pd.DataFrame) -> pd.DataFrame:
    """Team sessions: the calibration clip, same mount and sitting, is the baseline for that session's drive."""
    calibration = windows.loc[windows["clip"] == CLIP_CALIBRATION]
    baselines = calibration.groupby("session")[list(WINDOW_FEATURES)].mean()
    drive = windows.loc[windows["clip"] == CLIP_DRIVE]
    uncalibrated = sorted(set(drive["session"]) - set(baselines.index))
    if uncalibrated:
        logger.warning(
            "windows: dropped team sessions with no usable calibration window: %s",
            uncalibrated,
        )
    kept = drive.loc[drive["session"].isin(baselines.index)]
    return _with_deltas(kept, baselines.loc[kept["session"]], BASELINE_SCOPE_SESSION)


def add_baseline_deltas(windows: pd.DataFrame, spec: FeatureSpec = FEATURE_SPEC) -> pd.DataFrame:
    """Remove baseline windows and add ``delta_<feature>`` plus ``baseline_scope`` to every other window."""
    is_team = windows["subject"].map(dataset_key_of) == TEAM_DATASET_KEY
    parts = [
        _person_scope(windows.loc[~is_team], spec),
        _session_scope(windows.loc[is_team]),
    ]
    return pd.concat(parts, ignore_index=True)


def build_window_table(workspace: Workspace) -> pd.DataFrame:
    frames = read_parquet(workspace.frames_path, produced_by="features")
    windows = add_baseline_deltas(label_scripted_windows(slide_windows(frames), read_scripts(workspace)))
    write_parquet(workspace.windows_path, windows)
    logger.info(
        "windows: %d windows from %d people (%s)",
        len(windows),
        windows["subject"].nunique(),
        ", ".join(f"{scope}: {count}" for scope, count in windows["baseline_scope"].value_counts().items()),
    )
    return windows
