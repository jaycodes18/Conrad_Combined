"""Synthetic people for dev runs, so the pipeline runs with no real video.

Each person has their own eye size, blink habit and drowsiness visibility, so a fixed threshold works for some
people and not others. Each recording also gets its own camera placement, which shifts measured eye opening,
mouth opening, head pose and gaze. A model can learn "different recording" instead of "drowsy" (Geirhos et al.,
Nature Machine Intelligence 2, 2020), so dev data carries that confound. ``label_effect=False`` keeps the labels
but removes every sign of drowsiness; the null-signal control uses it.

The random draw order is fixed: every dev gate is calibrated on the data it produces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..spec import (
    ACTED_DROWSINESS,
    ACTED_EYES_OFF,
    ACTED_NOD,
    ACTED_YAWN,
    CLIP_CALIBRATION,
    CLIP_DRIVE,
    FRAME_FEATURES,
    SCRIPT_COLUMNS,
    UNLABELED,
)

ALERTNESS_LABELS: tuple[int, ...] = (0, 1, 2)
_CLOSED_EYE_EAR = 0.07
_FPS = 10


@dataclass(frozen=True)
class CameraPlacement:
    """One recording's mount: distance and angle scale the eye and mouth, and offset head pose and gaze."""

    ear_scale: float
    mar_scale: float
    pitch_offset_degrees: float
    yaw_offset_degrees: float
    gaze_offset: float

    @classmethod
    def draw(cls, rng: np.random.Generator) -> CameraPlacement:
        ear_scale = float(rng.uniform(0.85, 1.15))
        mar_scale = float(rng.uniform(0.90, 1.10))
        pitch_offset = float(rng.normal(0, 6))
        yaw_offset = float(rng.normal(0, 6))
        gaze_offset = float(rng.normal(0, 0.03))
        return cls(ear_scale, mar_scale, pitch_offset, yaw_offset, gaze_offset)


@dataclass(frozen=True)
class PersonProfile:
    wears_glasses: bool
    open_eye_ear: float
    blinks_per_minute: float
    blink_seconds: float
    drowsiness_visibility: float
    resting_mar: float
    resting_pitch: float
    resting_yaw: float

    @classmethod
    def draw(cls, rng: np.random.Generator) -> PersonProfile:
        wears_glasses = bool(rng.random() < 0.3)
        open_eye_ear = float(rng.normal(0.24 if wears_glasses else 0.31, 0.03))
        blinks_per_minute = float(rng.uniform(9, 22))
        blink_seconds = float(rng.uniform(0.10, 0.32))
        drowsiness_visibility = float(rng.uniform(0.3, 1.0))
        resting_mar = float(rng.normal(0.12, 0.03))
        resting_pitch = float(rng.normal(-5, 4))
        resting_yaw = float(rng.normal(0, 4))
        return cls(
            wears_glasses=wears_glasses,
            open_eye_ear=open_eye_ear,
            blinks_per_minute=blinks_per_minute,
            blink_seconds=blink_seconds,
            drowsiness_visibility=drowsiness_visibility,
            resting_mar=resting_mar,
            resting_pitch=resting_pitch,
            resting_yaw=resting_yaw,
        )


def _eye_openness(rng: np.random.Generator, person: PersonProfile, severity: float, n: int) -> np.ndarray:
    ear = person.open_eye_ear * (1 - 0.12 * severity) + rng.normal(0, 0.015, n)
    session_seconds = n / _FPS
    t = 0.0
    while True:
        t += rng.exponential(60 / (person.blinks_per_minute + 8 * severity))
        if t > session_seconds:
            return ear
        is_long = not rng.random() > 0.25 * severity
        duration = rng.uniform(0.4, 1.6) if is_long else rng.uniform(0.6, 1.4) * person.blink_seconds
        start, end = int(t * _FPS), int((t + duration) * _FPS) + 1
        ear[start:end] = rng.normal(_CLOSED_EYE_EAR, 0.02)


def _mouth_openness(
    rng: np.random.Generator,
    person: PersonProfile,
    severity: float,
    n: int,
    minutes: int,
) -> np.ndarray:
    mar = person.resting_mar + rng.normal(0, 0.02, n)
    for _ in range(rng.poisson(minutes * (rng.uniform(0.05, 0.4) + 0.9 * severity))):
        start = rng.integers(0, n - 60)
        length = rng.integers(25, 55)
        mar[start : start + length] = rng.normal(0.75, 0.05)
    return mar


def _head_pitch(
    rng: np.random.Generator,
    person: PersonProfile,
    severity: float,
    n: int,
    minutes: int,
) -> np.ndarray:
    pitch = person.resting_pitch + rng.normal(0, 2 + 3 * severity, n)
    for _ in range(rng.poisson(minutes * 0.8 * severity)):
        start = rng.integers(0, n - 30)
        length = rng.integers(8, 25)
        pitch[start : start + length] -= rng.uniform(12, 25)
    return pitch


def _simulate_session(
    rng: np.random.Generator,
    person: PersonProfile,
    subject: str,
    label: int,
    minutes: int,
    *,
    label_effect: bool = True,
) -> pd.DataFrame:
    camera = CameraPlacement.draw(rng)
    n = minutes * 60 * _FPS
    severity = label / 2 * person.drowsiness_visibility if label_effect else 0.0
    ear = _eye_openness(rng, person, severity, n)
    mar = _mouth_openness(rng, person, severity, n, minutes)
    pitch = _head_pitch(rng, person, severity, n, minutes)
    face = rng.random(n) > 0.02
    yaw = person.resting_yaw + rng.normal(0, 4, n)
    gaze = rng.normal(0, 0.05, n)
    ear_left = ear + rng.normal(0, 0.005, n)
    ear_right = ear + rng.normal(0, 0.005, n)
    roll = rng.normal(0, 2, n)
    # Blendshapes are normalized per face by MediaPipe, so they come from the closure before the camera applies.
    blink = np.clip(1 - ear / person.open_eye_ear, 0, 1)
    seen_mar = mar * camera.mar_scale
    seen_pitch = pitch + camera.pitch_offset_degrees
    signals = {
        "ear_left": ear_left * camera.ear_scale,
        "ear_right": ear_right * camera.ear_scale,
        "ear": ear * camera.ear_scale,
        "mar": seen_mar,
        "yaw": yaw + camera.yaw_offset_degrees,
        "pitch": seen_pitch,
        "roll": roll,
        "gaze_horizontal": gaze + camera.gaze_offset,
        "blink_left": blink,
        "blink_right": blink,
        "jaw_open": np.clip(seen_mar - 0.1, 0, 1),
        "look_down": np.clip(-seen_pitch / 40, 0, 1),
    }
    frame = pd.DataFrame(
        {
            "subject": subject,
            "session": f"{subject}_L{label}",
            "clip": CLIP_DRIVE,
            "label": label,
            "t": np.arange(n) / _FPS,
            **{name: np.where(face, values, np.nan) for name, values in signals.items()},
            "face": face.astype(float),
        }
    )
    return frame[["subject", "session", "clip", "label", "t", *FRAME_FEATURES]]


def simulate_sessions(
    subject_count: int = 30,
    minutes: int = 10,
    seed: int = 7,
    *,
    label_effect: bool = True,
) -> pd.DataFrame:
    """Frames for ``subject_count`` people, each with an alert, a low-vigilance and a drowsy session.

    Each session is its own recording with its own camera placement. With ``label_effect=False`` every session is
    an alert face whatever its label."""
    rng = np.random.default_rng(seed)
    sessions = []
    for index in range(subject_count):
        subject = f"S{index:03d}"
        person = PersonProfile.draw(rng)
        for label in ALERTNESS_LABELS:
            sessions.append(_simulate_session(rng, person, subject, label, minutes, label_effect=label_effect))
    return pd.concat(sessions, ignore_index=True)


# Acted events, as a team member performs them in a parked car (L11). The closed-eye EAR is a measured value,
# so it is set after the camera: with an open level near 0.31 it is about 80% closed.
ACTED_CLOSED_EAR = 0.06
ACTED_BLOCK_FRAMES = 30
ACTED_CLOSURE_FRAMES = 13  # closed for 13 of every 30 frames: 1.3 s microsleeps
ACTED_HEAD_DROP_DEGREES = 10
ACTED_YAWN_MAR = 0.75
ACTED_NOD_DEGREES = 20
ACTED_LOOK_AWAY_YAW_DEGREES = 40
TEAM_DRIVE_SCRIPT: tuple[tuple[float, float, str], ...] = (
    (150.0, 40.0, ACTED_DROWSINESS),
    (330.0, 10.0, ACTED_YAWN),
    (480.0, 5.0, ACTED_NOD),
    (600.0, 3.0, ACTED_EYES_OFF),
    (720.0, 40.0, ACTED_DROWSINESS),
)
TEAM_DRIVE_SECONDS = 900
TEAM_CALIBRATION_SECONDS = 180


def _rows(frames: pd.DataFrame, start_s: float, seconds: float) -> slice:
    first = int(start_s * _FPS)
    return slice(first, int((start_s + seconds) * _FPS) - 1)


def act_event(frames: pd.DataFrame, start_s: float, seconds: float, event: str) -> None:
    """Write one acted event into an alert recording, in place; ``frames`` has a 0..n-1 index and t from 0."""
    rows = _rows(frames, start_s, seconds)
    if event == ACTED_DROWSINESS:
        for block in range(rows.start, rows.stop + 1, ACTED_BLOCK_FRAMES):
            frames.loc[block : block + ACTED_CLOSURE_FRAMES - 1, "ear"] = ACTED_CLOSED_EAR
        frames.loc[rows, "pitch"] -= ACTED_HEAD_DROP_DEGREES
    elif event == ACTED_YAWN:
        frames.loc[rows, "mar"] = ACTED_YAWN_MAR
    elif event == ACTED_NOD:
        frames.loc[rows, "pitch"] -= ACTED_NOD_DEGREES
    elif event == ACTED_EYES_OFF:
        frames.loc[rows, "yaw"] += ACTED_LOOK_AWAY_YAW_DEGREES
    else:
        raise ValueError(f"unknown acted event {event!r}")


def simulate_team_session(
    subject: str,
    session_name: str,
    seed: int,
    *,
    script: tuple[tuple[float, float, str], ...] = TEAM_DRIVE_SCRIPT,
    calibration_seconds: int = TEAM_CALIBRATION_SECONDS,
    drive_seconds: int = TEAM_DRIVE_SECONDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A team session per RECORDING_PROTOCOL.md: a calibration clip, then a scripted drive, one camera placement.

    Both clips come from one alert recording, so they share the person and the camera draw, as a phone left in
    its mount does. Calibration frames are label 0; drive frames are ``UNLABELED`` until the script labels them.
    """
    rng = np.random.default_rng(seed)
    person = PersonProfile.draw(rng)
    total_minutes = -(-(calibration_seconds + drive_seconds) // 60)
    frames = _simulate_session(rng, person, subject, 0, total_minutes)
    frames = frames.loc[frames["t"] < calibration_seconds + drive_seconds].copy()
    is_drive = frames["t"] >= calibration_seconds
    frames["session"] = f"{subject}:{session_name}"
    frames["clip"] = np.where(is_drive, CLIP_DRIVE, CLIP_CALIBRATION)
    frames["label"] = np.where(is_drive, UNLABELED, 0)
    calibration = frames.loc[~is_drive].reset_index(drop=True)
    drive = frames.loc[is_drive].reset_index(drop=True)
    drive["t"] = np.round(drive["t"] - calibration_seconds, 6)
    for start_s, seconds, event in script:
        act_event(drive, start_s, seconds, event)
    events = pd.DataFrame(
        [{"start_s": start_s, "end_s": start_s + seconds, "event": event} for start_s, seconds, event in script],
        columns=list(SCRIPT_COLUMNS),
    )
    return pd.concat([calibration, drive], ignore_index=True), events


# ---- scripted sessions for the indicator submodels -----------------------------------------------------------
# Events the indicator tests and Studio's synthetic driver act out, each with the reading it should produce.
# Times are drive seconds; the first ``baseline_seconds`` (120 s) are calibration, so no event starts before it.
CLOSURE_5S = "closure_5s"
GAZE_ERRATIC_10S = "gaze_erratic_10s"
HEAD_DROP_NO_RESPONSE = "head_drop_no_response"
NODDING_RUN = "nodding_run"
SUNGLASSES_ON = "sunglasses_on"
MIC_ROAD_NOISE = "mic_road_noise"
MIC_SILENCE_PARKED = "mic_silence_parked"
INDICATOR_EVENTS: dict[str, str] = {
    CLOSURE_5S: "eyes closure_critical; drowsy alert, the driver then responds",
    GAZE_ERRATIC_10S: "ocular erratic_10s; one family alone, so never possible_impairment",
    HEAD_DROP_NO_RESPONSE: "possible_unresponsiveness high",
    NODDING_RUN: "head nodding; drowsy",
    SUNGLASSES_ON: "eyes unavailable; monitoring_degraded with sunglasses_or_eye_occlusion",
    MIC_ROAD_NOISE: "breathing unavailable: road_noise",
    MIC_SILENCE_PARKED: "breathing not_detected_quiet_cabin once parked and quiet for 20 s",
    ACTED_DROWSINESS: "drowsiness model window over threshold",
}
INDICATOR_COLUMNS: tuple[str, ...] = ("eye_visibility", "mouth_visibility", "shoulder_y")
SUNGLASSES_EYE_VISIBILITY = 0.1
HEAD_DROP_DEGREES = 35.0
ERRATIC_GAZE_AMPLITUDE = 0.25
ERRATIC_GAZE_STEP_FRAMES = 3
NOD_DIP_DEGREES = 20.0
NOD_PERIOD_SECONDS = 4.0
NOD_DIP_SECONDS = 1.0
INDICATOR_DEMO_SCRIPT: tuple[tuple[float, float, str], ...] = (
    (150.0, 5.0, CLOSURE_5S),
    (200.0, 12.0, GAZE_ERRATIC_10S),
    (260.0, 40.0, NODDING_RUN),
    (340.0, 12.0, HEAD_DROP_NO_RESPONSE),
    (400.0, 60.0, SUNGLASSES_ON),
    (430.0, 12.0, HEAD_DROP_NO_RESPONSE),
    (500.0, 40.0, ACTED_DROWSINESS),
    (600.0, 5.0, CLOSURE_5S),
)


def _act_indicator_event(frames: pd.DataFrame, start_s: float, seconds: float, event: str) -> None:
    """Write one indicator event into ``frames`` in place; both eyes move together unless the event says not."""
    rows = _rows(frames, start_s, seconds)
    eye_columns = ["ear", "ear_left", "ear_right"]
    if event == CLOSURE_5S:
        frames.loc[rows, eye_columns] = ACTED_CLOSED_EAR
    elif event == GAZE_ERRATIC_10S:
        # Jumps right, back to centre, left, back to centre: scattered, but crossing the road ahead every 0.6 s,
        # so it is erratic movement, not one continuous look away.
        index = frames.loc[rows].index
        pattern = np.array([1.0, 0.0, -1.0, 0.0])[(np.arange(len(index)) // ERRATIC_GAZE_STEP_FRAMES) % 4]
        frames.loc[index, "gaze_horizontal"] = frames.loc[index, "gaze_horizontal"] + ERRATIC_GAZE_AMPLITUDE * pattern
    elif event == HEAD_DROP_NO_RESPONSE:
        frames.loc[rows, "pitch"] -= HEAD_DROP_DEGREES
        frames.loc[rows, eye_columns] = ACTED_CLOSED_EAR
    elif event == NODDING_RUN:
        t = frames.loc[rows, "t"] - start_s
        dipping = (t % NOD_PERIOD_SECONDS) < NOD_DIP_SECONDS
        frames.loc[t.index[dipping], "pitch"] -= NOD_DIP_DEGREES
    elif event == SUNGLASSES_ON:
        # Behind dark lenses MediaPipe still places eyelids, so EAR reads a plausible open value it cannot see.
        frames.loc[rows, "eye_visibility"] = SUNGLASSES_EYE_VISIBILITY
        open_level = float(np.nanmedian(frames["ear"].to_numpy(dtype=float)))
        frames.loc[rows, eye_columns] = open_level
    elif event in {MIC_ROAD_NOISE, MIC_SILENCE_PARKED}:
        return  # audio-only: see ``cabin_audio_chunks``
    elif event == ACTED_DROWSINESS:
        act_event(frames, start_s, seconds, event)
        segment = frames.loc[rows]
        closed = segment.index[segment["ear"] == ACTED_CLOSED_EAR]
        frames.loc[closed, ["ear_left", "ear_right"]] = ACTED_CLOSED_EAR
    else:
        raise ValueError(f"unknown indicator event {event!r}")


def scripted_synthetic_session(
    seed: int,
    *,
    script: tuple[tuple[float, float, str], ...] = INDICATOR_DEMO_SCRIPT,
    minutes: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """An alert recording with the scripted indicator events, and their ground truth.

    The frames carry the 13 frame features plus ``eye_visibility``, ``mouth_visibility`` (both 1.0 unless an
    event changes them) and ``shoulder_y`` (NaN: the shoulders are not in frame). The events table has
    ``start_s``, ``end_s``, ``event`` and ``expected``, the reading the event should produce. The replay gate keeps
    its own session in ``staging.replay`` so its numbers do not move.
    """
    frames = simulate_sessions(subject_count=1, minutes=minutes, seed=seed + 1)
    frames = frames.loc[frames["label"] == 0].reset_index(drop=True)
    frames["eye_visibility"] = 1.0
    frames["mouth_visibility"] = 1.0
    frames["shoulder_y"] = np.nan
    for start_s, seconds, event in script:
        _act_indicator_event(frames, start_s, seconds, event)
    events = pd.DataFrame(
        [
            {"start_s": start_s, "end_s": start_s + seconds, "event": event, "expected": INDICATOR_EVENTS[event]}
            for start_s, seconds, event in script
        ],
        columns=[*SCRIPT_COLUMNS, "expected"],
    )
    frames["session"] = "INDICATORS_1"
    frames["subject"] = "INDICATORS"
    return frames, events


# ---- synthetic cabin audio ---------------------------------------------------------------------------------------
AUDIO_RATE_HZ = 16000
AUDIO_CHUNK_SECONDS = 0.1
QUIET_FLOOR_RMS = 0.008
BREATH_RMS = 0.03
BREATH_HZ = 0.25
ROAD_NOISE_RMS = 0.12
GASP_RMS = 0.15
GASP_SECONDS = 0.4
GASP_TIMES_SECONDS: tuple[float, ...] = (24.0, 33.0, 45.0, 52.0)  # long, irregular gaps: 9 s, 12 s, 7 s


def cabin_audio_chunks(
    kind: str, seconds: float, seed: int = 0, *, start_s: float = 0.0
) -> list[tuple[float, np.ndarray]]:
    """100 ms float32 PCM chunks ``(end_t, samples)`` of a synthetic cabin. Generated in memory only.

    ``kind`` is ``quiet_breathing`` (a quiet cabin with breath sounds every 4 s), ``mic_silence_parked`` (a quiet
    cabin with no breath sounds), ``mic_road_noise`` (broadband road noise), ``music`` (a loud chord), ``speech``
    (a voiced, pitch-varying tone), ``clipping`` (a saturated tone) or ``gasps`` (a quiet cabin with isolated
    gasp-like bursts at long, irregular intervals).
    """
    rng = np.random.default_rng(seed)
    chunk = int(AUDIO_RATE_HZ * AUDIO_CHUNK_SECONDS)
    count = round(seconds / AUDIO_CHUNK_SECONDS)
    chunks: list[tuple[float, np.ndarray]] = []
    for index in range(count):
        t0 = start_s + index * AUDIO_CHUNK_SECONDS
        t = t0 + np.arange(chunk) / AUDIO_RATE_HZ
        noise = rng.normal(0.0, 1.0, chunk)
        if kind == "mic_road_noise":
            pcm = ROAD_NOISE_RMS * noise
        elif kind == "music":
            pcm = 0.25 * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 277 * t) + np.sin(2 * np.pi * 330 * t))
        elif kind == "speech":
            pitch = 220 + 60 * np.sin(2 * np.pi * 3 * t)
            pcm = 0.2 * np.sin(2 * np.pi * np.cumsum(pitch) / AUDIO_RATE_HZ) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))
        elif kind == "clipping":
            pcm = np.clip(1.4 * np.sin(2 * np.pi * 200 * t), -1.0, 1.0)
        else:
            pcm = QUIET_FLOOR_RMS * noise
            local = t0 - start_s
            if kind == "quiet_breathing":
                envelope = max(0.0, float(np.sin(2 * np.pi * BREATH_HZ * local))) ** 4
                pcm = pcm + BREATH_RMS * envelope * noise
            elif kind == "gasps" and any(gasp <= local < gasp + GASP_SECONDS for gasp in GASP_TIMES_SECONDS):
                pcm = GASP_RMS * noise
            elif kind not in {"mic_silence_parked", "gasps"}:
                raise ValueError(f"unknown cabin audio kind {kind!r}")
        chunks.append((round(t0 + AUDIO_CHUNK_SECONDS, 6), pcm.astype(np.float32)))
    return chunks
