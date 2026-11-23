"""Abnormal-scenario lab runners against real vision and audio pipelines, and the scripted condition suite.

Results are labeled simulated_transform, simulated_audio, or hardware. OpenCV
condition transforms are not cabin recordings.

``run_scripted_session`` replays a scripted synthetic session (``features.synthetic``) under one of
``conditions.FEATURE_CONDITIONS`` through the whole indicator, fusion and escalation stack, with synthetic cabin
audio generated in memory (no microphone is opened, so the research-audio gate does not apply).
``evaluate_trace`` turns the trace into one row per scripted event: detected, latency, false alarms, verdict.
Studio's Test tab and ``tests/test_robustness.py`` both use them. Rows that need hardware are ``NOT RUN``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

import numpy as np

from ..features.synthetic import (
    CLOSURE_5S,
    GAZE_ERRATIC_10S,
    HEAD_DROP_NO_RESPONSE,
    MIC_ROAD_NOISE,
    MIC_SILENCE_PARKED,
    NODDING_RUN,
    cabin_audio_chunks,
    scripted_synthetic_session,
)
from ..indicators import CameraHealth, FrameSignals, IndicatorName
from ..indicators.breathing import BreathingAnalyzer, BreathingEstimate
from ..product import load_product
from ..risk import ContactBook, EscalationConfig, RiskSession
from .audio_health import AudioFault, AudioHealthMonitor, BreathingAudioState
from .conditions import (
    FAIL_DETECTION,
    FAIL_EYE_VISIBILITY,
    FeatureCondition,
    ImageLandmarker,
    Sample,
    apply_feature_condition,
    assess,
)
from .observations import SourceKind


class InputKind(StrEnum):
    SIMULATED_TRANSFORM = "simulated_transform"
    SIMULATED_AUDIO = "simulated_audio"
    HARDWARE = "hardware"
    PRERECORDED = "prerecorded"


@dataclass(frozen=True)
class ScenarioResult:
    test_id: str
    input_source: str
    scenario_type: str
    expected_behavior: str
    actual_behavior: str
    passed: bool | None
    confidence: float | None
    latency_ms: float
    failure_reason: str
    input_kind: InputKind

    def to_json(self) -> dict[str, Any]:
        return {
            "test_id": self.test_id,
            "input_source": self.input_source,
            "scenario_type": self.scenario_type,
            "expected_behavior": self.expected_behavior,
            "actual_behavior": self.actual_behavior,
            "passed": self.passed,
            "status": "not_run" if self.passed is None else "passed" if self.passed else "failed",
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
            "failure_reason": self.failure_reason,
            "input_kind": self.input_kind.value,
        }


def _result(
    test_id: str,
    scenario_type: str,
    expected: str,
    actual: str,
    passed: bool | None,
    *,
    confidence: float | None,
    latency_ms: float,
    input_kind: InputKind,
    input_source: str,
    failure_reason: str = "",
) -> ScenarioResult:
    reported_failure = "" if passed is True else failure_reason or (actual if passed is False else "")
    return ScenarioResult(
        test_id=test_id,
        input_source=input_source,
        scenario_type=scenario_type,
        expected_behavior=expected,
        actual_behavior=actual,
        passed=passed,
        confidence=confidence,
        latency_ms=latency_ms,
        failure_reason=reported_failure,
        input_kind=input_kind,
    )


def _format_measurement(value: float) -> str:
    return "unavailable" if not np.isfinite(value) else f"{value:.2f}"


def run_vision_condition_scenarios(
    samples: list[Sample],
    landmarker: ImageLandmarker,
) -> list[ScenarioResult]:
    """Run condition transforms on captured camera frames and test quality-gating invariants."""
    started = time.perf_counter()
    rows = assess(samples, landmarker)
    latency_ms = (time.perf_counter() - started) * 1000.0
    by_key = {row.key: row for row in rows}
    specs: tuple[tuple[str, str, str], ...] = (
        (
            "sunglasses",
            "sunglasses",
            "Preserve face tracking; withhold eye-derived measures when eyes are not visible.",
        ),
        (
            "glare_on_glasses",
            "glare",
            "Report measured eye visibility and withhold closure metrics below the visibility threshold.",
        ),
        ("night", "night", "Report measured face/eye availability and reject unobservable eye measurements."),
        ("backlighting", "backlit", "Report measured tracking quality under a backlit image transform."),
        ("motion_blur", "blur", "Report measured tracking quality under motion blur."),
        ("lower_face_obstruction", "hand", "Report measured face and eye availability under partial obstruction."),
        ("low_resolution", "low_res", "Report measured tracking quality after downsampling."),
    )
    results: list[ScenarioResult] = []
    for test_id, key, expected in specs:
        row = by_key.get(key)
        if row is None:
            results.append(
                _result(
                    test_id,
                    test_id,
                    expected,
                    "condition transform is unavailable",
                    None,
                    confidence=None,
                    latency_ms=latency_ms,
                    input_kind=InputKind.SIMULATED_TRANSFORM,
                    input_source="conditions.assess",
                    failure_reason="no matching condition transform",
                )
            )
            continue
        raw = row.as_captured
        below_face_gate = raw.detection < FAIL_DETECTION
        eyes_hidden = raw.eye_visibility < FAIL_EYE_VISIBILITY
        eye_metrics_withheld = (
            not np.isfinite(raw.ear_open)
            and not np.isfinite(raw.ear_closed)
            and not np.isfinite(raw.closure_separation)
        )
        passed = (not below_face_gate or raw.verdict == "fails") and (
            not eyes_hidden or (raw.verdict == "fails" and eye_metrics_withheld)
        )
        actual = (
            f"face_detection={raw.detection:.2f}; relative_eye_visibility={raw.eye_visibility:.2f}; "
            f"closure_separation={_format_measurement(raw.closure_separation)}; "
            f"verdict={raw.verdict}"
        )
        results.append(
            _result(
                test_id,
                test_id,
                expected,
                actual,
                passed,
                confidence=None,
                latency_ms=latency_ms,
                input_kind=InputKind.SIMULATED_TRANSFORM,
                input_source=row.name,
            )
        )
    for name in (
        "prescription_glasses",
        "diverse_skin_tone_groups",
        "camera_disconnection",
        "frozen_video",
        "driver_out_of_frame",
    ):
        results.append(
            _result(
                name,
                name,
                "requires the actual hardware condition or consented representative data",
                "not executed by the available image transforms",
                None,
                confidence=None,
                latency_ms=0.0,
                input_kind=InputKind.SIMULATED_TRANSFORM,
                input_source="not_available",
                failure_reason="no valid input or transform is available for this scenario",
            )
        )
    return results


def run_audio_scenarios() -> list[ScenarioResult]:
    """Run isolated synthetic PCM fixtures through the production analysis classes."""
    rate = 16000
    results: list[ScenarioResult] = []

    def tone(seconds: float, hz: float, amplitude: float) -> np.ndarray:
        samples = int(seconds * rate)
        t = np.arange(samples, dtype=np.float64) / rate
        return (amplitude * np.sin(2.0 * np.pi * hz * t)).astype(np.float32)

    fixtures = (
        ("microphone_disconnection", np.zeros(0, dtype=np.float32), None, None, "unavailable"),
        ("wind_interference_hint", tone(1.0, 80.0, 0.4), AudioFault.WIND, None, "environmental_interference"),
        ("music_interference_hint", tone(1.0, 440.0, 0.5), AudioFault.MUSIC, None, "environmental_interference"),
        ("passenger_speech_hint", tone(1.0, 180.0, 0.2), AudioFault.SPEECH, None, "environmental_interference"),
        ("clipping", tone(0.5, 200.0, 1.0), None, None, "environmental_interference"),
        ("low_gain", tone(1.0, 80.0, 0.002), None, None, "active_reduced"),
    )
    for index, (name, samples, hint, sample_rate, expected_state) in enumerate(fixtures):
        started = time.perf_counter()
        health_monitor = AudioHealthMonitor()
        analyzer = BreathingAnalyzer()
        health = health_monitor.push(
            float(index),
            samples,
            sample_rate_hz=sample_rate,
            source_kind=SourceKind.SIMULATED,
            hint=hint,
        )
        estimate = analyzer.push(float(index), samples, health)
        passed = health.state.value == expected_state
        results.append(
            _result(
                name,
                name,
                f"classify fixture as {expected_state}",
                f"fault={health.fault.value}; state={health.state.value}; bpm={estimate.breaths_per_minute!r}",
                passed,
                confidence=health.confidence,
                latency_ms=(time.perf_counter() - started) * 1000.0,
                input_kind=InputKind.SIMULATED_AUDIO,
                input_source="synthetic_pcm_fixture",
            )
        )

    rate_change_monitor = AudioHealthMonitor()
    rate_change_monitor.push(0.0, tone(0.1, 80.0, 0.1), sample_rate_hz=16000, source_kind=SourceKind.SIMULATED)
    rate_change = rate_change_monitor.push(
        0.1,
        tone(0.1, 80.0, 0.1),
        sample_rate_hz=8000,
        source_kind=SourceKind.SIMULATED,
    )
    results.append(
        _result(
            "sample_rate_change",
            "sample_rate_change",
            "Mark changed sample rate unavailable",
            f"fault={rate_change.fault.value}; state={rate_change.state.value}",
            rate_change.fault is AudioFault.SAMPLE_RATE_CHANGED,
            confidence=rate_change.confidence,
            latency_ms=0.0,
            input_kind=InputKind.SIMULATED_AUDIO,
            input_source="synthetic_pcm_fixture",
        )
    )

    stale_monitor = AudioHealthMonitor(stale_after_seconds=1.0)
    stale_monitor.push(0.0, tone(0.1, 80.0, 0.1), source_kind=SourceKind.SIMULATED)
    stale = stale_monitor.check_stale(1.1)
    results.append(
        _result(
            "frozen_stream",
            "frozen_stream",
            "Expire old readings when no chunks arrive",
            "no stale report" if stale is None else f"fault={stale.fault.value}; state={stale.state.value}",
            stale is not None and stale.state is BreathingAudioState.UNAVAILABLE,
            confidence=None if stale is None else stale.confidence,
            latency_ms=0.0,
            input_kind=InputKind.SIMULATED_AUDIO,
            input_source="clock_advance_fixture",
        )
    )

    permission_monitor = AudioHealthMonitor()
    permission = permission_monitor.permission_denied(0.0)
    results.append(
        _result(
            "microphone_permission_denied",
            "permission_denied",
            "Mark microphone unavailable when permission is denied",
            f"fault={permission.fault.value}; state={permission.state.value}",
            permission.state is BreathingAudioState.UNAVAILABLE,
            confidence=permission.confidence,
            latency_ms=0.0,
            input_kind=InputKind.SIMULATED_AUDIO,
            input_source="permission_state_fixture",
        )
    )

    silence_monitor, silence_analyzer = AudioHealthMonitor(), BreathingAnalyzer()
    silence = np.zeros(rate // 10, dtype=np.float32)
    silence_health = silence_monitor.push(0.1, silence, source_kind=SourceKind.SIMULATED)
    silence_estimate = silence_analyzer.push(0.1, silence, silence_health)
    results.append(
        _result(
            "silence_does_not_mean_zero_bpm",
            "silence_handling",
            "Withhold rate when the audio is silent",
            f"bpm={silence_estimate.breaths_per_minute!r}; state={silence_health.state.value}",
            silence_estimate.breaths_per_minute is None,
            confidence=silence_estimate.confidence,
            latency_ms=0.0,
            input_kind=InputKind.SIMULATED_AUDIO,
            input_source="digital_silence_fixture",
        )
    )

    breathing_monitor, breathing_analyzer = AudioHealthMonitor(), BreathingAnalyzer()
    duration_seconds = 24.0
    full_t = np.arange(int(duration_seconds * rate), dtype=np.float64) / rate
    carrier = (0.25 + 0.15 * np.sin(2.0 * np.pi * 0.25 * full_t)) * np.sin(2.0 * np.pi * 80.0 * full_t)
    rate_estimate: BreathingEstimate | None = None
    chunk_size = rate // 10
    for offset in range(0, len(carrier), chunk_size):
        chunk_end = min(offset + chunk_size, len(carrier))
        end_t = chunk_end / rate
        chunk = carrier[offset:chunk_end].astype(np.float32)
        health = breathing_monitor.push(end_t, chunk, source_kind=SourceKind.SIMULATED)
        rate_estimate = breathing_analyzer.push(end_t, chunk, health)
    if rate_estimate is None:
        raise RuntimeError("breathing fixture produced no analysis result")
    rate_ok = rate_estimate.breaths_per_minute is not None and 10.0 <= rate_estimate.breaths_per_minute <= 20.0
    results.append(
        _result(
            "breathing_candidate_rate",
            "breathing_candidate_rate",
            "Estimate a synthetic 0.25 Hz amplitude pattern within 10-20 candidates/min",
            f"bpm={rate_estimate.breaths_per_minute!r}; events={len(rate_estimate.events)}; "
            f"state={rate_estimate.status.value}",
            rate_ok,
            confidence=rate_estimate.confidence,
            latency_ms=0.0,
            input_kind=InputKind.SIMULATED_AUDIO,
            input_source="amplitude_modulated_tone_fixture",
        )
    )
    for name in ("bluetooth_microphone_switch", "background_audio_interruption", "multiple_speakers"):
        results.append(
            _result(
                name,
                name,
                "requires representative hardware or consented evaluation data",
                "not executed by the synthetic audio pipeline",
                None,
                confidence=None,
                latency_ms=0.0,
                input_kind=InputKind.SIMULATED_AUDIO,
                input_source="not_available",
                failure_reason="requires real hardware or appropriately consented labeled data",
            )
        )
    return results


# ---- scripted condition suite ------------------------------------------------------------------------------
ROBUSTNESS_SCRIPT: tuple[tuple[float, float, str], ...] = (
    (140.0, 5.0, CLOSURE_5S),
    (170.0, 12.0, HEAD_DROP_NO_RESPONSE),
    (205.0, 12.0, GAZE_ERRATIC_10S),
    (235.0, 20.0, MIC_ROAD_NOISE),
    (260.0, 30.0, MIC_SILENCE_PARKED),
    (300.0, 40.0, NODDING_RUN),
)
SCRIPTED_MINUTES = 6
SCRIPTED_SEED = 20
SCRIPTED_CALIBRATION_SECONDS = 120.0
NORMAL_MARGIN_SECONDS = 8.0  # how long after an event before frames count as normal driving again
DETECTION_GRACE_SECONDS = 1.0
DEGRADED_SHARE = 0.9
CLEAR_CAMERA = CameraHealth(mean_luma=0.5, face_luma=0.5, sharpness=100.0)
HARDWARE_ROWS: tuple[tuple[str, str], ...] = (
    ("real_cabin_camera", "a phone or dashcam in a car, parked, with a consenting driver"),
    ("real_glasses_on_a_face", "sunglasses and clear glasses worn by real people across skin tones"),
    ("road_audio", "a microphone in a moving car, after the written legal review"),
)
ALARM_CONDITIONS = frozenset({"possible_unresponsiveness", "possible_impairment"})
ALERT_LEVELS = frozenset({"elevated", "high"})


@dataclass(frozen=True)
class TraceStep:
    t_seconds: float
    driving: bool
    condition: str
    level: str
    situations: tuple[str, ...]
    states: dict[str, str]
    breathing_reason: str | None


@dataclass(frozen=True)
class SessionTrace:
    condition_key: str
    steps: tuple[TraceStep, ...]
    script: tuple[tuple[float, float, str], ...]

    def between(self, start: float, end: float) -> list[TraceStep]:
        return [step for step in self.steps if start <= step.t_seconds <= end]

    def normal_driving(self) -> list[TraceStep]:
        """Frames after calibration that are not inside, or just after, any scripted event."""
        return [
            step
            for step in self.steps
            if step.t_seconds >= SCRIPTED_CALIBRATION_SECONDS + 5.0
            and not any(start - 1.0 <= step.t_seconds <= start + seconds + NORMAL_MARGIN_SECONDS
                        for start, seconds, _event in self.script)
        ]  # fmt: skip


def _audio_plan(script: tuple[tuple[float, float, str], ...], total_seconds: float) -> list[tuple[float, np.ndarray]]:
    """Cabin audio for the whole session: quiet breathing, with the scripted microphone events spliced in."""
    mic_kinds = {MIC_ROAD_NOISE, MIC_SILENCE_PARKED}
    mic_events = sorted(item for item in script if item[2] in mic_kinds)
    chunks: list[tuple[float, np.ndarray]] = []
    cursor = 0.0
    for start, seconds, event in [*mic_events, (total_seconds, 0.0, "")]:
        if start > cursor:
            chunks += cabin_audio_chunks("quiet_breathing", start - cursor, seed=int(cursor), start_s=cursor)
        if event:
            chunks += cabin_audio_chunks(event, seconds, seed=int(start), start_s=start)
        cursor = max(cursor, start + seconds)
    return chunks


def run_scripted_session(
    condition: FeatureCondition | None,
    *,
    script: tuple[tuple[float, float, str], ...] = ROBUSTNESS_SCRIPT,
    seed: int = SCRIPTED_SEED,
    minutes: int = SCRIPTED_MINUTES,
    escalation: EscalationConfig | None = None,
) -> SessionTrace:
    frames, _events = scripted_synthetic_session(seed, script=script, minutes=minutes)
    if condition is not None:
        frames = apply_feature_condition(frames, condition, seed)
    camera = CLEAR_CAMERA if condition is None else condition.camera
    session = RiskSession(
        session_id="scripted",
        driver_ref="drv_scripted",
        escalation=escalation or EscalationConfig.from_product(load_product()),
        contacts=ContactBook(),
        audio_allowed=True,  # synthetic PCM generated in memory; no microphone is opened
    )
    parked = [(start, start + seconds) for start, seconds, event in script if event == MIC_SILENCE_PARKED]
    audio = iter(_audio_plan(script, minutes * 60.0))
    monitor = AudioHealthMonitor()
    epoch = datetime(2026, 1, 1, tzinfo=UTC)
    steps: list[TraceStep] = []
    for record in frames.to_dict(orient="records"):
        row = {str(key): value for key, value in record.items()}
        t = float(row["t"])
        driving = not any(start <= t < end for start, end in parked)
        chunk = next(audio, None)
        if chunk is not None:
            health = monitor.push(chunk[0], chunk[1], source_kind=SourceKind.SIMULATED)
            session.push_audio(chunk[0], chunk[1], health, driving=driving)
        frame = FrameSignals.from_features(
            t, row, eye_visibility=float(row["eye_visibility"]), mouth_visibility=float(row["mouth_visibility"])
        )
        signal = session.push_frame(
            frame,
            camera,
            calibrating=t < SCRIPTED_CALIBRATION_SECONDS,
            driving=driving,
            emitted_at=epoch + timedelta(seconds=t),
        )
        breathing = signal.indicators[IndicatorName.BREATHING]
        steps.append(
            TraceStep(
                t_seconds=t,
                driving=driving,
                condition=signal.condition.value,
                level=signal.level.value,
                situations=signal.monitoring_degraded.situations,
                states={name.value: snapshot.state for name, snapshot in signal.indicators.items()},
                breathing_reason=breathing.unavailable_reason,
            )
        )
    return SessionTrace(
        condition_key="clean" if condition is None else condition.key, steps=tuple(steps), script=script
    )


@dataclass(frozen=True)
class ScriptedResult:
    condition: str
    event: str
    expected: str
    detected: bool | None
    latency_seconds: float | None
    false_alarms: int | None
    verdict: str
    detail: str

    def to_json(self) -> dict[str, Any]:
        return {
            "condition": self.condition,
            "event": self.event,
            "expected": self.expected,
            "detected": self.detected,
            "latency_seconds": self.latency_seconds,
            "false_alarms": self.false_alarms,
            "verdict": self.verdict,
            "detail": self.detail,
        }


def _first(steps: list[TraceStep], start: float, test: Any) -> float | None:
    """Seconds from ``start`` to the first step that passes ``test``, or None."""
    for step in steps:
        if test(step):
            return round(step.t_seconds - start, 2)
    return None


def _alarm_episodes(steps: list[TraceStep]) -> int:
    """Times the reported condition entered possible unresponsiveness or impairment at an alert level."""
    count, inside = 0, False
    for step in steps:
        alarm = step.condition in ALARM_CONDITIONS and step.level in ALERT_LEVELS
        count += alarm and not inside
        inside = alarm
    return count


@dataclass(frozen=True)
class _Check:
    """What one scripted event should produce, judged on the frames from its start to its end plus a grace."""

    expected: str
    detected: bool
    latency_seconds: float | None
    passed: bool
    detail: str = ""


def _closure(window: list[TraceStep], start: float, seconds: float, eyes_visible: bool) -> _Check:
    if not eyes_visible:
        withheld = not any(step.states["eyes"] in {"closure_warning", "closure_critical"} for step in window)
        return _Check("eye closure withheld: the eyes cannot be seen", withheld, None, withheld)
    latency = _first(window, start, lambda s: s.states["eyes"] == "closure_critical" and s.level in ALERT_LEVELS)
    return _Check("eyes closure_critical and a drowsy alert", latency is not None, latency, latency is not None)


def _head_drop(window: list[TraceStep], start: float, seconds: float, eyes_visible: bool) -> _Check:
    latency = _first(window, start, lambda s: s.condition == "possible_unresponsiveness" and s.level == "high")
    return _Check("possible_unresponsiveness high", latency is not None, latency, latency is not None)


def _gaze(window: list[TraceStep], start: float, seconds: float, eyes_visible: bool) -> _Check:
    impaired = any(step.condition == "possible_impairment" for step in window)
    if not eyes_visible:
        quiet = not any(step.states["ocular"].startswith("erratic") for step in window) and not impaired
        return _Check("ocular unavailable: the eyes cannot be seen", quiet, None, quiet)
    latency = _first(window, start, lambda s: s.states["ocular"] == "erratic_10s")
    ok = latency is not None and not impaired
    return _Check("ocular erratic_10s, never possible_impairment", latency is not None, latency, ok, "one family alone")


def _nodding(window: list[TraceStep], start: float, seconds: float, eyes_visible: bool) -> _Check:
    latency = _first(window, start, lambda s: s.states["head"] == "nodding" and s.condition == "drowsy")
    return _Check("head nodding and a drowsy alert", latency is not None, latency, latency is not None)


def _road_noise(window: list[TraceStep], start: float, seconds: float, eyes_visible: bool) -> _Check:
    tail = [step for step in window if start + seconds / 2 <= step.t_seconds <= start + seconds]
    heard = bool(tail) and all(step.breathing_reason == "road_noise" for step in tail)
    silent = all(step.condition == "none" for step in window)
    return _Check("breathing unavailable: road_noise, and no condition from it", heard, None, heard and silent)


def _parked_silence(window: list[TraceStep], start: float, seconds: float, eyes_visible: bool) -> _Check:
    latency = _first(window, start, lambda s: s.states["breathing"] == "not_detected_quiet_cabin")
    silent = all(step.condition not in ALARM_CONDITIONS for step in window)
    expected = "not_detected_quiet_cabin after 20 s parked, and no condition from it"
    return _Check(expected, latency is not None, latency, latency is not None and silent)


_CHECKS: dict[str, Any] = {
    CLOSURE_5S: _closure,
    HEAD_DROP_NO_RESPONSE: _head_drop,
    GAZE_ERRATIC_10S: _gaze,
    NODDING_RUN: _nodding,
    MIC_ROAD_NOISE: _road_noise,
    MIC_SILENCE_PARKED: _parked_silence,
}


def _scripted_result(condition: str, event: str, check: _Check, false_alarms: int | None) -> ScriptedResult:
    return ScriptedResult(
        condition=condition,
        event=event,
        expected=check.expected,
        detected=check.detected,
        latency_seconds=check.latency_seconds,
        false_alarms=false_alarms,
        verdict="pass" if check.passed else "fail",
        detail=check.detail,
    )


def evaluate_trace(trace: SessionTrace, condition: FeatureCondition | None) -> list[ScriptedResult]:
    eyes_visible = condition is None or condition.eyes_visible
    rows: list[ScriptedResult] = []
    for start, seconds, event in trace.script:
        window = trace.between(start, start + seconds + DETECTION_GRACE_SECONDS)
        check = _CHECKS[event](window, start, seconds, eyes_visible)
        # An unresponsiveness alarm is the point of the head-drop event; anywhere else it is a false alarm.
        unexpected = 0 if event == HEAD_DROP_NO_RESPONSE else _alarm_episodes(window)
        rows.append(_scripted_result(trace.condition_key, event, check, unexpected))
    normal = trace.normal_driving()
    alarms = _alarm_episodes(normal)
    expected = "no possible_unresponsiveness or possible_impairment"
    detail = f"{len(normal)} frames of normal driving"
    rows.append(
        _scripted_result(
            trace.condition_key, "normal_driving", _Check(expected, alarms > 0, None, alarms == 0, detail), alarms
        )
    )
    if condition is not None:
        monitored = [step for step in trace.steps if step.t_seconds >= SCRIPTED_CALIBRATION_SECONDS]
        share = sum(condition.situation.value in step.situations for step in monitored) / max(len(monitored), 1)
        named = share >= DEGRADED_SHARE
        check = _Check(f"{condition.situation.value} named", named, None, named, f"named on {share:.0%} of frames")
        rows.append(_scripted_result(trace.condition_key, "monitoring_degraded", check, None))
    return rows


def run_scripted_scenarios(
    conditions: list[FeatureCondition | None],
    escalation: EscalationConfig | None = None,
    *,
    events: list[str] | None = None,
) -> list[ScriptedResult]:
    """Each condition through the robustness script (or the chosen ``events`` of it), plus the hardware rows."""
    script = ROBUSTNESS_SCRIPT if events is None else tuple(item for item in ROBUSTNESS_SCRIPT if item[2] in events)
    if not script:
        raise ValueError("choose at least one scripted event")
    rows: list[ScriptedResult] = []
    for condition in conditions:
        rows += evaluate_trace(run_scripted_session(condition, script=script, escalation=escalation), condition)
    rows += [
        ScriptedResult("hardware", name, need, None, None, None, "NOT RUN", "requires " + need)
        for name, need in HARDWARE_ROWS
    ]
    return rows
