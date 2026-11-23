"""Phase 4 robustness: eye-feature dropout, the condition suite, scripted ground truth and audio fault cases."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from conftest import make_frame, times

from driven_ml.features.synthetic import (
    INDICATOR_DEMO_SCRIPT,
    INDICATOR_EVENTS,
    cabin_audio_chunks,
    scripted_synthetic_session,
)
from driven_ml.indicators import BreathingIndicator, CameraHealth, IndicatorName
from driven_ml.modeling.candidates import EstimatorKind
from driven_ml.modeling.training import EYE_INPUTS, REPORT_COLUMNS, EyeMask, out_of_fold_logits
from driven_ml.product import load_product
from driven_ml.risk import ContactBook, DriverRiskSignal, EscalationConfig, RiskSession
from driven_ml.spec import EYE_DERIVED_FEATURES, EYE_MASK_TRAINING_FRACTION, INPUTS_WITH_BASELINE
from driven_ml.studio.audio_health import AudioFault, AudioHealthMonitor
from driven_ml.studio.conditions import FEATURE_CONDITIONS, FeatureCondition
from driven_ml.studio.observations import SourceKind
from driven_ml.studio.scenarios import ScriptedResult, SessionTrace, evaluate_trace, run_scripted_session

Traces = dict[str, tuple[FeatureCondition, SessionTrace]]

CLEAR = CameraHealth(mean_luma=0.5, face_luma=0.5, sharpness=100.0)


# ---- 1. feature dropout ---------------------------------------------------------------------------------------
def test_eye_mask_replaces_only_eye_inputs_with_the_development_medians() -> None:
    inputs = list(INPUTS_WITH_BASELINE)
    assert {*EYE_DERIVED_FEATURES, *(f"delta_{name}" for name in EYE_DERIVED_FEATURES)} == EYE_INPUTS
    medians = pd.Series({name: float(index) for index, name in enumerate(inputs)})
    mask = EyeMask.for_inputs(inputs, medians)
    features = np.full((4, len(inputs)), -1.0)
    masked = mask.apply(features, np.array([True, False, True, False]))
    for column, name in enumerate(inputs):
        expected = medians[name] if name in EYE_INPUTS else -1.0
        assert masked[0, column] == expected
        assert masked[1, column] == -1.0
    assert mask.fraction == EYE_MASK_TRAINING_FRACTION == 0.3
    sentinel = np.full((10000, len(inputs)), -1.0)
    augmented = mask.augment(sentinel, np.random.default_rng(0))
    masked_rows = (augmented[:, list(mask.columns)] != -1.0).all(axis=1)
    assert masked_rows.mean() == pytest.approx(0.3, abs=0.02)
    assert (augmented[:, [i for i in range(len(inputs)) if i not in mask.columns]] == -1.0).all()


def test_validation_reports_full_and_eyes_masked_metrics() -> None:
    assert {"auc_eyes_masked", "balanced_accuracy_eyes_masked"} <= set(REPORT_COLUMNS)
    rng = np.random.default_rng(1)
    inputs = list(INPUTS_WITH_BASELINE)
    rows, labels = 400, rng.integers(0, 2, 400)
    features = rng.normal(0.0, 1.0, (rows, len(inputs)))
    perclos = inputs.index("perclos")
    features[:, perclos] += 3.0 * labels  # the only informative input is eye-derived
    mask = EyeMask.for_inputs(inputs, pd.Series(dict.fromkeys(inputs, 0.0)))
    groups = np.repeat(np.arange(20), rows // 20)
    full, masked = out_of_fold_logits(features, labels, groups, EstimatorKind.LOGISTIC_REGRESSION, 7, eye_mask=mask)
    full_separation = np.mean(full[labels == 1]) - np.mean(full[labels == 0])
    masked_separation = np.mean(masked[labels == 1]) - np.mean(masked[labels == 0])
    print(f"logit separation: full {full_separation:.2f}, eyes masked {masked_separation:.2f}")
    assert full_separation > 1.0
    assert abs(masked_separation) < 0.25 * full_separation  # with the eyes hidden, this model has nothing left


# ---- 2. the condition suite on the scripted session -------------------------------------------------------------
@pytest.fixture(scope="module")
def traces() -> Traces:
    return {condition.key: (condition, run_scripted_session(condition)) for condition in FEATURE_CONDITIONS}


def rows_of(traces: Traces, key: str) -> dict[str, ScriptedResult]:
    condition, trace = traces[key]
    return {row.event: row for row in evaluate_trace(trace, condition)}


def table(rows: dict[str, ScriptedResult]) -> str:
    return "\n".join(
        f"  {row.event:22s} detected={row.detected} latency={row.latency_seconds} "
        f"false_alarms={row.false_alarms} {row.verdict} {row.detail}"
        for row in rows.values()
    )


CONDITION_KEYS = [condition.key for condition in FEATURE_CONDITIONS]


def test_the_suite_covers_every_required_condition() -> None:
    assert set(CONDITION_KEYS) == {"sunglasses", "ir", "low_light", "backlit", "glare", "blur", "fisheye", "hand"}


@pytest.mark.parametrize("key", CONDITION_KEYS)
def test_no_false_unresponsiveness_or_impairment_in_normal_driving(traces: Traces, key: str) -> None:
    rows = rows_of(traces, key)
    normal = rows["normal_driving"]
    assert normal.false_alarms == 0, f"{key}:\n{table(rows)}"
    assert all(row.false_alarms in (0, None) for row in rows.values()), f"{key}:\n{table(rows)}"


@pytest.mark.parametrize("key", CONDITION_KEYS)
def test_monitoring_degraded_names_the_condition(traces: Traces, key: str) -> None:
    rows = rows_of(traces, key)
    row = rows["monitoring_degraded"]
    assert row.verdict == "pass", f"{key}: {row.detail}\n{table(rows)}"


@pytest.mark.parametrize("key", CONDITION_KEYS)
def test_closure_caught_when_eyes_visible_and_head_drop_caught_when_not(traces: Traces, key: str) -> None:
    condition, _trace = traces[key]
    rows = rows_of(traces, key)
    closure, head = rows["closure_5s"], rows["head_drop_no_response"]
    if condition.eyes_visible:
        assert closure.verdict == "pass", f"{key}: closure missed\n{table(rows)}"
        assert closure.latency_seconds is not None
        assert closure.latency_seconds <= 5.0
    else:
        assert closure.verdict == "pass", f"{key}: closure read through hidden eyes\n{table(rows)}"
    assert head.verdict == "pass", f"{key}: head drop with no response missed\n{table(rows)}"
    assert all(row.verdict == "pass" for row in rows.values()), f"{key}:\n{table(rows)}"


# ---- 3. scripted sessions carry ground truth ------------------------------------------------------------------
def test_scripted_events_have_ground_truth_and_are_all_caught_on_a_clean_camera() -> None:
    frames, events = scripted_synthetic_session(20)
    assert {"eye_visibility", "mouth_visibility", "shoulder_y"} <= set(frames.columns)
    assert set(events["event"]) <= set(INDICATOR_EVENTS)
    assert all(events["expected"].str.len() > 0)
    assert {event for _start, _seconds, event in INDICATOR_DEMO_SCRIPT} >= {
        "closure_5s",
        "gaze_erratic_10s",
        "head_drop_no_response",
        "nodding_run",
        "sunglasses_on",
    }
    rows = {row.event: row for row in evaluate_trace(run_scripted_session(None), None)}
    assert {row.verdict for row in rows.values()} == {"pass"}, table(rows)
    for event in ("closure_5s", "head_drop_no_response", "gaze_erratic_10s", "nodding_run", "mic_silence_parked"):
        assert rows[event].latency_seconds is not None, table(rows)


# ---- 4. audio faults never contribute to a condition --------------------------------------------------------------
def cabin_session(
    kind: str, *, driving: bool, seconds: float = 30.0, hint: AudioFault | None = None, start: float = 10.0
) -> list[DriverRiskSignal]:
    session = RiskSession(
        session_id="audio",
        driver_ref="drv_audio",
        escalation=EscalationConfig.from_product(load_product()),
        contacts=ContactBook(),
        audio_allowed=True,  # synthetic PCM in memory; no microphone is opened
    )
    epoch = datetime(2026, 1, 1, tzinfo=UTC)
    for t in times(0.0, start):
        session.push_frame(make_frame(t), CLEAR, calibrating=True, driving=driving, emitted_at=epoch)
    monitor = AudioHealthMonitor()
    signals: list[DriverRiskSignal] = []
    for chunk_t, pcm in cabin_audio_chunks(kind, seconds, seed=3, start_s=start):
        health = monitor.push(chunk_t, pcm, source_kind=SourceKind.SIMULATED, hint=hint)
        session.push_audio(chunk_t, pcm, health, driving=driving)
        frame_t = round(chunk_t - 0.1, 6)
        signals.append(
            session.push_frame(
                make_frame(frame_t),
                CLEAR,
                calibrating=False,
                driving=driving,
                emitted_at=epoch + timedelta(seconds=frame_t),
            )
        )
    return signals


@pytest.mark.parametrize(
    ("kind", "hint", "reason"),
    [
        ("mic_road_noise", None, "road_noise"),
        ("music", AudioFault.MUSIC, "music"),
        ("speech", None, "speech"),
        ("clipping", None, "mic_fault"),
    ],
)
def test_noisy_or_faulty_audio_is_unavailable_and_raises_nothing(
    kind: str, hint: AudioFault | None, reason: str
) -> None:
    for driving in (True, False):
        signals = cabin_session(kind, driving=driving, hint=hint)
        breathing = [signal.indicators[IndicatorName.BREATHING] for signal in signals[100:]]
        assert {snapshot.unavailable_reason for snapshot in breathing} == {reason}, kind
        assert {signal.condition.value for signal in signals} == {"none"}, kind
        assert all(signal.indicators[IndicatorName.BREATHING].severity == 0.0 for signal in signals)


def test_a_disconnected_microphone_is_a_mic_fault_and_raises_nothing() -> None:
    signals = cabin_session("quiet_breathing", driving=False, seconds=5.0)
    assert signals[-1].indicators[IndicatorName.BREATHING].state == "unavailable"  # under 3 breath intervals yet
    session_monitor = AudioHealthMonitor()
    health = session_monitor.disconnected(20.0, source_kind=SourceKind.SIMULATED)
    breathing = BreathingIndicator(audio_allowed=True)
    reading = breathing.push_audio(20.0, np.zeros(0, dtype=np.float32), health, driving=False)
    assert reading.unavailable_reason == "mic_fault"
    assert reading.severity == 0.0


def test_gasps_and_silence_in_a_parked_cabin_are_flagged_but_trigger_nothing_alone() -> None:
    gasps = cabin_session("gasps", driving=False, seconds=45.0)
    states = [signal.indicators[IndicatorName.BREATHING].state for signal in gasps]
    assert "agonal_pattern_suspected" in states
    assert {signal.condition.value for signal in gasps} == {"none"}
    silence = cabin_session("mic_silence_parked", driving=False, seconds=30.0)
    states = [signal.indicators[IndicatorName.BREATHING].state for signal in silence]
    assert states[-1] == "not_detected_quiet_cabin"
    assert {signal.condition.value for signal in silence} == {"none"}
    assert {signal.escalation.tier for signal in [*gasps, *silence]} == {0}
