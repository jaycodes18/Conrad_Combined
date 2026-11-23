"""Wave A Studio behavior: observation freshness, research audio gates, and scenario reporting."""

from __future__ import annotations

import numpy as np
import pytest

from driven_ml.errors import ComplianceError, DataError
from driven_ml.studio.audio_health import AudioFault, AudioHealthMonitor, BreathingAudioState
from driven_ml.studio.catalog import CARDS, Approval, ApprovalStore
from driven_ml.studio.conditions import Sample
from driven_ml.studio.microphone import ResearchMicrophone
from driven_ml.studio.observations import (
    ObservationBus,
    SensorObservation,
    SensorStatus,
    SourceKind,
)
from driven_ml.studio.research_audio import research_audio_allowed
from driven_ml.studio.scenarios import run_audio_scenarios, run_vision_condition_scenarios


def observation(
    name: str,
    t_seconds: float,
    value: float | None,
    *,
    valid: bool = True,
    status: SensorStatus = SensorStatus.AVAILABLE,
    detail: str = "",
) -> SensorObservation:
    return SensorObservation(
        name=name,
        t_seconds=t_seconds,
        value=value,
        confidence=0.9 if valid else 0.0,
        status=status,
        valid=valid,
        source_kind=SourceKind.SIMULATED,
        detail=detail,
    )


def test_observation_bus_expires_values_without_reusing_them_as_live() -> None:
    bus = ObservationBus()
    bus.publish(observation("gaze_duration", 1.0, 4.0))

    expired = bus.expire(4.0, max_age_seconds=2.0)

    assert len(expired) == 1
    assert expired[0].status is SensorStatus.UNAVAILABLE
    assert expired[0].value is None
    assert expired[0].source_kind is SourceKind.SIMULATED
    assert bus.latest("gaze_duration") == expired[0]


def test_observation_bus_rejects_non_monotonic_timestamps() -> None:
    bus = ObservationBus()
    bus.publish(observation("eye_closure", 2.0, 1.5))

    with pytest.raises(ValueError, match="moved backwards"):
        bus.publish(observation("eye_closure", 1.9, 2.0))


def test_audio_gate_requires_both_explicit_flags_and_recording_disabled() -> None:
    assert not research_audio_allowed(None)
    assert not research_audio_allowed({"studio_research_audio": True, "legal_opinion_studio_audio": True})
    assert not research_audio_allowed(
        {
            "record_audio": True,
            "studio_research_audio": True,
            "legal_opinion_studio_audio": True,
        }
    )
    assert research_audio_allowed(
        {
            "record_audio": False,
            "studio_research_audio": True,
            "legal_opinion_studio_audio": True,
        }
    )


def test_microphone_start_is_blocked_by_default_before_device_access() -> None:
    microphone = ResearchMicrophone()

    with pytest.raises(ComplianceError, match="legal_opinion_studio_audio is false"):
        microphone.start()

    assert not microphone.running


def test_audio_monitor_marks_invalid_chunks_and_preserves_simulation_provenance() -> None:
    monitor = AudioHealthMonitor()
    invalid = monitor.push(
        1.0,
        np.array([np.nan], dtype=np.float32),
        source_kind=SourceKind.SIMULATED,
    )
    stale_monitor = AudioHealthMonitor()
    stale_monitor.push(
        1.0,
        np.full(1600, 0.1, dtype=np.float32),
        source_kind=SourceKind.SIMULATED,
    )
    stale = stale_monitor.check_stale(2.1)

    assert invalid.state is BreathingAudioState.INTERFERENCE
    assert invalid.source_kind is SourceKind.SIMULATED
    assert stale is not None
    assert stale.fault is AudioFault.FROZEN
    assert stale.source_kind is SourceKind.SIMULATED


def test_audio_monitor_rejects_non_monotonic_chunks() -> None:
    monitor = AudioHealthMonitor()
    monitor.push(1.0, np.zeros(1600, dtype=np.float32))

    with pytest.raises(ValueError, match="increase monotonically"):
        monitor.push(1.0, np.zeros(1600, dtype=np.float32))


def test_scenario_results_separate_not_run_from_pass_and_fail() -> None:
    results = run_audio_scenarios()
    by_id = {result.test_id: result for result in results}

    assert by_id["microphone_disconnection"].passed is True
    assert "camera_disconnection" not in by_id
    assert by_id["bluetooth_microphone_switch"].to_json()["status"] == "not_run"
    assert "frozen_stream" in by_id
    rate = by_id["breathing_candidate_rate"]
    assert rate.passed is True
    assert "bpm=15.0" in rate.actual_behavior


def test_vision_hardware_scenarios_are_not_reported_as_passed() -> None:
    class StableLandmarker:
        def detect(self, _image: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
            return {"ear": 0.3, "yaw": 0.0}, points

    image = np.full((240, 240, 3), 127, dtype=np.uint8)
    points = np.column_stack([np.linspace(60, 180, 478), np.linspace(60, 200, 478)])

    results = run_vision_condition_scenarios([Sample(image, eyes_closed=False)], StableLandmarker())  # type: ignore[arg-type]
    by_id = {result.test_id: result for result in results}

    assert by_id["camera_disconnection"].passed is None
    assert by_id["driver_out_of_frame"].passed is None


def test_dataset_cards_recommend_research_without_granting_commercial_approval(tmp_path) -> None:
    store = ApprovalStore(tmp_path / "approvals.json")
    uta = next(item for item in store.table() if item["key"] == "uta_rldd")

    assert "Research benchmark" in uta["recommended_role"]
    assert uta["commercial_use"] == "unknown"
    assert not store.ingest_allowed("uta_rldd")
    assert CARDS["state_farm"].never_use
    with pytest.raises(DataError, match="never_use"):
        store.review("state_farm", Approval.APPROVED_RESEARCH, "Reviewer", "not permitted")
