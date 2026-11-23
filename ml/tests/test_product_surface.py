"""Camera contract, temporal monitor, contacts, and dataset approval. No camera and no download."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from driven_ml.errors import ComplianceError, DataError
from driven_ml.product import load_product
from driven_ml.risk import (
    Condition,
    ConsentState,
    ContactBook,
    EscalationConfig,
    EscalationEngine,
    FusionResult,
    RiskLevel,
    TierEventKind,
)
from driven_ml.studio.cameras import (
    CameraKind,
    classify_source,
    frame_gap_dropped,
    normalize_rgb,
    registry_json,
    timestamp_advances,
)
from driven_ml.studio.catalog import Approval, ApprovalStore, CommercialUse, ingest_plan, kaggle_credential_status
from driven_ml.studio.monitor import MonitorConfig, MonitoringState, TemporalTracker
from driven_ml.studio.profiles import DriverBook
from driven_ml.studio.status import MATRIX, render_report


def test_research_gaze_fires_at_three_seconds_and_rearms() -> None:
    tracker = TemporalTracker(MonitorConfig.research_brief())
    events = []
    for index in range(31):
        snap = tracker.push(index / 10, face=1, ear=0.3, yaw=40, gaze_horizontal=0)
        events.extend(snap.events)
    gaze = [item for item in events if item.kind == "research_gaze"]
    assert len(gaze) == 1
    assert gaze[0].t_seconds == pytest.approx(3.0)
    assert gaze[0].duration_seconds == pytest.approx(3.0)
    for index in range(31, 50):
        tracker.push(index / 10, face=1, ear=0.3, yaw=0, gaze_horizontal=0)
    more = []
    for index in range(50, 81):
        snap = tracker.push(index / 10, face=1, ear=0.3, yaw=40, gaze_horizontal=0)
        more.extend(item.kind for item in snap.events)
    assert more.count("research_gaze") == 1


def test_closure_and_unavailable_are_not_impairment_when_the_camera_drops() -> None:
    tracker = TemporalTracker(MonitorConfig.research_brief())
    closed = []
    for index in range(16):
        snap = tracker.push(index / 10, face=1, ear=0.05, yaw=0, gaze_horizontal=0, open_ear=0.3)
        closed.extend(snap.events)
    assert [item.kind for item in closed] == ["research_closure"]
    assert closed[0].t_seconds == pytest.approx(1.5)
    lost = []
    for index in range(16, 40):
        snap = tracker.push(index / 10, face=0, ear=float("nan"), yaw=0, gaze_horizontal=0, camera_connected=False)
        lost.extend(snap.events)
    assert lost[0].kind == "monitoring_unavailable"
    assert lost[0].message.startswith("Camera")
    assert all(item.kind != "research_closure" for item in lost)
    assert tracker.state is MonitoringState.UNAVAILABLE
    resumed = []
    for index in range(40, 55):
        snap = tracker.push(index / 10, face=1, ear=0.3, yaw=0, gaze_horizontal=0)
        resumed.extend(item.kind for item in snap.events)
    assert "monitoring_resumed" in resumed


def test_non_monotonic_timestamp_is_a_camera_fault() -> None:
    tracker = TemporalTracker()
    tracker.push(1.0, face=1, ear=0.3, yaw=0, gaze_horizontal=0)
    snap = tracker.push(1.0, face=1, ear=0.05, yaw=40, gaze_horizontal=0.4)
    assert snap.events[0].kind == "camera_fault"
    assert snap.eyes_closed is False
    with pytest.raises(Exception, match="positive"):
        MonitorConfig(off_forward_gaze_seconds=0).validate()


def test_escalation_recommends_text_and_call_and_refuses_911(tmp_path: Path) -> None:
    book = ContactBook()
    book.add(
        "parent",
        "Parent",
        "+12145550100",
        consent=ConsentState.RECORDED,
        consent_date="2026-11-01",
        calls_consented=True,
    )
    config = EscalationConfig.from_product(load_product())
    engine = EscalationEngine(config, book)
    with pytest.raises(ComplianceError):
        engine.request_emergency_dispatch()
    with pytest.raises(ComplianceError):
        EscalationEngine(config, book, allow_911=True)
    alert = engine.step(_result(0.0, Condition.POSSIBLE_UNRESPONSIVENESS, RiskLevel.ELEVATED), "waiting")
    assert alert.in_car_alert
    text = engine.step(_result(5.0, Condition.POSSIBLE_UNRESPONSIVENESS, RiskLevel.HIGH), "no_response")
    assert text.notify_contacts
    assert text.share_location == "precise"
    call = [
        engine.step(_result(5.0 + index / 10, Condition.POSSIBLE_UNRESPONSIVENESS, RiskLevel.HIGH), "no_response")
        for index in range(1, 110)
    ]
    assert sum(decision.voice_call_recommended for decision in call) == 1
    assert {decision.emergency_services for decision in [alert, text, *call]} == {"never_automated"}
    assert all(event.to_json()["dispatched"] is False for event in engine.history)
    assert "5550100" not in book.to_json()[0]["phone_masked"]
    assert not (tmp_path / "sent.txt").exists()


def test_missing_consent_means_nothing_would_be_sent() -> None:
    book = ContactBook()
    book.add("parent", "Parent", "+12145550100")
    engine = EscalationEngine(EscalationConfig.from_product(load_product()), book)
    engine.step(_result(0.0, Condition.DROWSY, RiskLevel.ELEVATED), "waiting")
    skipped = engine.step(_result(5.0, Condition.DROWSY, RiskLevel.ELEVATED), "no_response")
    assert not skipped.notify_contacts
    assert engine.history[-1].kind is TierEventKind.CONTACT_SKIPPED


def _result(t: float, condition: Condition, level: RiskLevel) -> FusionResult:
    return FusionResult(
        t_seconds=t,
        condition=condition,
        level=level,
        score=0.9,
        confidence=0.8,
        reasons=(),
        scores={},
        monitoring_degraded=False,
        degraded_situations=(),
        readings={},
    )


def test_dataset_approval_never_downloads_and_blocks_never_use(tmp_path: Path) -> None:
    store = ApprovalStore(tmp_path / "approvals.json")
    with pytest.raises(DataError):
        store.review("state_farm", Approval.APPROVED_RESEARCH, "Ada", "no")
    with pytest.raises(DataError):
        store.review("uta_rldd", Approval.APPROVED_COMMERCIAL, "Ada", "license unclear")
    store.review("uta_rldd", Approval.APPROVED_RESEARCH, "Ada", "research only")
    plan = ingest_plan("uta_rldd", store)
    assert plan["allowed"] is True
    assert plan["executed"] is False
    blocked = ingest_plan("state_farm", store)
    assert blocked["allowed"] is False
    assert blocked["executed"] is False
    saved = json.loads((tmp_path / "approvals.json").read_text(encoding="utf-8"))
    assert saved[0]["reviewer"] == "Ada"
    status = kaggle_credential_status()
    assert status["values_exposed"] is False
    assert "key" not in status or status["key_env"] == "KAGGLE_KEY"
    from driven_ml.studio.catalog import CARDS

    assert CARDS["state_farm"].commercial_use is CommercialUse.FORBIDDEN


def test_camera_contract_and_registry_do_not_claim_a_bench_test() -> None:
    assert classify_source("0") is CameraKind.PHONE
    assert classify_source("1") is CameraKind.USB
    assert classify_source("rtsp://192.168.1.254") is CameraKind.RTSP
    assert timestamp_advances(1.0, 1.1)
    assert not timestamp_advances(1.0, 1.0)
    assert frame_gap_dropped(0.0, 0.2, 0.1)
    frame = normalize_rgb(np.zeros((4, 6, 3), dtype=np.uint8), 0.1, 10.0, dropped=False, error=None)
    assert (frame.width, frame.height, frame.orientation) == (6, 4, "upright")
    rows = registry_json()
    viofo = next(item for item in rows if item["key"] == "viofo_a229_plus")
    assert viofo["status"] == "documented_only"
    assert viofo["tested_on"] == "not tested"


def test_driver_profile_round_trip(tmp_path: Path) -> None:
    book = DriverBook(tmp_path / "drivers.json")
    book.add("teen", "Teen")
    book.save_calibration("teen", {"perclos": 0.1}, {"yaw": 2.0, "gaze_horizontal": 0.0})
    again = DriverBook(tmp_path / "drivers.json")
    assert again.active() is not None
    assert again.active().calibrated is True
    assert again.active().centre["yaw"] == pytest.approx(2.0)


def test_status_report_does_not_claim_the_pilot() -> None:
    report = render_report()
    assert "stationary pilot" in report
    assert "No physical camera, dashcam or microphone was opened." in report
    assert all(row.feature_id for row in MATRIX)
    assert "VERIFIED" in report
    assert "NOT STARTED" in report
