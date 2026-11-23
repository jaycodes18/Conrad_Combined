"""Driven Studio: the live engine's decisions, checked without a camera or a trained pipeline."""

from __future__ import annotations

import json
import math
from itertools import pairwise
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from driven_ml.modeling.artifact import LogisticRegressionWeights
from driven_ml.spec import FEATURE_SPEC, INPUTS_WITH_BASELINE
from driven_ml.studio.events import EventHub
from driven_ml.studio.live import LiveEngine, _model_inputs, _Session

FPS = FEATURE_SPEC.target_fps


def stub_model(weight_on: str = "delta_perclos", weight: float = 20.0, threshold: float = 0.5) -> Any:
    """A logistic model that fires on one input: enough to test the engine's alert logic."""
    inputs = list(INPUTS_WITH_BASELINE)
    coefficients = [weight if name == weight_on else 0.0 for name in inputs]
    return SimpleNamespace(
        inputs=inputs,
        fill_values=[0.0] * len(inputs),
        scaler_mean=[0.0] * len(inputs),
        scaler_scale=[1.0] * len(inputs),
        weights=LogisticRegressionWeights(coefficients=coefficients, intercept=-weight * 0.2),
        temperature=1.0,
        threshold=threshold,
        cooldown_seconds=120.0,
    )


def frame(ear: float, yaw: float = 0.0) -> dict[str, float]:
    return {
        "ear": ear,
        "ear_left": ear,
        "ear_right": ear,
        "mar": 0.1,
        "yaw": yaw,
        "pitch": -5.0,
        "roll": 0.0,
        "gaze_horizontal": 0.0,
        "blink_left": 0.0,
        "blink_right": 0.0,
        "jaw_open": 0.0,
        "look_down": 0.1,
        "face": 1.0,
    }


def drive(session: _Session, seconds: float, start: float, make: Any) -> list[dict[str, Any]]:
    alerts = []
    for index in range(int(seconds * FPS)):
        t = start + index / FPS
        alerts += session.push(t, make(t))
    return alerts


@pytest.fixture
def engine(tmp_path: Any) -> LiveEngine:
    live = LiveEngine(EventHub(), tmp_path)
    live.model = stub_model()  # type: ignore[assignment]
    return live


def test_hub_sends_json_safe_events_and_drops_oldest_when_full() -> None:
    hub = EventHub()
    inbox = hub.subscribe()
    hub.publish("x", {"value": math.nan, "array": np.float64(1.5), "nested": [math.inf]})
    message = json.loads(inbox.get_nowait())
    assert message == {"type": "x", "value": None, "array": 1.5, "nested": [None]}
    for index in range(2100):
        hub.publish("n", {"i": index})
    assert inbox.qsize() == 2000
    assert json.loads(inbox.get_nowait())["i"] == 100


def test_model_inputs_follow_the_delta_rule() -> None:
    summary = {"perclos": 0.4, "blink_rate": 20.0}
    baseline = {"perclos": 0.1, "blink_rate": 15.0}
    values = _model_inputs(
        summary,
        baseline,
        ["perclos", "delta_perclos", "delta_blink_rate", "delta_yawns"],
    )
    assert values[:3] == pytest.approx([0.4, 0.3, 5.0])
    assert math.isnan(values[3])


def test_calibrates_then_alerts_on_closed_eyes_and_escalates(
    engine: LiveEngine,
) -> None:
    session = _Session(engine)
    alert_eyes = lambda t: frame(0.30)  # noqa: E731
    assert drive(session, 125, 0.0, alert_eyes) == []
    assert session.baseline is not None
    closing = lambda t: frame(0.05 if int(t * FPS) % 10 < 6 else 0.30)  # noqa: E731 - eyes shut 60% of the time
    alerts = drive(session, 360, 125.0, closing)
    drowsy = [a for a in alerts if a["kind"] == "drowsy"]
    assert len(drowsy) >= 2
    assert all(b["t"] - a["t"] >= 120 for a, b in pairwise(drowsy))
    assert engine.risk.escalation.tier >= 2  # a second drowsiness alert inside notify.min_gap_s
    tier2 = [
        event for event in engine.risk.escalation.history if event.tier == 2 and event.kind.value == "tier_changed"
    ]
    assert tier2
    # Eyes shut 60% of the time never stay open for the 0.5 s response hold, so either Tier 2 rule may fire first.
    assert tier2[0].reason in {"No visible response to the in-car alert.", "A second alert within the notify gap."}


def test_no_alerts_while_parked(engine: LiveEngine) -> None:
    engine.driving = False
    session = _Session(engine)
    drive(session, 125, 0.0, lambda t: frame(0.30))
    assert drive(session, 240, 125.0, lambda t: frame(0.05)) == []


def test_eyes_off_rule_fires_after_two_seconds_from_the_calibrated_centre(
    engine: LiveEngine,
) -> None:
    session = _Session(engine)
    drive(session, 125, 0.0, lambda t: frame(0.30, yaw=10.0))  # phone mounted off to one side
    assert session.centre["yaw"] == pytest.approx(10.0)
    assert drive(session, 5, 125.0, lambda t: frame(0.30, yaw=30.0)) == []  # 20 degrees from centre: not away
    alerts = drive(session, 3, 130.0, lambda t: frame(0.30, yaw=45.0))
    eyes_off = [a for a in alerts if a["kind"] == "eyes_off"]
    assert len(eyes_off) == 1
    assert eyes_off[0]["score"] >= FEATURE_SPEC.live_alerts.eyes_off_seconds
