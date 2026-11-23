"""Wave B reliability checks for real sensor paths without claiming hardware validation."""

from __future__ import annotations

import json
import sys
import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import numpy as np

from driven_ml.features.geometry import no_face_frame
from driven_ml.spec import FEATURE_SPEC
from driven_ml.studio import live as live_module
from driven_ml.studio.events import EventHub
from driven_ml.studio.live import LiveEngine, _Session
from driven_ml.studio.microphone import ResearchMicrophone
from driven_ml.studio.observations import SensorObservation, SensorStatus, SourceKind
from driven_ml.studio.server import Studio
from driven_ml.studio.sources import SourceFrame


def _observation(name: str, t_seconds: float, value: float | str | None, *, valid: bool = True) -> SensorObservation:
    return SensorObservation(
        name=name,
        t_seconds=t_seconds,
        value=value,
        confidence=1.0 if valid else 0.0,
        status=SensorStatus.AVAILABLE if valid else SensorStatus.UNAVAILABLE,
        valid=valid,
        source_kind=SourceKind.SIMULATED,
        detail="test input",
    )


def test_gated_microphone_state_does_not_enumerate_devices(monkeypatch: Any) -> None:
    def forbidden() -> list[Any]:
        raise AssertionError("device enumeration must wait for legal authorization")

    monkeypatch.setattr("driven_ml.studio.server.list_input_devices", forbidden)
    monkeypatch.setattr("driven_ml.studio.server.microphone_backend_available", forbidden)
    studio = object.__new__(Studio)
    studio.product_flags = {
        "record_audio": False,
        "studio_research_audio": False,
        "legal_opinion_studio_audio": False,
    }
    studio.microphone = SimpleNamespace(running=False, last_health=None, last_breathing=None)

    state = studio.microphone_state()

    assert state["allowed"] is False
    assert state["devices"] == []
    assert state["backend_available"] is False


def test_microphone_retries_after_temporary_device_failure(monkeypatch: Any) -> None:
    recovered = threading.Event()
    attempts = 0

    class PortAudioError(Exception):
        pass

    class InputStream:
        def __init__(self, *, callback: Any, **_kwargs: Any) -> None:
            self.callback = callback

        def __enter__(self) -> InputStream:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise PortAudioError("device disconnected")
            samples = np.full((1600, 1), 0.02, dtype=np.float32)
            self.callback(samples, 1600, None, None)
            recovered.set()
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

    fake_audio_backend = SimpleNamespace(PortAudioError=PortAudioError, InputStream=InputStream)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_audio_backend)
    monkeypatch.setattr("driven_ml.studio.microphone.RECONNECT_DELAY_SECONDS", 0.01)
    microphone = ResearchMicrophone()
    microphone.configure_flags(
        {
            "record_audio": False,
            "studio_research_audio": True,
            "legal_opinion_studio_audio": True,
        }
    )

    microphone.start()
    try:
        assert recovered.wait(timeout=2.0)
        assert attempts >= 2
        assert microphone.last_health is not None
        assert microphone.last_breathing is not None
    finally:
        microphone.stop()


def test_camera_gap_clears_continuous_events_and_partial_window(tmp_path: Any) -> None:
    engine = LiveEngine(EventHub(), tmp_path)
    session = _Session(engine)
    session.baseline = {"perclos": 0.0}
    session.eyes_off.since = 0.0
    session.last_monitor = {"monitoring": "available"}
    session.buffer.append({"face": 1.0})
    session.tracker.push(
        0.0,
        face=1.0,
        ear=0.3,
        yaw=0.0,
        gaze_horizontal=0.0,
    )

    session.camera_gap(0.0, 2.1)
    alerts = session.push(
        2.1,
        {
            "face": 1.0,
            "ear": 0.3,
            "yaw": 45.0,
            "gaze_horizontal": 0.5,
        },
    )

    assert len(session.buffer) == 1
    assert session.eyes_off.since is None
    assert session.last_monitor["monitoring"] == "unavailable"
    assert engine.observations.latest("drowsiness_probability").valid is False
    assert not any(alert["kind"] == "eyes_off" for alert in alerts)


def test_camera_reconnects_after_stream_read_failure(monkeypatch: Any, tmp_path: Any) -> None:
    attempts = 0
    engine = LiveEngine(EventHub(), tmp_path)
    inbox = engine.hub.subscribe()
    monkeypatch.setattr(live_module, "_open_detector", lambda *_args: nullcontext(None))
    monkeypatch.setattr(engine, "_render", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(live_module, "CAMERA_RECONNECT_INITIAL_SECONDS", 0.01)

    def camera_frames(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("test disconnect")
        return iter([SourceFrame(t=0.0, rgb=None, features=no_face_frame())])

    monkeypatch.setattr(live_module, "iter_camera", camera_frames)
    engine._run("0", 1.0)
    events = [json.loads(inbox.get_nowait()) for _ in range(inbox.qsize())]

    assert attempts == 2
    assert engine.camera_faults == 1
    assert any(event["type"] == "live_status" and event["state"] == "recovering" for event in events)
    assert any(event["type"] == "live_status" and event["state"] == "finished" for event in events)
    assert any(event["type"] == "frame" for event in events)


def test_latency_includes_landmark_detection(tmp_path: Any) -> None:
    engine = LiveEngine(EventHub(), tmp_path)
    monkeypatched_frame = SourceFrame(t=0.0, rgb=None, features=None)
    monkeypatched_started = time.perf_counter()

    def delayed_detector(_frame: SourceFrame) -> tuple[dict[str, float], None]:
        time.sleep(0.03)
        return no_face_frame(), None

    engine._render = lambda *_args, **_kwargs: None
    engine._loop(iter([monkeypatched_frame]), delayed_detector)
    elapsed_ms = (time.perf_counter() - monkeypatched_started) * 1000.0

    assert engine.last_latency_ms is not None
    assert engine.last_latency_ms >= 25.0
    assert engine.last_latency_ms <= elapsed_ms


def test_degraded_camera_quality_prevents_window_model_scoring(tmp_path: Any) -> None:
    engine = LiveEngine(EventHub(), tmp_path)
    inbox = engine.hub.subscribe()
    engine.model = object()  # type: ignore[assignment]
    session = _Session(engine)
    session.baseline = {"perclos": 0.0}
    for index in range(FEATURE_SPEC.frames_per_window):
        t_seconds = index / FEATURE_SPEC.target_fps
        session.buffer.append(
            {
                "t": t_seconds,
                "ear": 0.3,
                "ear_left": 0.3,
                "ear_right": 0.3,
                "mar": 0.1,
                "yaw": 0.0,
                "pitch": -5.0,
                "roll": 0.0,
                "gaze_horizontal": 0.0,
                "blink_left": 0.0,
                "blink_right": 0.0,
                "jaw_open": 0.0,
                "look_down": 0.1,
                "face": 1.0,
            }
        )
    engine.observations.publish(
        SensorObservation(
            name="camera_quality",
            t_seconds=time.monotonic(),
            value="degraded",
            confidence=0.5,
            status=SensorStatus.DEGRADED,
            valid=True,
            source_kind=SourceKind.HARDWARE,
            detail="low light",
        )
    )

    alerts = session._score_window(
        60.0,
        60.0,
        SourceKind.HARDWARE,
    )

    assert alerts == []
    window = json.loads(inbox.get_nowait())
    assert window["type"] == "window"
    assert "degraded" in window["note"]
    assert "probability" not in window
