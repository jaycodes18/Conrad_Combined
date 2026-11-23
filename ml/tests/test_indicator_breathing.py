"""Breathing: the policy gate, quiet-cabin rules, interference reasons, the gasp detector and the visual fallback."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import make_frame, times

from driven_ml.features.synthetic import cabin_audio_chunks
from driven_ml.indicators import BreathingIndicator, IndicatorReading
from driven_ml.studio.audio_health import AudioFault, AudioHealthMonitor
from driven_ml.studio.observations import SourceKind


def listen(
    kind: str,
    seconds: float,
    *,
    driving: bool = False,
    indicator: BreathingIndicator | None = None,
    hint: AudioFault | None = None,
    start_s: float = 0.0,
    monitor: AudioHealthMonitor | None = None,
) -> tuple[BreathingIndicator, list[IndicatorReading]]:
    breathing = indicator or BreathingIndicator(audio_allowed=True)
    health_monitor = monitor or AudioHealthMonitor()
    readings = []
    for t, pcm in cabin_audio_chunks(kind, seconds, seed=5, start_s=start_s):
        health = health_monitor.push(t, pcm, source_kind=SourceKind.SIMULATED, hint=hint)
        readings.append(breathing.push_audio(t, pcm, health, driving=driving))
    return breathing, readings


def test_closed_gate_is_unavailable_by_policy() -> None:
    breathing = BreathingIndicator(audio_allowed=False)
    _, readings = listen("quiet_breathing", 5.0, indicator=breathing)
    assert {reading.unavailable_reason for reading in readings} == {"mic_blocked_by_policy"}
    assert breathing.current(5.0).unavailable_reason == "mic_blocked_by_policy"


def test_quiet_breathing_gives_a_regular_rate() -> None:
    _, readings = listen("quiet_breathing", 30.0)
    last = readings[-1]
    assert last.state == "regular"
    assert last.value == pytest.approx(15.0, abs=2.0)


@pytest.mark.parametrize(
    ("kind", "hint", "reason"),
    [
        ("mic_road_noise", None, "road_noise"),
        ("speech", None, "speech"),
        ("music", AudioFault.MUSIC, "music"),
        ("clipping", None, "mic_fault"),
    ],
)
def test_interference_is_unavailable_with_its_reason(kind: str, hint: AudioFault | None, reason: str) -> None:
    _, readings = listen(kind, 25.0, hint=hint)
    later = readings[20:]
    assert all(not reading.available for reading in later), [reading.state for reading in later]
    assert all(reading.severity == 0.0 for reading in later)
    assert later[-1].unavailable_reason == reason


def test_disconnect_is_a_mic_fault() -> None:
    breathing, _ = listen("quiet_breathing", 3.0)
    monitor = AudioHealthMonitor()
    health = monitor.disconnected(3.1, source_kind=SourceKind.SIMULATED)
    reading = breathing.push_audio(3.1, np.zeros(0, dtype=np.float32), health, driving=False)
    assert reading.unavailable_reason == "mic_fault"


def test_silence_is_only_not_detected_when_parked_quiet_and_warmed_up() -> None:
    _, parked = listen("mic_silence_parked", 25.0)
    states = [(reading.state, reading.unavailable_reason) for reading in parked]
    assert states[100] == ("unavailable", "mic_warming_up")  # 10 s in: under the 20 s warm-up
    assert parked[-1].state == "not_detected_quiet_cabin"
    assert parked[-1].value is None
    _, moving = listen("mic_silence_parked", 25.0, driving=True)
    assert moving[-1].unavailable_reason == "road_noise"


def test_recent_speech_blocks_not_detected() -> None:
    monitor = AudioHealthMonitor()
    breathing, _ = listen("mic_silence_parked", 22.0, monitor=monitor)
    listen("speech", 2.0, indicator=breathing, monitor=monitor, start_s=22.0)
    _, after = listen("mic_silence_parked", 5.0, indicator=breathing, monitor=monitor, start_s=24.0)
    assert after[-1].unavailable_reason == "speech"


def test_isolated_gasps_in_a_quiet_parked_cabin_are_suspected() -> None:
    _, readings = listen("gasps", 40.0)
    agonal = [reading for reading in readings if reading.state == "agonal_pattern_suspected"]
    assert agonal, "no agonal_pattern_suspected reading"
    assert agonal[0].t_seconds >= 33.0  # the second gasp, 9 s after the first
    assert "Experimental" in agonal[0].evidence[0]
    _, moving = listen("gasps", 40.0, driving=True)
    assert not any(reading.state == "agonal_pattern_suspected" for reading in moving)


def test_visual_fallback_needs_the_face_and_the_shoulders() -> None:
    breathing = BreathingIndicator(audio_allowed=False)
    readings = [
        breathing.push_frame(make_frame(t, shoulder_y=0.6 + 0.01 * np.sin(2 * np.pi * 0.25 * t)))
        for t in times(0.0, 30.0)
    ]
    assert readings[-1].state == "regular"
    assert readings[-1].value == pytest.approx(15.0, abs=3.0)
    assert breathing.push_frame(make_frame(30.1, face=False)).unavailable_reason == "mic_blocked_by_policy"
    assert breathing._visual is not None
    assert breathing._visual.unavailable_reason == "face_not_tracked"
    assert breathing.push_frame(make_frame(30.2)).unavailable_reason == "mic_blocked_by_policy"
    assert breathing._visual.unavailable_reason == "shoulders_out_of_frame"
