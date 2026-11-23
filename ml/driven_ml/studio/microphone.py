"""Ephemeral server-side microphone capture for Studio research audio.

Audio chunks are analyzed in memory and discarded immediately. Capture starts
only when research_audio_allowed(flags) is true.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..errors import ComplianceError, ConfigurationError
from ..indicators.breathing import BreathingAnalyzer, BreathingEstimate
from .audio_health import AudioFault, AudioHealthMonitor, AudioHealthReport, BreathingAudioState
from .observations import SourceKind
from .research_audio import research_audio_allowed, research_audio_block_reason

logger = logging.getLogger(__name__)

RECONNECT_DELAY_SECONDS = 1.0


@dataclass(frozen=True)
class MicDevice:
    index: int
    name: str
    channels: int
    sample_rate_hz: float

    def to_json(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "channels": self.channels,
            "sample_rate_hz": self.sample_rate_hz,
        }


def list_input_devices() -> list[MicDevice]:
    """Enumerate host input devices when PortAudio is available; else empty."""
    try:
        import sounddevice as sd
    except ImportError:
        return []
    devices: list[MicDevice] = []
    try:
        host = sd.query_devices()
    except (sd.PortAudioError, OSError, RuntimeError, ValueError) as error:
        logger.info("microphone enumeration failed: %s", error)
        return []
    for index, item in enumerate(host):
        if int(item.get("max_input_channels") or 0) <= 0:
            continue
        devices.append(
            MicDevice(
                index=index,
                name=str(item.get("name") or f"input-{index}"),
                channels=int(item["max_input_channels"]),
                sample_rate_hz=float(item.get("default_samplerate") or 16000),
            )
        )
    return devices


def microphone_backend_available() -> bool:
    """Report whether the optional PortAudio backend is installed without opening a device."""
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        return False
    return True


class ResearchMicrophone:
    """Capture thread that feeds audio health + breathing analyzers."""

    def __init__(self) -> None:
        self.health = AudioHealthMonitor()
        self.breathing = BreathingAnalyzer()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._device_index: int | None = None
        self._running = False
        self._source_kind = SourceKind.HARDWARE
        self.last_health: AudioHealthReport | None = None
        self.last_breathing: BreathingEstimate | None = None
        self._flags: dict[str, Any] = {}
        self.on_update: Callable[[AudioHealthReport, BreathingEstimate], None] | None = None
        # Each analyzed chunk, still only in memory, for the breathing indicator; nothing is stored.
        self.on_chunk: Callable[[float, np.ndarray, AudioHealthReport], None] | None = None

    @property
    def running(self) -> bool:
        return self._running

    def configure_flags(self, flags: dict[str, Any]) -> None:
        self._flags = dict(flags)

    def start(self, device_index: int | None = None, *, sample_rate_hz: int = 16000) -> None:
        if not research_audio_allowed(self._flags):
            raise ComplianceError("Research microphone blocked", [research_audio_block_reason(self._flags)])
        if self._flags.get("record_audio") is True:
            raise ComplianceError(
                "Recording disabled",
                ["record_audio must stay false for Studio research audio"],
            )
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if device_index is not None and device_index not in {device.index for device in list_input_devices()}:
            raise ValueError(f"microphone input device {device_index} is not available")
        self.stop()
        self._device_index = device_index
        self._stop.clear()
        self._source_kind = SourceKind.HARDWARE
        self._thread = threading.Thread(
            target=self._run_capture,
            args=(sample_rate_hz,),
            name="driven-research-mic",
            daemon=True,
        )
        self._running = True
        try:
            self._thread.start()
        except RuntimeError:
            self._thread = None
            self._running = False
            raise

    def stop(self) -> None:
        was_running = self._running
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():
                raise TimeoutError("microphone stream did not stop within 2 seconds")
        self._thread = None
        self._running = False
        if was_running:
            t_seconds = time.monotonic()
            health = AudioHealthReport(
                t_seconds=t_seconds,
                fault=AudioFault.NONE,
                state=BreathingAudioState.UNAVAILABLE,
                rms=0.0,
                peak=0.0,
                noise_level=0.0,
                confidence=0.0,
                detail="microphone stopped by user",
            )
            estimate = self.breathing.unavailable(t_seconds, health)
            self.last_health = health
            self.last_breathing = estimate
            self._notify(health, estimate)

    def push_synthetic(
        self,
        t_seconds: float,
        samples: np.ndarray,
        *,
        hint: AudioFault | None = None,
        source_kind: SourceKind = SourceKind.SIMULATED,
    ) -> BreathingEstimate:
        """Test/scenario path: feed PCM without opening a device."""
        health = self.health.push(
            t_seconds,
            samples,
            source_kind=source_kind,
            hint=hint,
        )
        estimate = self.breathing.push(t_seconds, samples, health)
        self.last_health = health
        self.last_breathing = estimate
        self._notify(health, estimate, samples)
        return estimate

    def mark_permission_denied(self, t_seconds: float) -> BreathingEstimate:
        health = self.health.permission_denied(t_seconds, source_kind=self._source_kind)
        estimate = self.breathing.unavailable(t_seconds, health)
        self.last_health = health
        self.last_breathing = estimate
        self._notify(health, estimate)
        return estimate

    def mark_disconnected(self, t_seconds: float) -> BreathingEstimate:
        health = self.health.disconnected(t_seconds)
        estimate = self.breathing.unavailable(t_seconds, health)
        self.last_health = health
        self.last_breathing = estimate
        self._notify(health, estimate)
        return estimate

    def _run_capture(self, sample_rate_hz: int) -> None:
        try:
            import sounddevice as sd
        except ImportError:
            logger.info("sounddevice not installed; research mic stays unavailable")
            self.last_health = self.health.disconnected(time.monotonic(), detail="sounddevice not installed")
            self.last_breathing = self.breathing.unavailable(time.monotonic(), self.last_health)
            self._notify(self.last_health, self.last_breathing)
            self._running = False
            return

        block = max(256, sample_rate_hz // 10)
        try:
            while not self._stop.is_set():
                if not self._capture_once(sd, sample_rate_hz, block):
                    break
                if self._stop.wait(RECONNECT_DELAY_SECONDS):
                    break
                self._reset_analyzers(sample_rate_hz)
        finally:
            self._running = False

    def _capture_once(self, sd: Any, sample_rate_hz: int, block: int) -> bool:
        stream_started_t = time.monotonic()

        def callback(indata: np.ndarray, frames: int, _time_info: Any, status: Any) -> None:
            self._process_chunk(indata, frames, status, sample_rate_hz)

        try:
            with sd.InputStream(
                device=self._device_index,
                channels=1,
                samplerate=sample_rate_hz,
                blocksize=block,
                dtype="float32",
                callback=callback,
            ):
                return self._wait_for_audio(stream_started_t)
        except (sd.PortAudioError, OSError, RuntimeError, ValueError) as error:
            return self._handle_stream_error(error)

    def _wait_for_audio(self, stream_started_t: float) -> bool:
        while not self._stop.wait(0.05):
            now = time.monotonic()
            stale = self.health.check_stale(now)
            last_health = self.last_health
            last_signal_t = (
                last_health.t_seconds
                if last_health is not None and last_health.t_seconds >= stream_started_t
                else stream_started_t
            )
            if stale is None and now - last_signal_t > self.health.stale_after_seconds:
                stale = self.health.disconnected(
                    now,
                    detail="microphone stream opened but delivered no audio chunks",
                )
            if stale is not None:
                estimate = self.breathing.unavailable(stale.t_seconds, stale)
                self.last_health = stale
                self.last_breathing = estimate
                self._notify(stale, estimate)
                return True
        return False

    def _process_chunk(
        self,
        indata: np.ndarray,
        frames: int,
        status: Any,
        sample_rate_hz: int,
    ) -> None:
        t_seconds = time.monotonic()
        mono = np.asarray(indata[:, 0], dtype=np.float32)
        hint = AudioFault.STREAM_INTERRUPTED if status or frames != mono.size else None
        try:
            health = self.health.push(
                t_seconds,
                mono,
                sample_rate_hz=sample_rate_hz,
                source_kind=self._source_kind,
                hint=hint,
            )
            estimate = self.breathing.push(t_seconds, mono, health)
        except (RuntimeError, ValueError) as error:
            logger.exception("microphone chunk analysis failed: %s", error)
            return
        self.last_health = health
        self.last_breathing = estimate
        self._notify(health, estimate, mono)

    def _handle_stream_error(self, error: Exception) -> bool:
        logger.info("research microphone failed: %s", error)
        detail = str(error).lower()
        t_seconds = time.monotonic()
        if "permission" in detail or "denied" in detail:
            self.mark_permission_denied(t_seconds)
            return False
        self.mark_disconnected(t_seconds)
        return True

    def _reset_analyzers(self, sample_rate_hz: int) -> None:
        self.health = AudioHealthMonitor(sample_rate_hz=sample_rate_hz)
        self.breathing = BreathingAnalyzer(sample_rate_hz=sample_rate_hz)
        self.last_health = None
        self.last_breathing = None

    def _notify(
        self, health: AudioHealthReport, estimate: BreathingEstimate, samples: np.ndarray | None = None
    ) -> None:
        chunk = np.zeros(0, dtype=np.float32) if samples is None else samples
        try:
            if self.on_update is not None:
                self.on_update(health, estimate)
            if self.on_chunk is not None:
                self.on_chunk(health.t_seconds, chunk, health)
        except (RuntimeError, TypeError, ValueError) as error:
            logger.exception("microphone update callback failed: %s", error)


def require_gate_or_raise(flags: dict[str, Any] | None) -> None:
    if not research_audio_allowed(flags):
        raise ComplianceError("Research microphone blocked", [research_audio_block_reason(flags)])
    if flags is not None and flags.get("record_audio") is True:
        raise ConfigurationError("record_audio must remain false")
