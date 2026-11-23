"""Microphone health classification for research breathing analysis.

Detects disconnect, mute, clipping, low/high gain, noise interference, frozen
streams, and recovery. A failed mic never implies impairment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np

from .observations import SensorObservation, SensorStatus, SourceKind


class AudioFault(StrEnum):
    NONE = "none"
    DISCONNECTED = "disconnected"
    PERMISSION_DENIED = "permission_denied"
    MUTED = "muted"
    GAIN_TOO_LOW = "gain_too_low"
    GAIN_TOO_HIGH = "gain_too_high"
    CLIPPING = "clipping"
    WIND = "wind"
    ROAD_NOISE = "road_noise"
    HVAC = "hvac"
    MUSIC = "music"
    SPEECH = "speech"
    STREAM_INTERRUPTED = "stream_interrupted"
    FROZEN = "frozen"
    SAMPLE_RATE_CHANGED = "sample_rate_changed"
    INSUFFICIENT_SIGNAL = "insufficient_signal"


class BreathingAudioState(StrEnum):
    ACTIVE_RELIABLE = "active_reliable"
    ACTIVE_REDUCED = "active_reduced"
    INTERFERENCE = "environmental_interference"
    UNAVAILABLE = "unavailable"
    RECOVERING = "recovering"


@dataclass(frozen=True)
class AudioHealthReport:
    t_seconds: float
    fault: AudioFault
    state: BreathingAudioState
    rms: float
    peak: float
    noise_level: float
    confidence: float
    detail: str
    source_kind: SourceKind = SourceKind.HARDWARE

    def to_json(self) -> dict[str, Any]:
        return {
            "t": self.t_seconds,
            "fault": self.fault.value,
            "state": self.state.value,
            "rms": self.rms,
            "peak": self.peak,
            "noise_level": self.noise_level,
            "confidence": self.confidence,
            "detail": self.detail,
            "source_kind": self.source_kind.value,
        }

    def observation(self) -> SensorObservation:
        status = {
            BreathingAudioState.ACTIVE_RELIABLE: SensorStatus.AVAILABLE,
            BreathingAudioState.ACTIVE_REDUCED: SensorStatus.DEGRADED,
            BreathingAudioState.INTERFERENCE: SensorStatus.DEGRADED,
            BreathingAudioState.UNAVAILABLE: SensorStatus.UNAVAILABLE,
            BreathingAudioState.RECOVERING: SensorStatus.RECOVERING,
        }[self.state]
        return SensorObservation(
            name="microphone_quality",
            t_seconds=self.t_seconds,
            value=self.state.value,
            confidence=self.confidence,
            status=status,
            valid=self.state
            in {
                BreathingAudioState.ACTIVE_RELIABLE,
                BreathingAudioState.ACTIVE_REDUCED,
                BreathingAudioState.INTERFERENCE,
                BreathingAudioState.RECOVERING,
            },
            source_kind=self.source_kind,
            detail=self.detail,
        )


@dataclass
class AudioHealthMonitor:
    """Stateful classifier over successive PCM chunks (float32 mono, -1..1)."""

    sample_rate_hz: int = 16000
    mute_rms: float = 0.001
    low_gain_rms: float = 0.005
    high_peak: float = 0.95
    stale_after_seconds: float = 1.0
    _last_chunk_t: float | None = None
    _last_rms: float = 0.0
    _was_fault: bool = False
    _expected_rate: int | None = None
    _stale_reported: bool = False
    _last_seen_t: float | None = None
    _last_source_kind: SourceKind = SourceKind.HARDWARE
    history: list[AudioHealthReport] = field(default_factory=list)

    def disconnected(
        self,
        t_seconds: float,
        *,
        detail: str = "microphone disconnected",
        source_kind: SourceKind = SourceKind.HARDWARE,
    ) -> AudioHealthReport:
        report = AudioHealthReport(
            t_seconds=t_seconds,
            fault=AudioFault.DISCONNECTED,
            state=BreathingAudioState.UNAVAILABLE,
            rms=0.0,
            peak=0.0,
            noise_level=0.0,
            confidence=0.0,
            detail=detail,
            source_kind=source_kind,
        )
        self._was_fault = True
        self.history.append(report)
        return report

    def permission_denied(
        self, t_seconds: float, *, source_kind: SourceKind = SourceKind.HARDWARE
    ) -> AudioHealthReport:
        report = AudioHealthReport(
            t_seconds=t_seconds,
            fault=AudioFault.PERMISSION_DENIED,
            state=BreathingAudioState.UNAVAILABLE,
            rms=0.0,
            peak=0.0,
            noise_level=0.0,
            confidence=0.0,
            detail="microphone permission denied or revoked",
            source_kind=source_kind,
        )
        self._was_fault = True
        self.history.append(report)
        return report

    def push(
        self,
        t_seconds: float,
        samples: np.ndarray,
        *,
        sample_rate_hz: int | None = None,
        source_kind: SourceKind = SourceKind.HARDWARE,
        hint: AudioFault | None = None,
    ) -> AudioHealthReport:
        rate = sample_rate_hz or self.sample_rate_hz
        if rate <= 0 or not math.isfinite(t_seconds):
            raise ValueError("audio timestamp and sample rate must be valid")
        if self._last_seen_t is not None and t_seconds <= self._last_seen_t:
            raise ValueError("audio chunk timestamps must increase monotonically")
        self._last_seen_t = t_seconds
        self._last_source_kind = source_kind
        if self._expected_rate is None:
            self._expected_rate = rate
        if rate != self._expected_rate:
            previous_rate = self._expected_rate
            self._expected_rate = rate
            self._was_fault = True
            return self._record(
                _report(
                    t_seconds,
                    AudioFault.SAMPLE_RATE_CHANGED,
                    BreathingAudioState.UNAVAILABLE,
                    f"sample rate changed from {previous_rate} to {rate}",
                    source_kind=source_kind,
                )
            )
        if self._last_chunk_t is not None and t_seconds - self._last_chunk_t > self.stale_after_seconds:
            fault = AudioFault.FROZEN if samples.size == 0 else AudioFault.STREAM_INTERRUPTED
            state = BreathingAudioState.RECOVERING if self._was_fault else BreathingAudioState.UNAVAILABLE
            self._was_fault = True
            self._last_chunk_t = t_seconds
            return self._record(
                _report(
                    t_seconds,
                    fault,
                    state,
                    "audio stream interrupted or frozen",
                    source_kind=source_kind,
                )
            )
        if samples.size == 0:
            self._was_fault = True
            return self._record(
                _report(
                    t_seconds,
                    AudioFault.DISCONNECTED,
                    BreathingAudioState.UNAVAILABLE,
                    "empty audio chunk",
                    source_kind=source_kind,
                )
            )

        pcm = np.asarray(samples, dtype=np.float64).reshape(-1)
        if not np.isfinite(pcm).all():
            self._was_fault = True
            return self._record(
                _report(
                    t_seconds,
                    AudioFault.GAIN_TOO_HIGH,
                    BreathingAudioState.INTERFERENCE,
                    "non-finite samples; audio signal is invalid",
                    source_kind=source_kind,
                )
            )
        rms = float(np.sqrt(np.mean(np.square(pcm))))
        peak = float(np.max(np.abs(pcm)))
        noise = _high_frequency_noise_ratio(pcm, rate)
        self._last_chunk_t = t_seconds
        self._last_rms = rms
        self._stale_reported = False

        if hint is not None and hint is not AudioFault.NONE:
            report = self._from_hint(t_seconds, rms, peak, noise, hint, source_kind=source_kind)
            return self._record(report)
        fault, state, detail, confidence = _classify_pcm(
            pcm,
            rate,
            rms,
            peak,
            noise,
            mute_rms=self.mute_rms,
            low_gain_rms=self.low_gain_rms,
            high_peak=self.high_peak,
            recovering=self._was_fault,
        )
        if fault is not AudioFault.NONE and state is not BreathingAudioState.RECOVERING:
            self._was_fault = True
        elif state is BreathingAudioState.RECOVERING or fault is AudioFault.NONE:
            self._was_fault = False
        return self._record(
            _report(
                t_seconds,
                fault,
                state,
                detail,
                source_kind=source_kind,
                rms=rms,
                peak=peak,
                noise_level=noise,
                confidence=confidence,
            )
        )

    def _record(self, report: AudioHealthReport) -> AudioHealthReport:
        self.history.append(report)
        return report

    def check_stale(self, t_seconds: float) -> AudioHealthReport | None:
        """Mark a frozen stream unavailable after the last good chunk goes stale."""
        if not math.isfinite(t_seconds) or (self._last_seen_t is not None and t_seconds < self._last_seen_t):
            raise ValueError("stale-check timestamp must be finite and not precede the latest audio chunk")
        if self._last_chunk_t is None or t_seconds - self._last_chunk_t <= self.stale_after_seconds:
            return None
        if self._stale_reported:
            return None
        self._stale_reported = True
        self._was_fault = True
        report = AudioHealthReport(
            t_seconds=t_seconds,
            fault=AudioFault.FROZEN,
            state=BreathingAudioState.UNAVAILABLE,
            rms=self._last_rms,
            peak=0.0,
            noise_level=0.0,
            confidence=0.0,
            detail="audio stream is stale; previous breathing values are no longer current",
            source_kind=self._last_source_kind,
        )
        self.history.append(report)
        return report

    def _from_hint(
        self,
        t_seconds: float,
        rms: float,
        peak: float,
        noise: float,
        hint: AudioFault,
        *,
        source_kind: SourceKind,
    ) -> AudioHealthReport:
        mapping: dict[AudioFault, tuple[BreathingAudioState, float, str]] = {
            AudioFault.WIND: (BreathingAudioState.INTERFERENCE, 0.25, "wind interference"),
            AudioFault.HVAC: (BreathingAudioState.INTERFERENCE, 0.3, "HVAC noise"),
            AudioFault.MUSIC: (BreathingAudioState.INTERFERENCE, 0.25, "loud music interference"),
            AudioFault.SPEECH: (BreathingAudioState.INTERFERENCE, 0.3, "passenger conversation"),
            AudioFault.GAIN_TOO_HIGH: (BreathingAudioState.INTERFERENCE, 0.2, "gain too high"),
            AudioFault.CLIPPING: (BreathingAudioState.INTERFERENCE, 0.2, "clipping"),
            AudioFault.STREAM_INTERRUPTED: (
                BreathingAudioState.RECOVERING,
                0.0,
                "audio stream interrupted",
            ),
            AudioFault.INSUFFICIENT_SIGNAL: (
                BreathingAudioState.ACTIVE_REDUCED,
                0.2,
                "insufficient respiratory signal",
            ),
        }
        state, conf, detail = mapping.get(
            hint,
            (BreathingAudioState.UNAVAILABLE, 0.0, hint.value),
        )
        self._was_fault = state is not BreathingAudioState.ACTIVE_RELIABLE
        return AudioHealthReport(
            t_seconds=t_seconds,
            fault=hint,
            state=state,
            rms=rms,
            peak=peak,
            noise_level=noise,
            confidence=conf,
            detail=detail,
            source_kind=source_kind,
        )


def _likely_speech(pcm: np.ndarray, sample_rate_hz: int) -> bool:
    """Rough voiced-energy heuristic; not a voiceprint and not stored."""
    if pcm.size < sample_rate_hz // 10:
        return False
    # Zero-crossing rate is higher for noise; voiced speech has moderate ZCR and energy.
    signs = np.sign(pcm)
    zcr = float(np.mean(np.abs(np.diff(signs)) > 0))
    rms = float(np.sqrt(np.mean(np.square(pcm))))
    return 0.02 < zcr < 0.25 and rms > 0.02 and math.isfinite(zcr)


def _high_frequency_noise_ratio(pcm: np.ndarray, sample_rate_hz: int) -> float:
    if pcm.size < 2:
        return 0.0
    spectrum = np.fft.rfft(pcm * np.hanning(pcm.size))
    power = np.square(np.abs(spectrum))
    total_power = float(np.sum(power))
    if total_power <= 0.0:
        return 0.0
    frequencies = np.fft.rfftfreq(pcm.size, d=1.0 / sample_rate_hz)
    return float(np.sum(power[frequencies >= 500.0]) / total_power)


def _report(
    t_seconds: float,
    fault: AudioFault,
    state: BreathingAudioState,
    detail: str,
    *,
    source_kind: SourceKind = SourceKind.HARDWARE,
    rms: float = 0.0,
    peak: float = 0.0,
    noise_level: float = 0.0,
    confidence: float = 0.0,
) -> AudioHealthReport:
    return AudioHealthReport(
        t_seconds=t_seconds,
        fault=fault,
        state=state,
        rms=rms,
        peak=peak,
        noise_level=noise_level,
        confidence=confidence,
        detail=detail,
        source_kind=source_kind,
    )


def _classify_pcm(
    pcm: np.ndarray,
    sample_rate_hz: int,
    rms: float,
    peak: float,
    noise: float,
    *,
    mute_rms: float,
    low_gain_rms: float,
    high_peak: float,
    recovering: bool,
) -> tuple[AudioFault, BreathingAudioState, str, float]:
    if peak >= high_peak:
        result = AudioFault.CLIPPING, BreathingAudioState.INTERFERENCE, "clipping or distortion detected", 0.2
    elif rms < mute_rms:
        result = AudioFault.MUTED, BreathingAudioState.UNAVAILABLE, "microphone appears muted or silent", 0.0
    elif rms < low_gain_rms:
        result = (
            AudioFault.GAIN_TOO_LOW,
            BreathingAudioState.ACTIVE_REDUCED,
            "microphone gain too low for reliable respiratory signal",
            0.25,
        )
    elif _likely_speech(pcm, sample_rate_hz):
        result = AudioFault.SPEECH, BreathingAudioState.INTERFERENCE, "competing speech energy detected", 0.3
    elif noise > 0.08 and rms > 0.05:
        result = AudioFault.ROAD_NOISE, BreathingAudioState.INTERFERENCE, "broadband noise interference", 0.35
    elif recovering:
        result = AudioFault.NONE, BreathingAudioState.RECOVERING, "audio quality recovering", 0.55
    else:
        result = (
            AudioFault.NONE,
            BreathingAudioState.ACTIVE_RELIABLE,
            "microphone signal usable for research breathing analysis",
            0.85,
        )
    return result
