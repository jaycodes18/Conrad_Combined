"""Respiration, audio first with a visual fallback, as one indicator.

Audio: 100 ms PCM chunks are analyzed in memory and dropped. Nothing is written to disk and there is no
speech-to-text. The microphone path runs only behind ``research_audio_allowed(flags)``; with the gate closed
(the default product state) every reading is ``unavailable`` with reason ``mic_blocked_by_policy``. Microphone
health comes from ``studio.audio_health``.

Absence of breathing sound means nothing on its own. ``not_detected_quiet_cabin`` is allowed only when the
vehicle is stationary, audio health is ok with the noise floor under ``quiet_noise_floor_rms``, no speech has
been heard for ``speech_quiet_seconds`` and the microphone has been live for ``mic_warmup_seconds``. In every
other case, no breathing sound is ``unavailable`` with reason ``road_noise``, ``speech``, ``music``, ``hvac`` or
``mic_fault``. While the car moves, road noise is assumed, so absence there is ``road_noise``.

``agonal_pattern_suspected`` is EXPERIMENTAL: isolated, gasp-like broadband bursts at long, irregular intervals
in a quiet, parked cabin, with no regular breathing in between. It is modelled on Chan et al., npj Digital
Medicine 2:52 (2019), doi:10.1038/s41746-019-0128-7, who classified agonal breathing in 9-1-1 call audio with an
SVM on smart-device audio. Here it is a hand-built detector on RMS and spectral flatness; no Driven data
validates it.

Visual fallback: chest and shoulder motion, used only when the shoulders are in frame (``shoulder_y``). If the
face is lost, the visual reading is unavailable. The face landmarker does not track shoulders, so Studio's live
camera path never supplies ``shoulder_y`` today.

``value`` is breaths per minute when a rate is measured. States: ``regular``, ``irregular``,
``agonal_pattern_suspected``, ``not_detected_quiet_cabin``, ``unavailable``.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import pairwise
from typing import Any

import numpy as np

from ..spec import INDICATOR_SPEC, BreathingConfig
from ..studio.audio_health import AudioFault, AudioHealthReport, BreathingAudioState
from ..studio.observations import SensorObservation, SensorStatus, SourceKind
from .base import FrameSignals, IndicatorName, IndicatorReading, clamp01, unavailable_reading

REGULAR = "regular"
IRREGULAR = "irregular"
AGONAL_PATTERN_SUSPECTED = "agonal_pattern_suspected"
NOT_DETECTED_QUIET_CABIN = "not_detected_quiet_cabin"
SUSPECTED_STATES = frozenset({AGONAL_PATTERN_SUSPECTED, NOT_DETECTED_QUIET_CABIN})


class BreathingUnavailable(StrEnum):
    """Why there is no breathing reading. The first five are reasons a live microphone hears no breathing."""

    ROAD_NOISE = "road_noise"
    SPEECH = "speech"
    MUSIC = "music"
    HVAC = "hvac"
    MIC_FAULT = "mic_fault"
    MIC_BLOCKED_BY_POLICY = "mic_blocked_by_policy"
    MIC_WARMING_UP = "mic_warming_up"
    FACE_NOT_TRACKED = "face_not_tracked"
    SHOULDERS_OUT_OF_FRAME = "shoulders_out_of_frame"
    COLLECTING = "collecting"


CABIN_NOISE_REASONS = frozenset(
    {
        BreathingUnavailable.ROAD_NOISE,
        BreathingUnavailable.SPEECH,
        BreathingUnavailable.MUSIC,
        BreathingUnavailable.HVAC,
    }
)
_FAULT_REASON: dict[AudioFault, BreathingUnavailable] = {
    AudioFault.ROAD_NOISE: BreathingUnavailable.ROAD_NOISE,
    AudioFault.WIND: BreathingUnavailable.ROAD_NOISE,
    AudioFault.SPEECH: BreathingUnavailable.SPEECH,
    AudioFault.MUSIC: BreathingUnavailable.MUSIC,
    AudioFault.HVAC: BreathingUnavailable.HVAC,
    AudioFault.CLIPPING: BreathingUnavailable.MIC_FAULT,
    AudioFault.GAIN_TOO_HIGH: BreathingUnavailable.MIC_FAULT,
    AudioFault.GAIN_TOO_LOW: BreathingUnavailable.MIC_FAULT,
    AudioFault.DISCONNECTED: BreathingUnavailable.MIC_FAULT,
    AudioFault.PERMISSION_DENIED: BreathingUnavailable.MIC_FAULT,
    AudioFault.MUTED: BreathingUnavailable.MIC_FAULT,
    AudioFault.FROZEN: BreathingUnavailable.MIC_FAULT,
    AudioFault.STREAM_INTERRUPTED: BreathingUnavailable.MIC_FAULT,
    AudioFault.SAMPLE_RATE_CHANGED: BreathingUnavailable.MIC_FAULT,
    AudioFault.INSUFFICIENT_SIGNAL: BreathingUnavailable.MIC_FAULT,
}
_HARD_FAULTS = frozenset(
    {
        AudioFault.DISCONNECTED,
        AudioFault.PERMISSION_DENIED,
        AudioFault.MUTED,
        AudioFault.FROZEN,
        AudioFault.STREAM_INTERRUPTED,
        AudioFault.SAMPLE_RATE_CHANGED,
    }
)
_MESSAGES: dict[str, str] = {
    BreathingUnavailable.ROAD_NOISE: "Road or wind noise covers breathing sounds.",
    BreathingUnavailable.SPEECH: "People are talking, so breathing sounds cannot be separated.",
    BreathingUnavailable.MUSIC: "Music covers breathing sounds.",
    BreathingUnavailable.HVAC: "Fan noise covers breathing sounds.",
    BreathingUnavailable.MIC_FAULT: "The microphone signal is not usable.",
    BreathingUnavailable.MIC_BLOCKED_BY_POLICY: "The microphone is off until a written legal review allows it.",
    BreathingUnavailable.MIC_WARMING_UP: "The microphone has not been live long enough to judge the cabin.",
    BreathingUnavailable.FACE_NOT_TRACKED: "The face is not tracked, so visual breathing is not measured.",
    BreathingUnavailable.SHOULDERS_OUT_OF_FRAME: "The shoulders are not in frame, so visual breathing is not measured.",
    BreathingUnavailable.COLLECTING: "Collecting enough signal for a breathing estimate.",
}


# ---- breath candidates from audio (moved from studio/breathing.py) ---------------------------------------
class BreathEventKind(StrEnum):
    CONFIRMED = "confirmed_candidate"
    UNCERTAIN = "uncertain"
    NOISE = "noise"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class BreathEvent:
    t_seconds: float
    kind: BreathEventKind
    amplitude: float

    def to_json(self) -> dict[str, Any]:
        return {"t": self.t_seconds, "kind": self.kind.value, "amplitude": self.amplitude}


@dataclass(frozen=True)
class BreathingEstimate:
    """The analyzer's output. A silent microphone gives no rate, never 0 breaths per minute."""

    t_seconds: float
    breaths_per_minute: float | None
    regularity: float | None
    confidence: float
    status: SensorStatus
    detail: str
    intervals_seconds: tuple[float, ...]
    events: tuple[BreathEvent, ...]
    audio_state: BreathingAudioState
    source_kind: SourceKind = SourceKind.HARDWARE
    baseline_bpm: float | None = None
    baseline_delta_percent: float | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "t": self.t_seconds,
            "breaths_per_minute": self.breaths_per_minute,
            "regularity": self.regularity,
            "confidence": self.confidence,
            "status": self.status.value,
            "detail": self.detail,
            "intervals_seconds": list(self.intervals_seconds),
            "events": [event.to_json() for event in self.events],
            "audio_state": self.audio_state.value,
            "source_kind": self.source_kind.value,
            "baseline_bpm": self.baseline_bpm,
            "baseline_delta_percent": self.baseline_delta_percent,
        }

    def rate_observation(self) -> SensorObservation:
        return SensorObservation(
            name="breathing_rate",
            t_seconds=self.t_seconds,
            value=self.breaths_per_minute,
            confidence=self.confidence,
            status=self.status,
            valid=self.breaths_per_minute is not None and self.status is not SensorStatus.UNAVAILABLE,
            source_kind=self.source_kind,
            unit="breaths_per_minute",
            detail=self.detail,
        )

    def regularity_observation(self) -> SensorObservation:
        return SensorObservation(
            name="breathing_regularity",
            t_seconds=self.t_seconds,
            value=self.regularity,
            confidence=self.confidence,
            status=self.status,
            valid=self.regularity is not None and self.status is not SensorStatus.UNAVAILABLE,
            source_kind=self.source_kind,
            detail=self.detail,
        )

    def confidence_observation(self) -> SensorObservation:
        return SensorObservation(
            name="respiratory_confidence",
            t_seconds=self.t_seconds,
            value=self.confidence,
            confidence=1.0 if self.status is not SensorStatus.UNAVAILABLE else 0.0,
            status=self.status,
            valid=True,
            source_kind=self.source_kind,
            detail=self.detail,
        )


@dataclass
class BreathingAnalyzer:
    """Band-limited envelope peak detection over a short RAM ring buffer."""

    sample_rate_hz: int = 16000
    min_interval_seconds: float = 1.5
    max_interval_seconds: float = 8.0
    min_confirmed_intervals: int = 3
    buffer_seconds: float = 30.0
    min_peak_ratio: float = INDICATOR_SPEC.breathing.breath_min_peak_ratio
    _envelope: deque[tuple[float, float]] = field(default_factory=deque)
    _event_times: deque[float] = field(default_factory=deque)
    _events: deque[BreathEvent] = field(default_factory=lambda: deque(maxlen=12))
    _baseline_bpm: float | None = None
    _last_processed_t: float | None = None
    _last_chunk_t: float | None = None
    last: BreathingEstimate | None = None

    def unavailable(
        self,
        t_seconds: float,
        health: AudioHealthReport,
        *,
        detail: str | None = None,
    ) -> BreathingEstimate:
        estimate = BreathingEstimate(
            t_seconds=t_seconds,
            breaths_per_minute=None,
            regularity=None,
            confidence=0.0,
            status=SensorStatus.UNAVAILABLE,
            detail=detail or "breathing signal unavailable",
            intervals_seconds=(),
            events=(BreathEvent(t_seconds, BreathEventKind.UNAVAILABLE, 0.0),),
            audio_state=health.state,
            source_kind=health.source_kind,
        )
        self.last = estimate
        return estimate

    def push(self, t_seconds: float, samples: np.ndarray, health: AudioHealthReport) -> BreathingEstimate:
        if health.state is BreathingAudioState.UNAVAILABLE or health.fault in _HARD_FAULTS:
            return self.unavailable(t_seconds, health)
        pcm = np.asarray(samples, dtype=np.float64).reshape(-1)
        if pcm.size == 0 or not np.isfinite(pcm).all():
            return self.unavailable(t_seconds, health, detail="breathing signal unavailable")
        chunk_duration = pcm.size / float(self.sample_rate_hz)
        block_samples = max(1, self.sample_rate_hz // 10)
        first_sample_t = t_seconds - chunk_duration
        for offset in range(0, pcm.size, block_samples):
            block = pcm[offset : offset + block_samples]
            if block.size == 0:
                continue
            sample_t = first_sample_t + (offset + block.size / 2) / self.sample_rate_hz
            if self._last_chunk_t is None or sample_t > self._last_chunk_t:
                self._envelope.append((sample_t, float(np.sqrt(np.mean(np.square(block))))))
                self._last_chunk_t = sample_t
        self._trim(t_seconds)
        events = self._detect_events(health)
        self._events.extend(events)
        intervals = _intervals(list(self._event_times), self.min_interval_seconds, self.max_interval_seconds)
        if health.state is BreathingAudioState.INTERFERENCE:
            return self._remember(
                BreathingEstimate(
                    t_seconds=t_seconds,
                    breaths_per_minute=None,
                    regularity=None,
                    confidence=min(0.35, health.confidence),
                    status=SensorStatus.DEGRADED,
                    detail="environmental interference; rate withheld",
                    intervals_seconds=tuple(intervals),
                    events=tuple(self._events),
                    audio_state=health.state,
                    source_kind=health.source_kind,
                )
            )
        if len(intervals) < self.min_confirmed_intervals:
            return self._remember(
                BreathingEstimate(
                    t_seconds=t_seconds,
                    breaths_per_minute=None,
                    regularity=None,
                    confidence=max(0.1, health.confidence * 0.4),
                    status=SensorStatus.DEGRADED
                    if health.state is BreathingAudioState.ACTIVE_REDUCED
                    else SensorStatus.UNAVAILABLE,
                    detail="breathing signal unavailable; insufficient confirmed respiratory intervals",
                    intervals_seconds=tuple(intervals),
                    events=tuple(self._events),
                    audio_state=health.state,
                    source_kind=health.source_kind,
                )
            )
        bpm = 60.0 / float(np.mean(intervals))
        regularity = float(1.0 / (1.0 + np.std(intervals)))
        confidence = float(min(0.95, health.confidence * (0.5 + 0.5 * min(1.0, len(intervals) / 6.0))))
        if self._baseline_bpm is None:
            self._baseline_bpm = bpm
        delta = abs(bpm - self._baseline_bpm) / max(self._baseline_bpm, 1e-6)
        detail = "unvalidated breathing-rate estimate from confirmed audio candidates"
        if delta > 0.35:
            detail = "breathing-rate change from this session's start; not a medical finding"
            confidence *= 0.85
        status = (
            SensorStatus.AVAILABLE if health.state is BreathingAudioState.ACTIVE_RELIABLE else SensorStatus.DEGRADED
        )
        return self._remember(
            BreathingEstimate(
                t_seconds=t_seconds,
                breaths_per_minute=bpm,
                regularity=regularity,
                confidence=confidence,
                status=status,
                detail=detail,
                intervals_seconds=tuple(intervals),
                events=tuple(self._events),
                audio_state=health.state,
                source_kind=health.source_kind,
                baseline_bpm=self._baseline_bpm,
                baseline_delta_percent=delta * 100.0,
            )
        )

    def _remember(self, estimate: BreathingEstimate) -> BreathingEstimate:
        self.last = estimate
        return estimate

    def _trim(self, t_seconds: float) -> None:
        cutoff = t_seconds - self.buffer_seconds
        while self._envelope and self._envelope[0][0] < cutoff:
            self._envelope.popleft()
        while self._event_times and self._event_times[0] < cutoff:
            self._event_times.popleft()
        while self._events and self._events[0].t_seconds < cutoff:
            self._events.popleft()

    def _detect_events(self, health: AudioHealthReport) -> list[BreathEvent]:
        if len(self._envelope) < 4:
            return []
        times = np.array([item[0] for item in self._envelope], dtype=np.float64)
        values = np.array([item[1] for item in self._envelope], dtype=np.float64)
        smoothed = np.convolve(values, np.ones(3, dtype=np.float64) / 3.0, mode="same")
        median = float(np.median(smoothed))
        threshold = max(median + 0.5 * float(np.std(smoothed)), self.min_peak_ratio * median)
        last_t = self._event_times[-1] if self._event_times else -math.inf
        index = len(smoothed) - 2
        t = float(times[index])
        if self._last_processed_t is not None and t <= self._last_processed_t:
            return []
        self._last_processed_t = t
        amplitude = float(smoothed[index])
        if amplitude <= threshold or amplitude < smoothed[index - 1] or amplitude < smoothed[index + 1]:
            return []
        if t - last_t < self.min_interval_seconds:
            return [BreathEvent(t, BreathEventKind.NOISE, amplitude)]
        if health.state is BreathingAudioState.INTERFERENCE:
            return [BreathEvent(t, BreathEventKind.UNCERTAIN, amplitude)]
        self._event_times.append(t)
        return [BreathEvent(t, BreathEventKind.CONFIRMED, amplitude)]


def _intervals(times: list[float], minimum: float, maximum: float) -> list[float]:
    return [current - previous for previous, current in pairwise(times) if minimum <= current - previous <= maximum]


# ---- visual fallback (moved from studio/visual_respiration.py) -------------------------------------------
@dataclass
class VisualRespirationTracker:
    """Zero crossings of the smoothed shoulder line as a weak respiration-rate proxy."""

    window_seconds: float = 25.0
    _samples: deque[tuple[float, float]] = field(default_factory=deque)

    def push(self, t_seconds: float, shoulder_y: float) -> float | None:
        """Breaths per minute, or None while the signal is too short or too weak."""
        self._samples.append((t_seconds, shoulder_y))
        while self._samples and self._samples[0][0] < t_seconds - self.window_seconds:
            self._samples.popleft()
        if len(self._samples) < 20:
            return None
        times = np.array([item[0] for item in self._samples], dtype=np.float64)
        values = np.array([item[1] for item in self._samples], dtype=np.float64)
        smooth = np.convolve(values - np.mean(values), np.ones(5) / 5.0, mode="valid")
        if smooth.size < 8:
            return None
        crossings = int(np.sum(np.abs(np.diff(np.sign(smooth))) > 0))
        duration = float(times[-1] - times[0])
        if duration < 8.0 or crossings < 2:
            return None
        bpm = 60.0 * (crossings / 2.0) / duration
        return bpm if 4.0 <= bpm <= 40.0 else None

    def reset(self) -> None:
        self._samples.clear()


# ---- the indicator -----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class _Chunk:
    t_seconds: float
    rms: float
    flatness: float
    fault: AudioFault


def spectral_flatness(pcm: np.ndarray) -> float:
    """Geometric over arithmetic mean of the power spectrum: near 0 for a tone or voice, higher for noise."""
    if pcm.size < 16:
        return 0.0
    power = np.square(np.abs(np.fft.rfft(pcm * np.hanning(pcm.size)))) + 1e-12
    return float(np.exp(np.mean(np.log(power))) / np.mean(power))


class BreathingIndicator:
    """``push_audio`` per microphone chunk and ``push_frame`` per camera frame; both return the current reading."""

    def __init__(
        self,
        config: BreathingConfig = INDICATOR_SPEC.breathing,
        *,
        audio_allowed: bool = False,
        sample_rate_hz: int = 16000,
    ) -> None:
        self.config = config
        self.audio_allowed = audio_allowed
        self.sample_rate_hz = sample_rate_hz
        self.reset()

    def reset(self) -> None:
        self.analyzer = BreathingAnalyzer(sample_rate_hz=self.sample_rate_hz)
        self.visual = VisualRespirationTracker()
        self._chunks: deque[_Chunk] = deque()
        self._mic_live_since: float | None = None
        self._last_speech_t = -math.inf
        self._audio: IndicatorReading | None = None
        self._visual: IndicatorReading | None = None
        self.gasp_times: tuple[float, ...] = ()

    # ---- audio --------------------------------------------------------------------------------------------
    def push_audio(
        self, t_seconds: float, samples: np.ndarray, health: AudioHealthReport, *, driving: bool
    ) -> IndicatorReading:
        if not self.audio_allowed:
            self._audio = self._unavailable(t_seconds, BreathingUnavailable.MIC_BLOCKED_BY_POLICY)
            return self._audio
        if health.fault in _HARD_FAULTS or health.state is BreathingAudioState.UNAVAILABLE:
            self._mic_live_since = None
            self._chunks.clear()
            self.analyzer.push(t_seconds, samples, health)
            self._audio = self._unavailable(t_seconds, BreathingUnavailable.MIC_FAULT)
            return self._audio
        if self._mic_live_since is None:
            self._mic_live_since = t_seconds
        pcm = np.asarray(samples, dtype=np.float64).reshape(-1)
        rms = float(np.sqrt(np.mean(np.square(pcm)))) if pcm.size else 0.0
        self._chunks.append(_Chunk(t_seconds, rms, spectral_flatness(pcm), health.fault))
        keep = max(self.config.gasp_pattern_window_seconds, self.config.cabin_window_seconds)
        while self._chunks and self._chunks[0].t_seconds < t_seconds - keep:
            self._chunks.popleft()
        if health.fault is AudioFault.SPEECH:
            self._last_speech_t = t_seconds
        estimate = self.analyzer.push(t_seconds, pcm, health)
        self._audio = self._assess_audio(t_seconds, estimate, health, driving=driving)
        return self._audio

    def _assess_audio(
        self, t: float, estimate: BreathingEstimate, health: AudioHealthReport, *, driving: bool
    ) -> IndicatorReading:
        config = self.config
        floor = self._noise_floor(t)
        cabin = self._cabin_problem(t, floor)
        if cabin is not None:
            return self._unavailable(t, cabin)
        self.gasp_times = self._gasp_bursts(floor)
        agonal = self._agonal_pattern(t, estimate)
        if agonal is not None and not driving:
            return agonal
        if estimate.breaths_per_minute is not None:
            intervals = np.asarray(estimate.intervals_seconds, dtype=float)
            cv = float(np.std(intervals) / np.mean(intervals)) if intervals.size else 0.0
            state = IRREGULAR if cv >= config.irregular_interval_cv else REGULAR
            return IndicatorReading(
                name=IndicatorName.BREATHING,
                t_seconds=t,
                available=True,
                state=state,
                value=estimate.breaths_per_minute,
                severity=0.3 if state == IRREGULAR else 0.0,
                confidence=clamp01(estimate.confidence),
                evidence=(
                    f"About {estimate.breaths_per_minute:.0f} breaths per minute from cabin audio "
                    f"({'irregular' if state == IRREGULAR else 'regular'} intervals). Experimental.",
                ),
                unavailable_reason=None,
            )
        return self._silence(t, health, floor, driving=driving)

    def _silence(self, t: float, health: AudioHealthReport, floor: float, *, driving: bool) -> IndicatorReading:
        """No breath sounds: ``not_detected_quiet_cabin`` only when every quiet-cabin condition holds."""
        config = self.config
        reason: BreathingUnavailable | None = None
        if driving:
            reason = BreathingUnavailable.ROAD_NOISE
        elif self._mic_live_since is None or t - self._mic_live_since < config.mic_warmup_seconds:
            reason = BreathingUnavailable.MIC_WARMING_UP
        elif t - self._last_speech_t < config.speech_quiet_seconds:
            reason = BreathingUnavailable.SPEECH
        elif health.fault is not AudioFault.NONE or health.state is not BreathingAudioState.ACTIVE_RELIABLE:
            reason = _FAULT_REASON.get(health.fault, BreathingUnavailable.MIC_FAULT)
        elif floor >= config.quiet_noise_floor_rms:
            reason = BreathingUnavailable.ROAD_NOISE
        if reason is not None:
            return self._unavailable(t, reason)
        return IndicatorReading(
            name=IndicatorName.BREATHING,
            t_seconds=t,
            available=True,
            state=NOT_DETECTED_QUIET_CABIN,
            value=None,
            severity=0.5,
            confidence=0.4,
            evidence=(
                "No breathing sounds detected in a quiet, parked cabin. "
                "This alone never raises an alert. Experimental.",
            ),
            unavailable_reason=None,
        )

    def _noise_floor(self, t: float) -> float:
        recent = [chunk.rms for chunk in self._chunks if chunk.t_seconds > t - 2 * self.config.cabin_window_seconds]
        return float(np.median(recent)) if recent else 0.0

    def _cabin_problem(self, t: float, floor: float) -> BreathingUnavailable | None:
        """The dominant interference over the last ``cabin_window_seconds``, ignoring isolated gasp bursts."""
        config = self.config
        recent = [chunk for chunk in self._chunks if chunk.t_seconds > t - config.cabin_window_seconds]
        if not recent:
            return None
        steady = [chunk for chunk in recent if not self._is_burst(chunk, floor)]
        counts: dict[BreathingUnavailable, int] = {}
        for chunk in steady:
            reason = _FAULT_REASON.get(chunk.fault)
            if reason is not None:
                counts[reason] = counts.get(reason, 0) + 1
        if not counts:
            return None
        reason, count = max(counts.items(), key=lambda item: item[1])
        return reason if count / len(recent) >= config.cabin_fault_fraction else None

    def _is_burst(self, chunk: _Chunk, floor: float) -> bool:
        return (
            floor > 0
            and chunk.rms >= self.config.gasp_rms_ratio * floor
            and chunk.flatness >= self.config.gasp_min_flatness
        )

    def _gasp_bursts(self, floor: float) -> tuple[float, ...]:
        """Start times of runs of burst chunks no longer than ``gasp_max_seconds``."""
        bursts: list[float] = []
        run: list[_Chunk] = []
        for chunk in (*self._chunks, None):
            if chunk is not None and self._is_burst(chunk, floor):
                run.append(chunk)
                continue
            if run and run[-1].t_seconds - run[0].t_seconds <= self.config.gasp_max_seconds:
                bursts.append(run[0].t_seconds)
            run = []
        return tuple(bursts)

    def _agonal_pattern(self, t: float, estimate: BreathingEstimate) -> IndicatorReading | None:
        config = self.config
        recent = [when for when in self.gasp_times if when > t - config.gasp_pattern_window_seconds]
        if len(recent) < config.gasp_min_count or estimate.breaths_per_minute is not None:
            return None
        intervals = np.diff(np.asarray(recent, dtype=float))
        if intervals.size == 0 or float(np.min(intervals)) < config.gasp_min_interval_seconds:
            return None
        return IndicatorReading(
            name=IndicatorName.BREATHING,
            t_seconds=t,
            available=True,
            state=AGONAL_PATTERN_SUSPECTED,
            value=None,
            severity=0.8,
            confidence=0.35,
            evidence=(
                f"{len(recent)} isolated gasp-like sounds at long intervals with no regular breathing between them. "
                "Experimental; not validated on Driven data.",
            ),
            unavailable_reason=None,
        )

    # ---- visual ---------------------------------------------------------------------------------------------
    def push_frame(self, frame: FrameSignals) -> IndicatorReading:
        """Update the visual fallback and return the current combined reading."""
        t = frame.t_seconds
        if not frame.face:
            self.visual.reset()
            self._visual = self._unavailable(t, BreathingUnavailable.FACE_NOT_TRACKED)
        elif frame.shoulder_y is None or not math.isfinite(frame.shoulder_y):
            self.visual.reset()
            self._visual = self._unavailable(t, BreathingUnavailable.SHOULDERS_OUT_OF_FRAME)
        else:
            bpm = self.visual.push(t, frame.shoulder_y)
            self._visual = (
                self._unavailable(t, BreathingUnavailable.COLLECTING)
                if bpm is None
                else IndicatorReading(
                    name=IndicatorName.BREATHING,
                    t_seconds=t,
                    available=True,
                    state=REGULAR,
                    value=bpm,
                    severity=0.0,
                    confidence=0.3,
                    evidence=(f"About {bpm:.0f} breaths per minute from shoulder motion. Experimental.",),
                    unavailable_reason=None,
                )
            )
        return self.current(t)

    def current(self, t_seconds: float) -> IndicatorReading:
        """The audio reading while it is fresh and available, else the visual one, else the best reason why not."""
        audio = self._audio
        fresh_audio = (
            audio is not None
            and self.audio_allowed
            and t_seconds - audio.t_seconds <= self.config.audio_max_age_seconds
        )
        if fresh_audio and audio is not None and audio.available:
            return audio
        if self._visual is not None and self._visual.available:
            return self._visual
        if not self.audio_allowed:
            return self._unavailable(t_seconds, BreathingUnavailable.MIC_BLOCKED_BY_POLICY)
        if audio is None:
            return self._unavailable(t_seconds, BreathingUnavailable.MIC_WARMING_UP)
        if fresh_audio and audio.unavailable_reason is not None:
            return self._unavailable(t_seconds, BreathingUnavailable(audio.unavailable_reason))
        return self._unavailable(t_seconds, BreathingUnavailable.MIC_FAULT)  # the stream stopped sending chunks

    def _unavailable(self, t: float, reason: BreathingUnavailable) -> IndicatorReading:
        return unavailable_reading(IndicatorName.BREATHING, t, reason.value, (_MESSAGES[reason],))
