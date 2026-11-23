"""One driving session end to end: indicators, fusion, escalation, and the ``DriverRiskSignal`` per frame.

Studio's live engine, the scripted-session tests and the golden contract fixtures all run this class, so they
run the same logic. Order within a frame: the indicators read the frame, fusion scores the readings, escalation
maps the result to a tier, and a Tier 1 alert opens the response check that the next frames evaluate. A move
into possible unresponsiveness during an episode that already had its Tier 1 alert opens a fresh check.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from ..indicators import BEHAVIOUR_INDICATORS, CameraHealth, FrameSignals, IndicatorReading, IndicatorSuite
from ..modeling.artifact import ModelArtifact
from ..settings import Track
from ..spec import FEATURE_SPEC, INDICATOR_SPEC, IndicatorSpec
from ..studio.audio_health import AudioHealthReport
from .escalation import ContactBook, EscalationConfig, EscalationDecision, EscalationEngine
from .fusion import LEVEL_ORDER, Condition, FusionInputs, FusionResult, RiskFusion, RiskLevel
from .signal import DriverRiskSignal, ModelInfo, build_signal

# No indicator has been validated on Driven-recorded data; each is reported as experimental until one is.
EXPERIMENTAL_INDICATORS: tuple[str, ...] = tuple(name.value for name in BEHAVIOUR_INDICATORS)


class RiskSession:
    def __init__(
        self,
        *,
        session_id: str,
        driver_ref: str,
        escalation: EscalationConfig,
        contacts: ContactBook,
        audio_allowed: bool = False,
        model: ModelArtifact | None = None,
        spec: IndicatorSpec = INDICATOR_SPEC,
    ) -> None:
        self.session_id = session_id
        self.driver_ref = driver_ref
        self.suite = IndicatorSuite(spec, audio_allowed=audio_allowed)
        self.fusion = RiskFusion(spec.fusion)
        self.escalation = EscalationEngine(escalation, contacts)
        self.model: ModelArtifact | None = None
        self._probability: tuple[float, float, float | None] | None = None
        self.last_signal: DriverRiskSignal | None = None
        self.last_result: FusionResult | None = None
        self.last_decision: EscalationDecision | None = None
        self.set_model(model)

    def set_model(self, model: ModelArtifact | None) -> None:
        self.model = model
        self._probability = None

    @property
    def model_info(self) -> ModelInfo:
        model = self.model
        return ModelInfo(
            feature_spec_version=FEATURE_SPEC.version if model is None else model.feature_spec_version,
            artifact_hash=None if model is None else model.sha256,
            track=Track.RESEARCH.value if model is None else model.track,
            experimental=EXPERIMENTAL_INDICATORS,
        )

    def set_probability(self, probability: float, t_seconds: float, threshold: float | None = None) -> None:
        """The 60 s drowsiness model's latest window score, timestamped at the window's end, and its threshold."""
        self._probability = (probability, t_seconds, threshold)

    def clear_probability(self) -> None:
        self._probability = None

    def reset(self) -> None:
        """Recalibrate: indicators and fusion start over; the escalation history and text cap carry on."""
        self.suite.reset()
        self.fusion.reset()
        self._probability = None

    def push_audio(
        self, t_seconds: float, samples: np.ndarray, health: AudioHealthReport, *, driving: bool
    ) -> IndicatorReading:
        return self.suite.push_audio(t_seconds, samples, health, driving=driving)

    def push_frame(
        self,
        frame: FrameSignals,
        camera: CameraHealth,
        *,
        calibrating: bool,
        driving: bool,
        emitted_at: datetime,
        eyes_off_seconds: float | None = None,
    ) -> DriverRiskSignal:
        readings = self.suite.push_frame(frame, camera, calibrating=calibrating)
        away = self.suite.eyes_off_seconds if eyes_off_seconds is None else eyes_off_seconds
        probability, probability_t, threshold = (
            self._probability if self._probability is not None else (None, None, None)
        )
        if threshold is None and self.model is not None:
            threshold = self.model.threshold
        quality = self.suite.sensor_quality
        result = self.fusion.assess(
            FusionInputs(
                t_seconds=frame.t_seconds,
                readings=readings,
                driving=driving,
                coverage=quality.coverage,
                situations=quality.situations,
                drowsiness_probability=probability,
                probability_t_seconds=probability_t,
                probability_threshold=threshold,
                eyes_off_seconds=away,
            )
        )
        responsiveness = self.suite.responsiveness
        decision = self.escalation.step(result, responsiveness.state)
        if decision.open_response_window or self._newly_unresponsive(result):
            responsiveness.open_window(frame.t_seconds)
        signal = build_signal(
            session_id=self.session_id,
            driver_ref=self.driver_ref,
            emitted_at=emitted_at,
            result=result,
            readings=readings,
            coverage=quality.coverage,
            decision=decision,
            responsiveness_check=responsiveness.state,
            model=self.model_info,
            eyes_off_seconds=away,
        )
        self.last_result, self.last_decision, self.last_signal = result, decision, signal
        return signal

    def _newly_unresponsive(self, result: FusionResult) -> bool:
        """A move into possible unresponsiveness mid-episode needs its own response check, not the earlier one."""
        previous = self.last_result
        entering = result.condition is Condition.POSSIBLE_UNRESPONSIVENESS and (
            previous is None or previous.condition is not Condition.POSSIBLE_UNRESPONSIVENESS
        )
        return entering and LEVEL_ORDER[result.level] >= LEVEL_ORDER[RiskLevel.ELEVATED]
