"""``DriverRiskSignal`` v1: the one message the app consumes from the ML side, once per assessment.

The JSON Schema is ``ml/contracts/driver_risk_signal.v1.schema.json`` and the TypeScript type is
``ml/contracts/driver_risk_signal.v1.ts``; ``tests/test_contract.py`` keeps all three in step with the golden
fixtures. The signal never carries pixels, landmarks, audio, transcripts or GPS coordinates. Location belongs to
the app: ``share_location = "precise"`` tells the app this event warrants sending the contacts the phone's current
location, which the app reads under its own permission flow.

``app_alert`` carries the app's existing alert fields unchanged (``cue``, ``level``, ``evidence_ms``, as in
``src/risk.ts`` ``RiskAlert`` and the ``guardian_test_alerts`` table), so the app can map a signal onto what it
already stores without renaming anything.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..errors import DataError
from ..indicators import BEHAVIOUR_INDICATORS, Coverage, IndicatorName, IndicatorReading
from ..indicators.eyes import CLOSED_STATES
from ..indicators.head import LOST_CONTROL_STATES
from .escalation import EMERGENCY_SERVICES, EscalationDecision
from .fusion import Condition, FusionResult, RiskLevel

SCHEMA_VERSION = "driver_risk_signal.v1"
DRIVER_REF = re.compile(r"[A-Za-z0-9_-]{1,64}")
DISCLAIMER = (
    "Driven reports observable driving-risk indicators. It does not measure alcohol or identify any medical condition."
)
RESPONSIVENESS_CHECKS = frozenset({"idle", "waiting", "responded", "no_response", "unavailable"})
APP_CUES: dict[Condition, str] = {
    Condition.DROWSY: "eye-closure",
    Condition.POSSIBLE_UNRESPONSIVENESS: "eye-closure",
    Condition.DISTRACTED: "head-away",
}


@dataclass(frozen=True)
class IndicatorSnapshot:
    available: bool
    state: str
    value: float | None
    severity: float
    confidence: float
    evidence: tuple[str, ...]
    unavailable_reason: str | None

    @classmethod
    def of(cls, reading: IndicatorReading) -> IndicatorSnapshot:
        return cls(
            available=reading.available,
            state=reading.state,
            value=None if reading.value is None else round(reading.value, 4),
            severity=round(reading.severity, 4),
            confidence=round(reading.confidence, 4),
            evidence=reading.evidence,
            unavailable_reason=reading.unavailable_reason,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "state": self.state,
            "value": self.value,
            "severity": self.severity,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "unavailable_reason": self.unavailable_reason,
        }

    @classmethod
    def from_json(cls, document: Mapping[str, Any]) -> IndicatorSnapshot:
        return cls(
            available=bool(document["available"]),
            state=str(document["state"]),
            value=None if document["value"] is None else float(document["value"]),
            severity=float(document["severity"]),
            confidence=float(document["confidence"]),
            evidence=tuple(str(line) for line in document["evidence"]),
            unavailable_reason=None if document["unavailable_reason"] is None else str(document["unavailable_reason"]),
        )


@dataclass(frozen=True)
class MonitoringDegraded:
    active: bool
    situations: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {"active": self.active, "situations": list(self.situations)}


@dataclass(frozen=True)
class EscalationAdvice:
    tier: int
    in_car_alert: bool
    responsiveness_check: str
    notify_contacts: bool
    share_location: str
    voice_call_recommended: bool
    emergency_services: str
    reason: str

    def to_json(self) -> dict[str, Any]:
        return {
            "tier": self.tier,
            "in_car_alert": self.in_car_alert,
            "responsiveness_check": self.responsiveness_check,
            "notify_contacts": self.notify_contacts,
            "share_location": self.share_location,
            "voice_call_recommended": self.voice_call_recommended,
            "emergency_services": self.emergency_services,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ModelInfo:
    feature_spec_version: int
    artifact_hash: str | None
    track: str
    experimental: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "feature_spec_version": self.feature_spec_version,
            "artifact_hash": self.artifact_hash,
            "track": self.track,
            "experimental": list(self.experimental),
        }


@dataclass(frozen=True)
class AppAlert:
    """The app's own alert fields: ``cue`` and ``level`` are null when there is nothing to alert."""

    cue: str | None
    level: int | None
    evidence_ms: int | None

    def to_json(self) -> dict[str, Any]:
        return {"cue": self.cue, "level": self.level, "evidence_ms": self.evidence_ms}


@dataclass(frozen=True)
class DriverRiskSignal:
    session_id: str
    driver_ref: str
    t_seconds: float
    emitted_at: str
    condition: Condition
    level: RiskLevel
    score: float
    confidence: float
    indicators: Mapping[IndicatorName, IndicatorSnapshot]
    coverage: Mapping[IndicatorName, Coverage]
    monitoring_degraded: MonitoringDegraded
    escalation: EscalationAdvice
    model: ModelInfo
    app_alert: AppAlert
    schema_version: str = SCHEMA_VERSION
    disclaimer: str = DISCLAIMER

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.disclaimer != DISCLAIMER:
            raise DataError("schema_version and disclaimer are fixed for v1")
        if self.escalation.emergency_services != EMERGENCY_SERVICES:
            raise DataError("emergency_services must be never_automated")
        if self.escalation.responsiveness_check not in RESPONSIVENESS_CHECKS:
            raise DataError(f"unknown responsiveness_check {self.escalation.responsiveness_check!r}")
        if not math.isfinite(self.t_seconds) or not (0.0 <= self.score <= 1.0 and 0.0 <= self.confidence <= 1.0):
            raise DataError("t_seconds must be finite, and score and confidence between 0 and 1")
        if not DRIVER_REF.fullmatch(self.driver_ref):
            raise DataError("driver_ref must be a pseudonymous id (letters, digits, _ or -), never a name or an email")

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "driver_ref": self.driver_ref,
            "t_seconds": self.t_seconds,
            "emitted_at": self.emitted_at,
            "condition": self.condition.value,
            "level": self.level.value,
            "score": self.score,
            "confidence": self.confidence,
            "indicators": {name.value: snapshot.to_json() for name, snapshot in self.indicators.items()},
            "coverage": {name.value: value.value for name, value in self.coverage.items()},
            "monitoring_degraded": self.monitoring_degraded.to_json(),
            "escalation": self.escalation.to_json(),
            "model": self.model.to_json(),
            "app_alert": self.app_alert.to_json(),
            "disclaimer": self.disclaimer,
        }

    @classmethod
    def from_json(cls, document: Mapping[str, Any]) -> DriverRiskSignal:
        try:
            escalation, model = document["escalation"], document["model"]
            degraded, app = document["monitoring_degraded"], document["app_alert"]
            return cls(
                schema_version=str(document["schema_version"]),
                session_id=str(document["session_id"]),
                driver_ref=str(document["driver_ref"]),
                t_seconds=float(document["t_seconds"]),
                emitted_at=str(document["emitted_at"]),
                condition=Condition(document["condition"]),
                level=RiskLevel(document["level"]),
                score=float(document["score"]),
                confidence=float(document["confidence"]),
                indicators={
                    IndicatorName(name): IndicatorSnapshot.from_json(item)
                    for name, item in document["indicators"].items()
                },
                coverage={IndicatorName(name): Coverage(value) for name, value in document["coverage"].items()},
                monitoring_degraded=MonitoringDegraded(
                    active=bool(degraded["active"]), situations=tuple(str(item) for item in degraded["situations"])
                ),
                escalation=EscalationAdvice(
                    tier=int(escalation["tier"]),
                    in_car_alert=bool(escalation["in_car_alert"]),
                    responsiveness_check=str(escalation["responsiveness_check"]),
                    notify_contacts=bool(escalation["notify_contacts"]),
                    share_location=str(escalation["share_location"]),
                    voice_call_recommended=bool(escalation["voice_call_recommended"]),
                    emergency_services=str(escalation["emergency_services"]),
                    reason=str(escalation["reason"]),
                ),
                model=ModelInfo(
                    feature_spec_version=int(model["feature_spec_version"]),
                    artifact_hash=None if model["artifact_hash"] is None else str(model["artifact_hash"]),
                    track=str(model["track"]),
                    experimental=tuple(str(item) for item in model["experimental"]),
                ),
                app_alert=AppAlert(
                    cue=None if app["cue"] is None else str(app["cue"]),
                    level=None if app["level"] is None else int(app["level"]),
                    evidence_ms=None if app["evidence_ms"] is None else int(app["evidence_ms"]),
                ),
                disclaimer=str(document["disclaimer"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DataError(f"not a valid {SCHEMA_VERSION} document: {exc!r}") from exc


def iso_utc(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _app_alert(result: FusionResult, decision: EscalationDecision, eyes_off_seconds: float) -> AppAlert:
    """Map onto the app's ``RiskAlert``: cue by condition, level = escalation tier, evidence = how long it lasted."""
    cue = APP_CUES.get(result.condition)
    eyes = result.readings.get(IndicatorName.EYES)
    if result.condition is Condition.POSSIBLE_IMPAIRMENT:
        cue = "eye-closure" if eyes is not None and eyes.state in CLOSED_STATES else "head-away"
    if decision.tier == 0 or cue is None:
        return AppAlert(cue=None, level=None, evidence_ms=None)
    durations = [eyes_off_seconds] if cue == "head-away" else []
    durations += [
        reading.value or 0.0
        for reading in result.readings.values()
        if reading.state in CLOSED_STATES or reading.state in LOST_CONTROL_STATES
    ]
    return AppAlert(cue=cue, level=min(decision.tier, 3), evidence_ms=round(max(durations, default=0.0) * 1000))


def build_signal(
    *,
    session_id: str,
    driver_ref: str,
    emitted_at: datetime,
    result: FusionResult,
    readings: Mapping[IndicatorName, IndicatorReading],
    coverage: Mapping[IndicatorName, Coverage],
    decision: EscalationDecision,
    responsiveness_check: str,
    model: ModelInfo,
    eyes_off_seconds: float = 0.0,
) -> DriverRiskSignal:
    ordered = {name: IndicatorSnapshot.of(readings[name]) for name in IndicatorName if name in readings}
    return DriverRiskSignal(
        session_id=session_id,
        driver_ref=driver_ref,
        t_seconds=round(result.t_seconds, 3),
        emitted_at=iso_utc(emitted_at),
        condition=result.condition,
        level=result.level,
        score=round(result.score, 4),
        confidence=round(result.confidence, 4),
        indicators=ordered,
        coverage={name: coverage.get(name, Coverage.UNAVAILABLE) for name in BEHAVIOUR_INDICATORS},
        monitoring_degraded=MonitoringDegraded(
            active=result.monitoring_degraded,
            situations=tuple(situation.value for situation in result.degraded_situations),
        ),
        escalation=EscalationAdvice(
            tier=decision.tier,
            in_car_alert=decision.in_car_alert,
            responsiveness_check=responsiveness_check,
            notify_contacts=decision.notify_contacts,
            share_location=decision.share_location,
            voice_call_recommended=decision.voice_call_recommended,
            emergency_services=decision.emergency_services,
            reason=decision.reason,
        ),
        model=model,
        app_alert=_app_alert(result, decision, eyes_off_seconds),
    )
