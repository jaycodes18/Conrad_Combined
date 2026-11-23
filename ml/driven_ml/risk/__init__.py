"""Risk: fusing indicator readings into one condition, mapping it to escalation tiers, and the app contract.

``fusion`` is the model, ``escalation`` the mapping from risk to recommended actions, ``signal`` the
``DriverRiskSignal`` v1 contract, and ``session`` runs all of them for one driving session.
"""

from __future__ import annotations

from .escalation import (
    EMERGENCY_SERVICES,
    ConsentState,
    Contact,
    ContactBook,
    EscalationConfig,
    EscalationDecision,
    EscalationEngine,
    TierEvent,
    TierEventKind,
)
from .fusion import Condition, FusionInputs, FusionResult, FusionWeights, RiskFusion, RiskLevel, fit_fusion
from .session import RiskSession
from .signal import DISCLAIMER, SCHEMA_VERSION, DriverRiskSignal

__all__ = [
    "DISCLAIMER",
    "EMERGENCY_SERVICES",
    "SCHEMA_VERSION",
    "Condition",
    "ConsentState",
    "Contact",
    "ContactBook",
    "DriverRiskSignal",
    "EscalationConfig",
    "EscalationDecision",
    "EscalationEngine",
    "FusionInputs",
    "FusionResult",
    "FusionWeights",
    "RiskFusion",
    "RiskLevel",
    "RiskSession",
    "TierEvent",
    "TierEventKind",
    "fit_fusion",
]
