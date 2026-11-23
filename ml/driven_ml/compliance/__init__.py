"""Compliance gate: the team's legal rules as data, and the checks that enforce them."""

from __future__ import annotations

from .checks import (
    Finding,
    check_product_flags,
    enforce_compliance,
    scan_claims,
    scan_repository,
    scan_source,
)
from .rules import GATED_SETTINGS, LEGAL_RULES, GatedSetting, LegalRule

__all__ = [
    "GATED_SETTINGS",
    "LEGAL_RULES",
    "Finding",
    "GatedSetting",
    "LegalRule",
    "check_product_flags",
    "enforce_compliance",
    "scan_claims",
    "scan_repository",
    "scan_source",
]
