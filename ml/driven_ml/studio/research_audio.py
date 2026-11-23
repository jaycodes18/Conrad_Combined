"""Gate for Studio research microphone analysis.

``record_audio`` in product.json stays false: raw audio is never written to disk.
Ephemeral mic analysis runs only when both ``studio_research_audio`` and
``legal_opinion_studio_audio`` are true.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def research_audio_allowed(flags: Mapping[str, Any] | None) -> bool:
    """Return True only when the research-audio opinion gate is fully open."""
    if flags is None:
        return False
    return (
        flags.get("record_audio") is False
        and flags.get("studio_research_audio") is True
        and flags.get("legal_opinion_studio_audio") is True
    )


def research_audio_block_reason(flags: Mapping[str, Any] | None) -> str:
    """Human-readable reason the mic path is closed."""
    if flags is None:
        return "product flags unavailable; research audio stays closed"
    if flags.get("record_audio") is True:
        return "record_audio must stay false; refuse mic start while raw recording would be enabled"
    if not flags.get("legal_opinion_studio_audio"):
        return "legal_opinion_studio_audio is false; studio research audio needs a written legal opinion first"
    if not flags.get("studio_research_audio"):
        return "studio_research_audio is false; enable it only after the legal opinion flag is set"
    return ""
