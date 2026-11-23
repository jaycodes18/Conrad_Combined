"""Staging: package the candidate, replay scripted sessions and run the release gate."""

from __future__ import annotations

from .gate import GateCheck, StagingManifest, load_manifest, run_stage_gate
from .packaging import package_candidate, parity_inputs
from .replay import (
    ReplayResult,
    replay_recordings,
    replay_session,
    scripted_synthetic_session,
)

__all__ = [
    "GateCheck",
    "ReplayResult",
    "StagingManifest",
    "load_manifest",
    "package_candidate",
    "parity_inputs",
    "replay_recordings",
    "replay_session",
    "run_stage_gate",
    "scripted_synthetic_session",
]
