"""Dataset registry and consent. Discovery is imported from ``driven_ml.data.discovery`` directly."""

from __future__ import annotations

from .consent import load_consented_subjects, load_replay_only_subjects
from .registry import (
    DATASETS,
    TEAM_DATASET_KEY,
    DatasetEntry,
    dataset_key_of,
    is_allowed,
    registry_table,
)

__all__ = [
    "DATASETS",
    "TEAM_DATASET_KEY",
    "DatasetEntry",
    "dataset_key_of",
    "is_allowed",
    "load_consented_subjects",
    "load_replay_only_subjects",
    "registry_table",
]
