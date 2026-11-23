"""Consent records for the team's own recordings (L5).

A person's recordings are used only when they were given the biometric notice, signed consent on a real date,
and said clearly whether they are a minor; a minor also needs a guardian's dated consent. Anything unclear is
refused, never guessed.
"""

from __future__ import annotations

import logging
from collections.abc import Hashable, Mapping
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from ..errors import DataError

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS: tuple[str, ...] = (
    "subject_id",
    "biometric_notice_given",
    "subject_consent_date",
    "is_minor",
    "guardian_consent_date",
)


def _answer(value: Any) -> str:
    return str(value).strip().lower()


def _is_date(value: Any) -> bool:
    try:
        date.fromisoformat(str(value).strip())
    except ValueError:
        return False
    return True


def _is_consented(record: Mapping[Hashable, Any]) -> bool:
    """Fail closed: anything other than a clear yes, a real date, and a clear answer on minors is refused."""
    if _answer(record["biometric_notice_given"]) != "yes" or not _is_date(record["subject_consent_date"]):
        return False
    is_minor = _answer(record["is_minor"])
    if is_minor == "no":
        return True
    return is_minor == "yes" and _is_date(record["guardian_consent_date"])


def load_consented_subjects(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise DataError(f"{path} is missing columns: {', '.join(missing)}")
    consented, refused = set(), []
    for record in table.to_dict(orient="records"):
        subject = str(record["subject_id"]).strip()
        if _is_consented(record):
            consented.add(subject)
        else:
            refused.append(subject)
    if refused:
        logger.warning(
            "consent: not usable for %s (needs notice 'yes', ISO dates, is_minor yes/no)",
            refused,
        )
    return consented


REPLAY_ONLY_COLUMN = "replay_only"


def load_replay_only_subjects(path: Path) -> set[str]:
    """Consented people marked ``replay_only = yes``: held out of training, validation and test (see
    ml/RECORDING_PROTOCOL.md). The column is optional; without it nobody is replay-only.
    """
    if not path.is_file():
        return set()
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    if REPLAY_ONLY_COLUMN not in table.columns:
        return set()
    marked = {
        str(row["subject_id"]).strip()
        for row in table.to_dict(orient="records")
        if _answer(row[REPLAY_ONLY_COLUMN]) == "yes"
    }
    return marked & load_consented_subjects(path)
