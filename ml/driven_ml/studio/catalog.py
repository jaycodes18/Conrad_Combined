"""Dataset cards and a human approval step. Discovery still does not download anything.

A card records what the registry already knows, plus the fields a reviewer needs before ingestion.
Counts that this session did not verify are marked unknown. Approval is written only after a named
person decides, and a `never_use` dataset cannot be approved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from ..data.registry import DATASETS, DatasetEntry
from ..errors import DataError
from ..workspace import write_text_atomic

KAGGLE_USERNAME_ENV = "KAGGLE_USERNAME"
KAGGLE_KEY_ENV = "KAGGLE_KEY"


class Approval(StrEnum):
    NOT_REVIEWED = "not_reviewed"
    APPROVED_RESEARCH = "approved_research"
    APPROVED_COMMERCIAL = "approved_commercial"
    REJECTED = "rejected"


class CommercialUse(StrEnum):
    ALLOWED = "allowed"
    FORBIDDEN = "forbidden"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class DatasetCard:
    key: str
    name: str
    source: str
    download_location: str
    license: str
    commercial_use: CommercialUse
    subjects: str
    images_or_videos: str
    annotation_types: str
    label_quality: str
    camera_perspective: str
    lighting: str
    recommended_role: str
    limitations: str
    never_use: bool

    def to_json(self, approval: Approval, reviewer: str) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "source": self.source,
            "download_location": self.download_location,
            "license": self.license,
            "commercial_use": self.commercial_use.value,
            "subjects": self.subjects,
            "images_or_videos": self.images_or_videos,
            "annotation_types": self.annotation_types,
            "label_quality": self.label_quality,
            "camera_perspective": self.camera_perspective,
            "lighting": self.lighting,
            "recommended_role": self.recommended_role,
            "limitations": self.limitations,
            "never_use": self.never_use,
            "approval": approval.value,
            "reviewer": reviewer,
            "ingested_this_session": False,
        }


def _use(entry: DatasetEntry) -> CommercialUse:
    if entry.commercial_use is True:
        return CommercialUse.ALLOWED
    if entry.commercial_use is False:
        return CommercialUse.FORBIDDEN
    return CommercialUse.UNKNOWN


def _card(entry: DatasetEntry, **extra: str) -> DatasetCard:
    return DatasetCard(
        key=entry.key,
        name=entry.name,
        source=extra.get("source", entry.url),
        download_location=entry.url,
        license=entry.license,
        commercial_use=_use(entry),
        subjects=extra.get("subjects", "not counted in this session"),
        images_or_videos=extra.get("images_or_videos", "not counted in this session"),
        annotation_types=entry.labels,
        label_quality=extra.get("label_quality", "not re-graded in this session"),
        camera_perspective=extra.get("camera_perspective", "see the dataset card; not re-verified here"),
        lighting=extra.get("lighting", "not measured in this session"),
        recommended_role=extra.get("recommended_role", entry.intended_use),
        limitations=extra.get("limitations", entry.intended_use),
        never_use=entry.never_use,
    )


CARDS: dict[str, DatasetCard] = {entry.key: _card(entry) for entry in DATASETS.values()}
CARDS["uta_rldd"] = _card(
    DATASETS["uta_rldd"],
    source="University of Texas at Arlington, cited from the registry",
    subjects="60 people (registry label)",
    images_or_videos="video; file count not downloaded this session",
    label_quality="alert / low vigilance / drowsy, research labels",
    camera_perspective="driver-facing, real driving",
    lighting="daytime real driving; not re-measured here",
    recommended_role="Research benchmark for drowsiness; best initial candidate for driver-held-out evaluation.",
    limitations="Research license is not a commercial grant. Do not mix with Kaggle DDD across splits.",
)
CARDS["kaggle_ddd"] = _card(
    DATASETS["kaggle_ddd"],
    source="Kaggle mirror of UTA-RLDD frames",
    subjects="same people as UTA-RLDD; do not treat as new subjects",
    images_or_videos="frames; not downloaded this session",
    label_quality="inherits UTA-RLDD; frame labels can leak the video",
    camera_perspective="driver-facing",
    lighting="inherits UTA-RLDD",
    recommended_role="Convenient research mirror of UTA-RLDD; never combine people across the original and mirror.",
    limitations="Competition-style mirrors are not a new license. Research track only until a human approves.",
)
CARDS["state_farm"] = _card(
    DATASETS["state_farm"],
    source="Kaggle competition, State Farm",
    subjects="not used; competition rules forbid this product use",
    images_or_videos="still images; not downloaded",
    label_quality="10 distraction classes; not drowsiness and not intoxication",
    camera_perspective="cabin side view, stills",
    lighting="daylight competition images",
    recommended_role="Do not use for this product: competition terms prohibit product training.",
    limitations="never_use. A drinking-action class would not be proof of intoxication.",
)
CARDS["dif"] = _card(
    DATASETS["dif"],
    limitations="Scraped footage. Perceived drunk is not a measured blood-alcohol label. never_use.",
)


@dataclass(frozen=True)
class ReviewDecision:
    key: str
    approval: Approval
    reviewer: str
    note: str

    def to_json(self) -> dict[str, str]:
        return {
            "key": self.key,
            "approval": self.approval.value,
            "reviewer": self.reviewer,
            "note": self.note,
        }


class ApprovalStore:
    """Human approvals on disk. Missing file means nothing has been approved."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._decisions: dict[str, ReviewDecision] = {}
        if path.is_file():
            self._load()

    def approval_for(self, key: str) -> Approval:
        decision = self._decisions.get(key)
        return Approval.NOT_REVIEWED if decision is None else decision.approval

    def reviewer_for(self, key: str) -> str:
        decision = self._decisions.get(key)
        return "" if decision is None else decision.reviewer

    def review(self, key: str, approval: Approval, reviewer: str, note: str) -> ReviewDecision:
        if key not in CARDS:
            raise DataError(f"unknown dataset {key}")
        if not reviewer.strip():
            raise DataError("a named reviewer is required")
        card = CARDS[key]
        if card.never_use and approval is not Approval.REJECTED:
            raise DataError(f"{key} is never_use and cannot be approved")
        if approval is Approval.APPROVED_COMMERCIAL and card.commercial_use is not CommercialUse.ALLOWED:
            raise DataError(f"{key} has no commercial license and cannot be approved for the commercial track")
        decision = ReviewDecision(key, approval, reviewer.strip(), note.strip())
        self._decisions[key] = decision
        self._save()
        return decision

    def table(self) -> list[dict[str, Any]]:
        return [CARDS[key].to_json(self.approval_for(key), self.reviewer_for(key)) for key in CARDS]

    def ingest_allowed(self, key: str) -> bool:
        approval = self.approval_for(key)
        card = CARDS.get(key)
        if card is None or card.never_use:
            return False
        return approval in {Approval.APPROVED_RESEARCH, Approval.APPROVED_COMMERCIAL}

    def _load(self) -> None:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise DataError(f"{self.path} must be a list of review decisions")
        for item in raw:
            approval = Approval(str(item["approval"]))
            self._decisions[str(item["key"])] = ReviewDecision(
                str(item["key"]), approval, str(item["reviewer"]), str(item.get("note", ""))
            )

    def _save(self) -> None:
        payload = json.dumps([item.to_json() for item in self._decisions.values()], indent=2)
        write_text_atomic(self.path, payload + "\n")


def kaggle_credential_status() -> dict[str, bool | str]:
    """Report whether credential env vars are set. Never return the values."""
    import os

    return {
        "username_env": KAGGLE_USERNAME_ENV,
        "key_env": KAGGLE_KEY_ENV,
        "username_set": bool(os.environ.get(KAGGLE_USERNAME_ENV)),
        "key_set": bool(os.environ.get(KAGGLE_KEY_ENV)),
        "values_exposed": False,
    }


def ingest_plan(key: str, store: ApprovalStore) -> dict[str, Any]:
    """Describe the download. This function does not download."""
    if key not in CARDS:
        raise DataError(f"unknown dataset {key}")
    card = CARDS[key]
    allowed = store.ingest_allowed(key)
    reason = "approved" if allowed else "not approved, license blocked, or never_use"
    return {
        "key": key,
        "download_location": card.download_location,
        "allowed": allowed,
        "executed": False,
        "reason": reason,
        "credential_env": [KAGGLE_USERNAME_ENV, KAGGLE_KEY_ENV],
        "note": "Run a download only after this plan says allowed. Credentials stay in the environment.",
    }
