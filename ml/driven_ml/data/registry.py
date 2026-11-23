"""The reviewed dataset registry: what each dataset may be used for, and on which track.

The research track may use any dataset not marked ``never_use``. The commercial track, the only one that can
ship, may use a dataset only when its license clearly allows commercial use (L9). Scraped or stock footage is
never used for training on any track (L10).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from ..settings import Track

TEAM_DATASET_KEY = "team_recordings"


def dataset_key_of(subject_id: str) -> str:
    """Subject ids are ``<dataset key>:<person>``; synthetic dev people have no prefix."""
    return subject_id.split(":", 1)[0]


@dataclass(frozen=True)
class DatasetEntry:
    key: str
    name: str
    license: str
    commercial_use: bool | None
    labels: str
    url: str
    intended_use: str
    never_use: bool = False


DATASETS: dict[str, DatasetEntry] = {
    entry.key: entry
    for entry in (
        DatasetEntry(
            "team_recordings",
            "Driven parked-car recordings",
            "own, consented",
            True,
            "scripted events",
            "recordings/ folder with consent.csv",
            "Replay tests, baseline tuning, commercial training",
        ),
        DatasetEntry(
            "hf_closed_open_eyes",
            "closed-open-eyes",
            "odc-by",
            True,
            "eye open/closed images",
            "https://huggingface.co/datasets/MichalMlodawski/closed-open-eyes",
            "Eye-state check of landmark features; verify image provenance first",
        ),
        DatasetEntry(
            "uta_rldd",
            "UTA Real-Life Drowsiness Dataset",
            "research; cite CVPRW 2019",
            None,
            "alert / low vigilance / drowsy, 60 people",
            "https://sites.google.com/view/utarldd/home",
            "Research track: main drowsiness benchmark",
        ),
        DatasetEntry(
            "kaggle_ddd",
            "Driver Drowsiness Dataset (frames from UTA-RLDD)",
            "inherits UTA-RLDD",
            None,
            "drowsy / not drowsy frames",
            "https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd",
            "Research track only; same people as UTA-RLDD, never mix across splits",
        ),
        DatasetEntry(
            "nitymed",
            "NITYMED night-time drivers",
            "free with citation; commercial use not stated",
            None,
            "yawning, microsleep, 21 drivers at night",
            "https://ieee-dataport.org/documents/nitymed",
            "Night-time test set; ask the authors before commercial use",
        ),
        DatasetEntry(
            "dmd",
            "Driver Monitoring Dataset",
            "CC BY-NC-ND 4.0",
            False,
            "distraction, gaze, drowsiness, 37 drivers",
            "https://dmd.vicomtech.org/",
            "Research track only (distraction, gaze zones)",
        ),
        DatasetEntry(
            "yawdd",
            "YawDD",
            "research use",
            False,
            "yawning, talking, 29+ drivers",
            "https://ieee-dataport.org/open-access/yawdd-yawning-detection-dataset",
            "Research track only (yawn detection)",
        ),
        DatasetEntry(
            "ul_dd",
            "UL-DD",
            "research use license, restricted",
            None,
            "continuous drowsiness and biosignals, 19 people",
            "https://zenodo.org/records/17978727",
            "Research track if access is granted",
        ),
        DatasetEntry(
            "state_farm",
            "State Farm Distracted Driver",
            "competition rules",
            False,
            "10 distraction classes (images)",
            "https://www.kaggle.com/c/state-farm-distracted-driver-detection",
            "Not for the product",
            never_use=True,
        ),
        DatasetEntry(
            "dif",
            "Dataset of Intoxicated Faces",
            "scraped from YouTube",
            False,
            "perceived drunk / sober",
            "https://arxiv.org/abs/1805.10030",
            "Not for any track (L10)",
            never_use=True,
        ),
        DatasetEntry(
            "stock_footage",
            "Stock video of acted drunkenness",
            "site terms",
            False,
            "none",
            "https://help.pexels.com/hc/de-de/articles/900005880463",
            "Pitch-video demo only, with a license that covers it; never training (L10)",
            never_use=True,
        ),
    )
}


def is_allowed(key: str, track: Track) -> bool:
    entry = DATASETS.get(key)
    if entry is None or entry.never_use:
        return False
    if track is Track.RESEARCH:
        return True
    return entry.commercial_use is True


def registry_table() -> pd.DataFrame:
    """Every registry entry, plus which track may use it, for the documentation notebook."""
    rows = [
        {
            **asdict(entry),
            "research_track": is_allowed(entry.key, Track.RESEARCH),
            "commercial_track": is_allowed(entry.key, Track.COMMERCIAL),
        }
        for entry in DATASETS.values()
    ]
    return pd.DataFrame(rows)
