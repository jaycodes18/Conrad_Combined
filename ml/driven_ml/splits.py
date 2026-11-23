"""Split windows by person into development, a locked test set, and replay people.

Test people are chosen by a hash of their id, so nobody can reshuffle the split to get a better number. Replay
people (``replay_only = yes`` in consent.csv) are held out of both, so the replay gate scores faces the model
has never seen. A person's windows are only ever in one of the three sets. The binary task is alert (0)
against drowsy (2); low vigilance (1) is left out because its labels overlap both.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass

import pandas as pd

from .data import (
    TEAM_DATASET_KEY,
    dataset_key_of,
    is_allowed,
    load_consented_subjects,
    load_replay_only_subjects,
)
from .settings import DataSource, RunConfig
from .windows import ALERT_LABEL, DROWSY_LABEL
from .workspace import Workspace, read_parquet

logger = logging.getLogger(__name__)

TEST_BUCKETS = 5


@dataclass(frozen=True, eq=False)
class DatasetSplit:
    """Every person is in exactly one set. Replay people never reach training, validation or test."""

    development: pd.DataFrame
    test: pd.DataFrame
    replay: pd.DataFrame
    dropped_unlicensed_people: int


def is_locked_test_subject(subject_id: str) -> bool:
    digest = hashlib.sha256(subject_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % TEST_BUCKETS == 0


def _drop_unconsented_team_people(windows: pd.DataFrame, workspace: Workspace) -> pd.DataFrame:
    """Re-check consent here too, so reused features never keep a person who withdrew (L5)."""
    consented = load_consented_subjects(workspace.consent_path)
    is_team = windows["subject"].map(dataset_key_of) == TEAM_DATASET_KEY
    person = windows["subject"].str.split(":", n=1).str[-1]
    withdrawn = is_team & ~person.isin(consented)
    if withdrawn.any():
        logger.warning(
            "split: dropped %d team people without recorded consent (L5)",
            windows.loc[withdrawn, "subject"].nunique(),
        )
    return windows.loc[~withdrawn]


def split_windows(config: RunConfig, workspace: Workspace) -> DatasetSplit:
    windows = read_parquet(workspace.windows_path, produced_by="windows")
    dropped_people = 0
    if config.data_source is DataSource.REAL:
        windows = _drop_unconsented_team_people(windows, workspace)
        allowed = windows["subject"].map(lambda subject: is_allowed(dataset_key_of(subject), config.track))
        dropped_people = windows.loc[~allowed, "subject"].nunique()
        windows = windows.loc[allowed]
        if dropped_people:
            logger.info(
                "split: dropped %d people whose dataset the %s track may not use",
                dropped_people,
                config.track,
            )
    binary = windows.loc[windows["label"].isin((ALERT_LABEL, DROWSY_LABEL))].copy()
    binary["y"] = (binary["label"] == DROWSY_LABEL).astype(int)
    locked = binary["subject"].map(is_locked_test_subject).astype(bool)
    replay_people = replay_subjects(config, workspace)
    is_replay = binary["subject"].isin(replay_people) & ~locked
    development = binary.loc[~locked & ~is_replay].reset_index(drop=True)
    test = binary.loc[locked].reset_index(drop=True)
    replay = binary.loc[is_replay].reset_index(drop=True)
    _check_disjoint(development, test, replay)
    logger.info(
        "split: development %d windows from %d people, locked test %d windows from %d people, "
        "replay %d windows from %d people",
        len(development),
        development["subject"].nunique(),
        len(test),
        test["subject"].nunique(),
        len(replay),
        replay["subject"].nunique(),
    )
    return DatasetSplit(
        development=development,
        test=test,
        replay=replay,
        dropped_unlicensed_people=dropped_people,
    )


def replay_subjects(config: RunConfig, workspace: Workspace) -> set[str]:
    """Subject ids of replay people: consented team people marked ``replay_only = yes``. Locked test people stay
    in test even when marked, and their scripted sessions still count toward replay."""
    if config.data_source is not DataSource.REAL:
        return set()
    return {f"{TEAM_DATASET_KEY}:{person}" for person in load_replay_only_subjects(workspace.consent_path)}


def _check_disjoint(*sets: pd.DataFrame) -> None:
    people = [set(frame["subject"]) for frame in sets]
    for index, first in enumerate(people):
        for second in people[index + 1 :]:
            if first & second:
                raise AssertionError(f"people in two of development, test and replay: {sorted(first & second)}")
