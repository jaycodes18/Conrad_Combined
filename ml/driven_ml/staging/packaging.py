"""Package the candidate for staging: the model, its model card, the feature spec and parity fixtures.

The model card is the only text about the model that leaves this pipeline, so its wording is fixed here and
scanned for banned claims by the gate. The parity fixtures are rows from the locked test with the expected
probabilities, which any port of ``predict_probability`` must reproduce to 1e-6, missing values included.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..evaluation import TestReport, load_test_report
from ..modeling.artifact import ModelArtifact, predict_probability
from ..modeling.store import load_candidate
from ..spec import FEATURE_SPEC
from ..workspace import Workspace, write_json, write_text_atomic

logger = logging.getLogger(__name__)

PARITY_SAMPLE_SIZE = 50
PARITY_TOLERANCE = 1e-6
PARITY_GAPPED_ROWS = 15
PARITY_MAX_GAPS = 5
MIN_PARITY_ROWS_WITH_MISSING = PARITY_GAPPED_ROWS + 1  # the gapped rows plus the all-NaN row
MODEL_FILE = "model.json"
MODEL_CARD_FILE = "MODEL_CARD.md"
FEATURE_SPEC_FILE = "feature_spec.json"
PARITY_FILE = "parity_fixtures.json"

MODEL_CARD_TEMPLATE = """\
**What it does.** Scores 60-second windows of eye, head and mouth signals for signs of drowsiness,
compared with the driver's own alert baseline. It drives the in-car spoken alert and parent texts.

**What it does not do.** It does not measure alcohol or any other substance, does not identify
intoxication, and is not a medical device. It never replaces the rule of not driving after drinking.

**Data.** Track: {track}. Datasets: {datasets}.

**Results on {people} held-out people.**

| Model | AUC | Balanced accuracy | False in-car alerts per hour | Drowsy sessions caught |
|---|---|---|---|---|
| Driven | {driven_auc:.3f} | {driven_balanced:.3f} | {driven_false:.2f} | {driven_caught:.0%} |
| Fixed PERCLOS threshold (B1) | {fixed_auc:.3f} | {fixed_balanced:.3f} | {fixed_false:.2f} | {fixed_caught:.0%} |

**Replay.** Scripted, acted sessions are replayed for people the model never trained on.

**Limits.** Phone cameras lose eye detail in the dark and behind sunglasses; the app reports when it
cannot see the face. Results come from {people} test people and are re-checked during the pilot.
"""


@dataclass(frozen=True)
class PackagedCandidate:
    directory: Path
    artifact: ModelArtifact
    files: tuple[Path, ...]


def render_model_card(artifact: ModelArtifact, report: TestReport) -> str:
    return MODEL_CARD_TEMPLATE.format(
        track=artifact.track,
        datasets=", ".join(artifact.datasets),
        people=report.people,
        driven_auc=report.driven.auc,
        driven_balanced=report.driven.balanced_accuracy,
        driven_false=report.driven.false_alerts_per_hour,
        driven_caught=report.driven.drowsy_sessions_caught,
        fixed_auc=report.fixed_threshold.auc,
        fixed_balanced=report.fixed_threshold.balanced_accuracy,
        fixed_false=report.fixed_threshold.false_alerts_per_hour,
        fixed_caught=report.fixed_threshold.drowsy_sessions_caught,
    )


def parity_inputs(artifact: ModelArtifact, test: pd.DataFrame, seed: int) -> np.ndarray:
    """Up to 50 random locked-test rows in model input order, with gaps a port must fill.

    Missing values are kept. With the same generator, 15 of the rows get 1 to 5 more inputs set to NaN, and one
    row of nothing but NaN is added, so a port that skips ``fill_values`` cannot pass parity.
    """
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(test), size=min(PARITY_SAMPLE_SIZE, len(test)), replace=False)
    inputs = test[artifact.inputs].to_numpy(dtype=float)[rows]
    for row in rng.choice(len(inputs), size=min(PARITY_GAPPED_ROWS, len(inputs)), replace=False):
        missing = rng.choice(
            len(artifact.inputs),
            size=rng.integers(1, PARITY_MAX_GAPS + 1),
            replace=False,
        )
        inputs[row, missing] = np.nan
    return np.vstack([inputs, np.full((1, len(artifact.inputs)), np.nan)])


def rows_with_missing(inputs: np.ndarray) -> int:
    return int(np.count_nonzero(np.isnan(inputs).any(axis=1)))


def parity_document(artifact: ModelArtifact, inputs: np.ndarray) -> dict[str, object]:
    """Inputs, the reference probabilities and the tolerance; NaN is written as JSON null."""
    return {
        "inputs": inputs,
        "expected": predict_probability(artifact, inputs),
        "tolerance": PARITY_TOLERANCE,
        "missing_values": "null means NaN; apply fill_values before scaling",
        "rows_with_missing": rows_with_missing(inputs),
    }


def package_candidate(workspace: Workspace, test: pd.DataFrame, seed: int) -> PackagedCandidate:
    artifact = load_candidate(workspace)
    report = load_test_report(workspace, artifact)
    directory = workspace.staging_dir(artifact.short_sha)
    files = (
        write_json(directory / MODEL_FILE, artifact.to_json(), indent=None),
        write_text_atomic(directory / MODEL_CARD_FILE, render_model_card(artifact, report)),
        write_json(directory / FEATURE_SPEC_FILE, FEATURE_SPEC.to_json()),
        write_json(
            directory / PARITY_FILE,
            parity_document(artifact, parity_inputs(artifact, test, seed)),
            indent=None,
        ),
    )
    logger.info("package: staged %s in %s", artifact.short_sha, directory)
    return PackagedCandidate(directory=directory, artifact=artifact, files=files)
