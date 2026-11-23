"""Score the candidate once on the locked test people.

The test set is the only place the model meets these people. Every run is logged, but only the first result
for a model hash is kept as its test report: rerunning the test cannot replace a bad number with a lucky one.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .errors import DataError
from .modeling.alert_simulation import alert_metrics
from .modeling.artifact import ModelArtifact, predict_probability
from .modeling.candidates import fixed_threshold_alerts
from .modeling.store import load_candidate
from .scoring import (
    NO_RECORDINGS,
    ScopeScores,
    WithinRecording,
    log_scopes,
    scores_by_scope,
    within_recording,
)
from .windows import BASELINE_SCOPE_SESSION
from .workspace import Workspace, read_json, write_json

logger = logging.getLogger(__name__)

_FIXED_THRESHOLD_DECISION = 0.5


@dataclass(frozen=True)
class ModelScores:
    auc: float
    balanced_accuracy: float
    false_alerts_per_hour: float
    drowsy_sessions_caught: float


@dataclass(frozen=True)
class TestReport:
    __test__ = False

    model_sha256: str
    people: int
    windows: int
    driven: ModelScores
    fixed_threshold: ModelScores
    prior_runs: int
    by_scope: list[ScopeScores]
    within_recording: WithinRecording
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    @classmethod
    def from_json(cls, document: Mapping[str, Any]) -> TestReport:
        """Reports written before per-scope metrics existed load with none: the first result for a model hash
        stands, so an old report is never rewritten, and a missing within-recording result fails that gate.
        """
        try:
            return cls(
                **{
                    **document,
                    "driven": ModelScores(**document["driven"]),
                    "fixed_threshold": ModelScores(**document["fixed_threshold"]),
                    "by_scope": [ScopeScores(**score) for score in document.get("by_scope", [])],
                    "within_recording": WithinRecording(**document.get("within_recording", asdict(NO_RECORDINGS))),
                }
            )
        except (KeyError, TypeError) as exc:
            raise DataError(f"test report is malformed: {exc!r}") from exc


def _scores(
    test: pd.DataFrame,
    ranking: npt.ArrayLike,
    probabilities: np.ndarray,
    threshold: float,
) -> ModelScores:
    """AUC uses the continuous ranking score; the alert metrics use the decisions at ``threshold``."""
    labels = test["y"].to_numpy()
    metrics = alert_metrics(test, probabilities, threshold)
    return ModelScores(
        auc=float(roc_auc_score(labels, ranking)),
        balanced_accuracy=float(balanced_accuracy_score(labels, probabilities >= threshold)),
        false_alerts_per_hour=metrics.false_alerts_per_hour,
        drowsy_sessions_caught=metrics.drowsy_sessions_caught,
    )


def evaluate_on_test(workspace: Workspace, test: pd.DataFrame) -> TestReport:
    artifact = load_candidate(workspace)
    if set(test["y"].unique()) != {0, 1}:
        raise DataError("the locked test set needs both alert and drowsy windows")
    log = read_json(workspace.test_run_log, produced_by="test") if workspace.test_run_log.is_file() else []
    prior_runs = sum(entry["model_sha256"] == artifact.sha256 for entry in log)

    probabilities = predict_probability(artifact, test[artifact.inputs])
    team_sessions = test["baseline_scope"] == BASELINE_SCOPE_SESSION
    fixed_alerts = fixed_threshold_alerts(test, artifact.fixed_threshold_baseline)
    report = TestReport(
        model_sha256=artifact.sha256,
        people=test["subject"].nunique(),
        windows=len(test),
        driven=_scores(test, probabilities, probabilities, artifact.threshold),
        fixed_threshold=_scores(test, test["perclos"], fixed_alerts, _FIXED_THRESHOLD_DECISION),
        prior_runs=prior_runs,
        by_scope=[
            *scores_by_scope("Driven", test, probabilities, probabilities, artifact.threshold),
            *scores_by_scope(
                "B1 fixed PERCLOS",
                test,
                test["perclos"],
                fixed_alerts,
                _FIXED_THRESHOLD_DECISION,
            ),
        ],
        within_recording=within_recording(test.loc[team_sessions], probabilities[team_sessions.to_numpy()]),
    )
    write_json(workspace.test_run_log, [*log, report])
    if prior_runs == 0:
        write_json(workspace.test_report_path(artifact.short_sha), report)
    else:
        logger.warning(
            "test: model %s was already tested %d time(s); the first result stands",
            artifact.short_sha,
            prior_runs,
        )
    logger.info(
        "test: %d people, AUC %.3f, balanced accuracy %.3f (B1 %.3f), %.2f false alerts/h, %.0f%% drowsy caught",
        report.people,
        report.driven.auc,
        report.driven.balanced_accuracy,
        report.fixed_threshold.balanced_accuracy,
        report.driven.false_alerts_per_hour,
        report.driven.drowsy_sessions_caught * 100,
    )
    log_scopes(logger, "test", report.by_scope)
    logger.info(
        "test: within-recording AUC %.3f over %d held-out team sessions",
        report.within_recording.auc,
        report.within_recording.sessions,
    )
    return report


def load_test_report(workspace: Workspace, artifact: ModelArtifact) -> TestReport:
    return TestReport.from_json(read_json(workspace.test_report_path(artifact.short_sha), produced_by="test"))
