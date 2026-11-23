"""Null-signal control: does the pipeline score at chance when labels carry no drowsiness?

Each seed builds people whose drowsy sessions show no drowsiness but whose recordings differ by camera
placement, then runs the full pipeline and scores the locked test people once. Near AUC 0.5 is sound; well above
it means the model learned the recording (Ojala and Garriga, JMLR 11, 2010). It runs in a temporary workspace.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import tempfile
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score

from ..features.synthetic import simulate_sessions
from ..settings import RunConfig
from ..splits import split_windows
from ..windows import build_window_table
from ..workspace import Workspace, write_json, write_parquet
from .artifact import predict_probability
from .training import train

logger = logging.getLogger(__name__)

NULL_CONTROL_SEEDS: tuple[int, ...] = (101, 102, 103)
NULL_CONTROL_PEOPLE = 40
MAX_NULL_AUC_MEAN = 0.60
MAX_NULL_AUC_ANY = 0.65
NULL_CONTROL_FILE = "null_control.json"


@dataclass(frozen=True)
class NullControlResult:
    seed: int
    auc: float
    chosen_model: str
    test_people: int


@dataclass(frozen=True)
class NullControlReport:
    results: tuple[NullControlResult, ...]
    mean_auc: float

    @property
    def passed(self) -> bool:
        return self.mean_auc <= MAX_NULL_AUC_MEAN and all(result.auc <= MAX_NULL_AUC_ANY for result in self.results)

    def to_json(self) -> dict[str, object]:
        return {
            "results": list(self.results),
            "mean_auc": self.mean_auc,
            "passed": self.passed,
        }


@contextlib.contextmanager
def _quiet_pipeline_logs() -> Iterator[None]:
    """The control reruns the whole pipeline; only its own per-seed lines belong in the run's log."""
    package_logger = logging.getLogger("driven_ml")
    previous = package_logger.level
    package_logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        package_logger.setLevel(previous)


@functools.cache
def run_null_control(seed: int, *, subject_count: int = NULL_CONTROL_PEOPLE) -> NullControlResult:
    """One seed of the control, in a temporary dev workspace that is deleted afterwards.

    The result depends only on the code and the seed, so one process computes each seed once.
    """
    with (
        tempfile.TemporaryDirectory(prefix="driven-null-control-") as temp_root,
        warnings.catch_warnings(),
        _quiet_pipeline_logs(),
    ):
        # On labels with no signal the MLP is fitting noise and often stops at max_iter; that is expected here.
        warnings.simplefilter("ignore", ConvergenceWarning)
        config = RunConfig.create("dev", workspace_root=Path(temp_root), seed=seed, run_discovery=False)
        workspace = Workspace(config).ensure()
        write_parquet(
            workspace.frames_path,
            simulate_sessions(subject_count, seed=seed, label_effect=False),
        )
        build_window_table(workspace)
        split = split_windows(config, workspace)
        training = train(config, workspace, split.development)
        test = split.test
        probabilities = predict_probability(training.artifact, test[training.artifact.inputs])
        return NullControlResult(
            seed=seed,
            auc=float(roc_auc_score(test["y"], probabilities)),
            chosen_model=training.winner.name,
            test_people=test["subject"].nunique(),
        )


def null_control_report(
    workspace: Workspace,
    seeds: Sequence[int] = NULL_CONTROL_SEEDS,
    *,
    subject_count: int = NULL_CONTROL_PEOPLE,
) -> NullControlReport:
    """Run every seed, write ``reports/null_control.json`` and log one line per seed."""
    results = tuple(run_null_control(seed, subject_count=subject_count) for seed in seeds)
    report = NullControlReport(results=results, mean_auc=float(np.mean([result.auc for result in results])))
    write_json(workspace.reports_dir / NULL_CONTROL_FILE, report.to_json())
    for result in results:
        logger.info(
            "null control: seed %d -> AUC %.3f (%s, %d test people)",
            result.seed,
            result.auc,
            result.chosen_model,
            result.test_people,
        )
    logger.info(
        "null control: mean AUC %.3f (limit %.2f mean, %.2f any seed)",
        report.mean_auc,
        MAX_NULL_AUC_MEAN,
        MAX_NULL_AUC_ANY,
    )
    return report
