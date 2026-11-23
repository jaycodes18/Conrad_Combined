"""Person-grouped cross-validation of every candidate, then export of the winner.

Every number here comes from people the model did not train on: folds are grouped by person, temperature is
fitted on out-of-fold logits, and the alert threshold is chosen on out-of-fold probabilities by the false-alert
budget. The winner is the candidate with the fewest false alerts, then the highest AUC. If B3 (no baseline)
wins, the personal-baseline claim failed on this data, and that is a finding, not a bug.

Feature dropout: on ``EYE_MASK_TRAINING_FRACTION`` (0.3) of the training rows of every fold, and of the final
refit, the eye-derived inputs (``EYE_DERIVED_FEATURES`` and their baseline deltas) are replaced by the
development medians, which is exactly what inference does when sunglasses make them NaN. Validation is reported
twice: with full features, and with the eyes masked on every held-out row. The locked test is unchanged.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from ..data import dataset_key_of
from ..errors import DataError
from ..scoring import ScopeScores, log_scopes, scores_by_scope
from ..settings import DataSource, RunConfig
from ..spec import (
    ALERT_COOLDOWN_SECONDS,
    BASELINE_DELTA_PREFIX,
    EYE_DERIVED_FEATURES,
    EYE_MASK_TRAINING_FRACTION,
    FEATURE_SPEC,
    INPUTS_WITH_BASELINE,
)
from ..workspace import Workspace, write_json
from .alert_simulation import alert_metrics, choose_threshold
from .artifact import (
    LINEAR,
    RELU,
    DenseLayer,
    LogisticRegressionWeights,
    MlpWeights,
    ModelArtifact,
    predict_probability,
)
from .calibration import fit_temperature, sigmoid
from .candidates import (
    CANDIDATES,
    FIXED_THRESHOLD_BASELINE,
    FIXED_THRESHOLD_GRID,
    Candidate,
    Estimator,
    EstimatorKind,
    build_estimator,
    decision_logits,
    fixed_threshold_alerts,
)
from .store import save_candidate

logger = logging.getLogger(__name__)

CV_FOLDS = 5
EXPORT_CHECK_ROWS = 200
EXPORT_TOLERANCE = 1e-9
REPORT_COLUMNS: tuple[str, ...] = (
    "model",
    "inputs",
    "meets_budget",
    "auc",
    "balanced_accuracy",
    "false_alerts_per_hour",
    "drowsy_sessions_caught",
    "temperature",
    "threshold",
    "auc_eyes_masked",
    "balanced_accuracy_eyes_masked",
)
EYE_INPUTS: frozenset[str] = frozenset(
    (*EYE_DERIVED_FEATURES, *(f"{BASELINE_DELTA_PREFIX}{name}" for name in EYE_DERIVED_FEATURES))
)


@dataclass(frozen=True)
class CandidateResult:
    name: str
    kind: EstimatorKind
    inputs: tuple[str, ...]
    auc: float
    balanced_accuracy: float
    false_alerts_per_hour: float
    drowsy_sessions_caught: float
    temperature: float
    threshold: float
    meets_budget: bool
    by_scope: tuple[ScopeScores, ...] = ()
    auc_eyes_masked: float = float("nan")
    balanced_accuracy_eyes_masked: float = float("nan")


@dataclass(frozen=True)
class EyeMask:
    """Which input columns are eye-derived, and the development medians that stand in for them."""

    columns: tuple[int, ...]
    fill_values: tuple[float, ...]
    fraction: float = EYE_MASK_TRAINING_FRACTION

    @classmethod
    def for_inputs(cls, inputs: tuple[str, ...] | list[str], medians: pd.Series) -> EyeMask:
        columns = tuple(index for index, name in enumerate(inputs) if name in EYE_INPUTS)
        return cls(columns=columns, fill_values=tuple(float(medians[inputs[index]]) for index in columns))

    def apply(self, features: np.ndarray, rows: np.ndarray) -> np.ndarray:
        """A copy with the eye columns of ``rows`` (a boolean mask) set to their fill values."""
        masked = features.copy()
        if self.columns:
            masked[np.ix_(np.flatnonzero(rows), np.asarray(self.columns, dtype=np.intp))] = np.asarray(self.fill_values)
        return masked

    def augment(self, features: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return self.apply(features, rng.random(len(features)) < self.fraction)


@dataclass(frozen=True)
class FixedThresholdResult:
    perclos_threshold: float
    auc: float
    balanced_accuracy: float
    false_alerts_per_hour: float
    drowsy_sessions_caught: float
    by_scope: tuple[ScopeScores, ...] = ()


@dataclass(frozen=True)
class ValidationReport:
    candidates: tuple[CandidateResult, ...]
    fixed_threshold: FixedThresholdResult

    def best(self) -> CandidateResult:
        return min(
            self.candidates,
            key=lambda result: (result.false_alerts_per_hour, -result.auc),
        )

    def to_frame(self) -> pd.DataFrame:
        rows = [
            {
                "model": result.name,
                "inputs": len(result.inputs),
                "meets_budget": result.meets_budget,
                "auc": result.auc,
                "balanced_accuracy": result.balanced_accuracy,
                "false_alerts_per_hour": result.false_alerts_per_hour,
                "drowsy_sessions_caught": result.drowsy_sessions_caught,
                "temperature": result.temperature,
                "threshold": result.threshold,
                "auc_eyes_masked": result.auc_eyes_masked,
                "balanced_accuracy_eyes_masked": result.balanced_accuracy_eyes_masked,
            }
            for result in self.candidates
        ]
        fixed = self.fixed_threshold
        rows.append(
            {
                "model": FIXED_THRESHOLD_BASELINE,
                "inputs": 1,
                "meets_budget": None,
                "auc": fixed.auc,
                "balanced_accuracy": fixed.balanced_accuracy,
                "false_alerts_per_hour": fixed.false_alerts_per_hour,
                "drowsy_sessions_caught": fixed.drowsy_sessions_caught,
                "temperature": float("nan"),
                "threshold": fixed.perclos_threshold,
                "auc_eyes_masked": float("nan"),
                "balanced_accuracy_eyes_masked": float("nan"),
            }
        )
        return pd.DataFrame(rows, columns=list(REPORT_COLUMNS))

    def scope_frame(self) -> pd.DataFrame:
        """Every candidate's and B1's metrics for each baseline scope."""
        scopes = [score for result in self.candidates for score in result.by_scope]
        return pd.DataFrame([asdict(score) for score in [*scopes, *self.fixed_threshold.by_scope]])


@dataclass(frozen=True)
class TrainingResult:
    artifact: ModelArtifact
    validation: ValidationReport
    winner: CandidateResult
    path: Path


def _check_trainable(development: pd.DataFrame) -> None:
    classes = set(development["y"].unique())
    if classes != {0, 1}:
        raise DataError(f"training needs both alert and drowsy windows; found labels {sorted(classes)}")
    people = development["subject"].nunique()
    if people < CV_FOLDS:
        raise DataError(f"person-grouped cross-validation needs at least {CV_FOLDS} people; found {people}")


def _fill_missing(windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Fill NaN inputs with the development medians; an all-NaN column gets 0.0."""
    inputs = windows[list(INPUTS_WITH_BASELINE)]
    medians = inputs.median().fillna(0.0)
    return inputs.fillna(medians), medians


def out_of_fold_logits(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    kind: EstimatorKind,
    seed: int,
    *,
    eye_mask: EyeMask | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Held-out logits with full features, and with the held-out rows' eye inputs masked.

    With ``eye_mask``, each fold trains on rows whose eye inputs are masked at the mask's fraction.
    """
    logits = np.empty(len(labels), dtype=float)
    masked_logits = np.empty(len(labels), dtype=float)
    rng = np.random.default_rng(seed)
    everything = np.ones(len(labels), dtype=bool)
    for train_rows, held_out_rows in GroupKFold(n_splits=CV_FOLDS).split(features, labels, groups):
        train = features[train_rows] if eye_mask is None else eye_mask.augment(features[train_rows], rng)
        scaler = StandardScaler().fit(train)
        estimator = build_estimator(kind, seed).fit(scaler.transform(train), labels[train_rows])
        held_out = features[held_out_rows]
        logits[held_out_rows] = decision_logits(estimator, scaler.transform(held_out))
        hidden = held_out if eye_mask is None else eye_mask.apply(held_out, everything[: len(held_out)])
        masked_logits[held_out_rows] = decision_logits(estimator, scaler.transform(hidden))
    return logits, masked_logits


def _evaluate_candidate(
    candidate: Candidate, development: pd.DataFrame, filled: pd.DataFrame, medians: pd.Series, seed: int
) -> CandidateResult:
    labels = development["y"].to_numpy()
    logits, masked_logits = out_of_fold_logits(
        filled[list(candidate.inputs)].to_numpy(),
        labels,
        development["subject"].to_numpy(),
        candidate.kind,
        seed,
        eye_mask=EyeMask.for_inputs(candidate.inputs, medians),
    )
    temperature = fit_temperature(logits, labels)
    probabilities = sigmoid(logits / temperature)
    masked_probabilities = sigmoid(masked_logits / temperature)
    choice = choose_threshold(development, probabilities)
    metrics = alert_metrics(development, probabilities, choice.threshold)
    return CandidateResult(
        name=candidate.name,
        kind=candidate.kind,
        inputs=candidate.inputs,
        auc=float(roc_auc_score(labels, probabilities)),
        balanced_accuracy=float(balanced_accuracy_score(labels, probabilities >= choice.threshold)),
        false_alerts_per_hour=metrics.false_alerts_per_hour,
        drowsy_sessions_caught=metrics.drowsy_sessions_caught,
        temperature=temperature,
        threshold=choice.threshold,
        meets_budget=choice.meets_budget,
        by_scope=tuple(
            scores_by_scope(
                candidate.name,
                development,
                probabilities,
                probabilities,
                choice.threshold,
            )
        ),
        auc_eyes_masked=float(roc_auc_score(labels, masked_probabilities)),
        balanced_accuracy_eyes_masked=float(balanced_accuracy_score(labels, masked_probabilities >= choice.threshold)),
    )


def _evaluate_fixed_threshold(development: pd.DataFrame) -> FixedThresholdResult:
    labels = development["y"].to_numpy()
    perclos = development["perclos"].to_numpy(dtype=float)
    accuracies = [balanced_accuracy_score(labels, perclos >= value) for value in FIXED_THRESHOLD_GRID]
    best_value = float(FIXED_THRESHOLD_GRID[int(np.argmax(accuracies))])
    alerts = fixed_threshold_alerts(development, best_value)
    metrics = alert_metrics(development, alerts, 0.5)
    return FixedThresholdResult(
        perclos_threshold=best_value,
        auc=float(roc_auc_score(labels, perclos)),
        balanced_accuracy=float(max(accuracies)),
        false_alerts_per_hour=metrics.false_alerts_per_hour,
        drowsy_sessions_caught=metrics.drowsy_sessions_caught,
        by_scope=tuple(scores_by_scope(FIXED_THRESHOLD_BASELINE, development, perclos, alerts, 0.5)),
    )


def cross_validate(development: pd.DataFrame, seed: int) -> ValidationReport:
    _check_trainable(development)
    filled, medians = _fill_missing(development)
    candidates = tuple(_evaluate_candidate(candidate, development, filled, medians, seed) for candidate in CANDIDATES)
    return ValidationReport(candidates=candidates, fixed_threshold=_evaluate_fixed_threshold(development))


def _weights_of(estimator: Estimator) -> LogisticRegressionWeights | MlpWeights:
    if isinstance(estimator, LogisticRegression):
        return LogisticRegressionWeights(
            coefficients=[float(value) for value in estimator.coef_.ravel()],
            intercept=float(estimator.intercept_[0]),
        )
    last = len(estimator.coefs_) - 1
    return MlpWeights(
        layers=[
            DenseLayer(
                weights=weights.tolist(),
                bias=bias.tolist(),
                activation=LINEAR if index == last else RELU,
            )
            for index, (weights, bias) in enumerate(zip(estimator.coefs_, estimator.intercepts_, strict=True))
        ]
    )


def _datasets_of(config: RunConfig, development: pd.DataFrame) -> list[str]:
    if config.data_source is DataSource.SYNTHETIC:
        return [DataSource.SYNTHETIC.value]
    return sorted({dataset_key_of(subject) for subject in development["subject"].unique()})


def _export(
    config: RunConfig,
    development: pd.DataFrame,
    winner: CandidateResult,
    fixed: FixedThresholdResult,
) -> ModelArtifact:
    """Refit the winner on every development row, with the same eye-input dropout, and write it out as numbers."""
    filled, medians = _fill_missing(development)
    inputs = list(winner.inputs)
    features = filled[inputs].to_numpy()
    augmented = EyeMask.for_inputs(inputs, medians).augment(features, np.random.default_rng(config.seed))
    scaler = StandardScaler().fit(augmented)
    estimator = build_estimator(winner.kind, config.seed).fit(scaler.transform(augmented), development["y"].to_numpy())
    artifact = ModelArtifact(
        environment=config.environment.value,
        track=config.track.value,
        data_source=config.data_source.value,
        datasets=_datasets_of(config, development),
        inputs=inputs,
        fill_values=[float(medians[name]) for name in inputs],
        scaler_mean=[float(value) for value in scaler.mean_],
        scaler_scale=[float(value) for value in scaler.scale_],
        weights=_weights_of(estimator),
        temperature=winner.temperature,
        threshold=winner.threshold,
        cooldown_seconds=float(ALERT_COOLDOWN_SECONDS),
        fixed_threshold_baseline=fixed.perclos_threshold,
        feature_spec_version=FEATURE_SPEC.version,
    )
    sample = features[:EXPORT_CHECK_ROWS]
    expected = estimator.predict_proba(scaler.transform(sample))[:, 1]
    exported = predict_probability(artifact, sample, temperature=1.0)
    error = float(np.max(np.abs(expected - exported)))
    if error > EXPORT_TOLERANCE:
        raise AssertionError(f"exported model differs from scikit-learn by {error:.2e}")
    return artifact


def train(config: RunConfig, workspace: Workspace, development: pd.DataFrame) -> TrainingResult:
    report = cross_validate(development, config.seed)
    winner = report.best()
    artifact = _export(config, development, winner, report.fixed_threshold)
    path = save_candidate(workspace, artifact, report.to_frame())
    logger.info(
        "train: %s won (AUC %.3f, %.2f false alerts/h, threshold %.2f); candidate %s",
        winner.name,
        winner.auc,
        winner.false_alerts_per_hour,
        winner.threshold,
        path.name,
    )
    logger.info(
        "train: with the eye inputs masked on every held-out window, AUC %.3f, balanced accuracy %.3f",
        winner.auc_eyes_masked,
        winner.balanced_accuracy_eyes_masked,
    )
    log_scopes(logger, "train", winner.by_scope)
    if not winner.meets_budget:
        logger.warning(
            "train: no candidate met the false-alert budget in cross-validation; staging checks it on the locked test"
        )
    return TrainingResult(artifact=artifact, validation=report, winner=winner, path=path)


def write_validation_report(workspace: Workspace, report: ValidationReport) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    document = {
        "candidates": report.to_frame().to_dict(orient="records"),
        "by_scope": report.scope_frame().to_dict(orient="records"),
    }
    path = write_json(workspace.reports_dir / f"validation-{stamp}.json", document)
    logger.info("validate: report written to %s", path)
    return path


def validate(workspace: Workspace, development: pd.DataFrame, seed: int) -> ValidationReport:
    report = cross_validate(development, seed)
    write_validation_report(workspace, report)
    return report
