"""The release gate: every check a model must pass before a person may promote it.

The gate never promotes anything. It writes ``staging/<sha>/manifest.json``, and promotion refuses any model
whose manifest has a failing check. Check names are fixed because tests and the notebook match on them.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ..compliance import check_product_flags, scan_claims
from ..data import is_allowed
from ..economics import EconomicsReport
from ..errors import MissingArtifactError
from ..evaluation import load_test_report
from ..modeling.artifact import ModelArtifact, predict_probability
from ..modeling.null_control import (
    MAX_NULL_AUC_ANY,
    MAX_NULL_AUC_MEAN,
    NullControlReport,
    null_control_report,
)
from ..modeling.store import load_candidate
from ..product import Product
from ..scoring import WithinRecording, pooled
from ..settings import DataSource, Track
from ..spec import FALSE_ALERT_BUDGET_PER_HOUR
from ..workspace import Workspace, read_json, to_jsonable, write_json
from .packaging import (
    MIN_PARITY_ROWS_WITH_MISSING,
    MODEL_CARD_FILE,
    MODEL_FILE,
    PARITY_FILE,
    PARITY_TOLERANCE,
    rows_with_missing,
)
from .replay import GROUP_REPLAY, ReplayResult, detection_rate

if TYPE_CHECKING:
    from ..risk.fusion import FusionWeights

logger = logging.getLogger(__name__)

MIN_REPLAY_DETECTION = 0.9
MIN_REPLAY_PEOPLE = 2
MIN_REPLAY_EVENTS = 6
MIN_WITHIN_RECORDING_AUC = 0.80
RECORDING_PROTOCOL = "ml/RECORDING_PROTOCOL.md"
MANIFEST_FILE = "manifest.json"


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class StagingManifest:
    model_sha256: str
    track: str
    data_source: str
    datasets: list[str]
    checks: list[GateCheck]
    null_control: dict[str, Any] | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failed_checks(self) -> list[str]:
        return [check.name for check in self.checks if not check.passed]

    def to_json(self) -> dict[str, Any]:
        return {**asdict(self), "passed": self.passed}

    @classmethod
    def from_json(cls, document: Mapping[str, Any]) -> StagingManifest:
        body = {key: value for key, value in document.items() if key != "passed"}
        return cls(**{**body, "checks": [GateCheck(**check) for check in body["checks"]]})


def _flags_check(product: Product) -> GateCheck:
    findings = check_product_flags(product.flags)
    detail = "; ".join(str(finding) for finding in findings) or "L1-L7 and G1 satisfied"
    return GateCheck("compliance: product flags", not findings, detail)


def _model_card_check(card_text: str) -> GateCheck:
    findings = scan_claims(card_text, MODEL_CARD_FILE)
    detail = "; ".join(str(finding) for finding in findings) or "no banned claims"
    return GateCheck("compliance: no banned claims in the model card (L8)", not findings, detail)


def _datasets_check(artifact: ModelArtifact) -> GateCheck:
    name = "datasets allowed for the track (L9)"
    if artifact.data_source == DataSource.SYNTHETIC:
        return GateCheck(
            name,
            True,
            "skipped: synthetic data has no dataset licenses, and it can never ship",
        )
    refused = [key for key in artifact.datasets if not is_allowed(key, Track(artifact.track))]
    detail = (
        f"not allowed on the {artifact.track} track: {', '.join(refused)}" if refused else ", ".join(artifact.datasets)
    )
    return GateCheck(name, not refused, detail)


def _null_control_check(report: NullControlReport) -> GateCheck:
    """Tests the pipeline design, not the data: with no drowsiness in the labels it must score at chance."""
    aucs = ", ".join(f"seed {result.seed}: {result.auc:.3f}" for result in report.results)
    return GateCheck(
        "null control: pipeline scores at chance when labels carry no signal",
        report.passed,
        f"AUC {aucs}; mean {report.mean_auc:.3f} (limits: mean <= {MAX_NULL_AUC_MEAN}, each <= {MAX_NULL_AUC_ANY})",
    )


def fusion_weights_check(weights: FusionWeights) -> GateCheck:
    """Fitted fusion weights keep the model's track rule: research-track or synthetic weights never pass."""
    blockers = weights.promotion_blockers()
    detail = "; ".join(blockers) or f"{weights.people} people, out-of-fold AUC {weights.out_of_fold_auc:.3f}"
    return GateCheck("fusion weights: commercial track on real data (L9)", not blockers, detail)


def _replay_check(artifact: ModelArtifact, results: Sequence[ReplayResult]) -> GateCheck:
    """At least 90% of scripted drowsiness events caught, on people the model never trained on."""
    name = f"replay: at least {MIN_REPLAY_DETECTION:.0%} of scripted events caught"
    rate = detection_rate(list(results))
    events = sum(result.events for result in results)
    false_alerts = sum(result.false_alerts for result in results)
    replay_people = {result.person for result in results if result.group == GROUP_REPLAY}
    summary = f"{rate:.0%} of {events} events caught, {false_alerts} false alerts (replayed sessions: {len(results)})"
    if artifact.data_source == DataSource.SYNTHETIC:
        return GateCheck(
            name,
            bool(results) and rate >= MIN_REPLAY_DETECTION,
            f"{summary}; the synthetic scripted person is outside training",
        )
    if len(replay_people) < MIN_REPLAY_PEOPLE or events < MIN_REPLAY_EVENTS:
        return GateCheck(
            name,
            False,
            f"need ≥{MIN_REPLAY_PEOPLE} held-out replay people (see RECORDING_PROTOCOL.md); have "
            f"{len(replay_people)} replay people and {events} scripted drowsiness events",
        )
    return GateCheck(
        name,
        rate >= MIN_REPLAY_DETECTION,
        f"{summary} from {len(replay_people)} replay people",
    )


def _within_recording_check(
    artifact: ModelArtifact,
    held_out: WithinRecording,
    replay_results: Sequence[ReplayResult],
) -> GateCheck:
    """On real data a model must separate drowsy from alert inside one recording, which no camera can fake."""
    name = f"within-recording AUC on held-out team sessions >= {MIN_WITHIN_RECORDING_AUC:.2f}"
    if artifact.data_source == DataSource.SYNTHETIC:
        scripted = pooled(result.within_recording for result in replay_results)
        return GateCheck(
            name,
            True,
            f"not gated in dev; scripted session within-recording AUC {scripted.auc:.3f}",
        )
    held_out = held_out + pooled(result.within_recording for result in replay_results if result.group == GROUP_REPLAY)
    if held_out.sessions == 0:
        return GateCheck(
            name,
            False,
            f"no held-out team sessions: record team sessions per {RECORDING_PROTOCOL}",
        )
    return GateCheck(
        name,
        held_out.auc >= MIN_WITHIN_RECORDING_AUC,
        f"AUC {held_out.auc:.3f} over {held_out.sessions} sessions ({held_out.pairs} drowsy-alert pairs)",
    )


def _parity_check(stage_dir: Path) -> GateCheck:
    """Re-run the staged model.json, not the in-memory candidate, so the shipped file itself is checked."""
    staged = ModelArtifact.from_json(read_json(stage_dir / MODEL_FILE, produced_by="package"))
    fixtures = read_json(stage_dir / PARITY_FILE, produced_by="package")
    inputs = np.asarray(fixtures["inputs"], dtype=float)
    reproduced = predict_probability(staged, inputs)
    error = float(np.max(np.abs(reproduced - np.asarray(fixtures["expected"], dtype=float))))
    gapped = rows_with_missing(inputs)
    return GateCheck(
        "parity fixtures reproduce (< 1e-6)",
        error < PARITY_TOLERANCE and gapped >= MIN_PARITY_ROWS_WITH_MISSING,
        f"max error {error:.1e}; {gapped} of {len(inputs)} rows have missing values "
        f"(at least {MIN_PARITY_ROWS_WITH_MISSING} needed)",
    )


def run_stage_gate(
    workspace: Workspace,
    product: Product,
    replay_results: Sequence[ReplayResult],
    economics: EconomicsReport | None,
) -> StagingManifest:
    artifact = load_candidate(workspace)
    stage_dir = workspace.staging_dir(artifact.short_sha)
    card_path = stage_dir / MODEL_CARD_FILE
    if not card_path.is_file():
        raise MissingArtifactError(str(card_path), produced_by="package")
    report = load_test_report(workspace, artifact)
    null_control = null_control_report(workspace)
    driven, fixed = report.driven, report.fixed_threshold

    checks = [
        _flags_check(product),
        _model_card_check(card_path.read_text(encoding="utf-8")),
        _datasets_check(artifact),
        GateCheck(
            "beats the fixed-threshold baseline on test",
            driven.balanced_accuracy > fixed.balanced_accuracy,
            f"balanced accuracy {driven.balanced_accuracy:.3f} vs B1 {fixed.balanced_accuracy:.3f}",
        ),
        _null_control_check(null_control),
        _within_recording_check(artifact, report.within_recording, replay_results),
        GateCheck(
            f"false alerts at most {FALSE_ALERT_BUDGET_PER_HOUR}/h on test",
            driven.false_alerts_per_hour <= FALSE_ALERT_BUDGET_PER_HOUR,
            f"{driven.false_alerts_per_hour:.2f} false alerts per hour over {report.people} test people",
        ),
        _replay_check(artifact, replay_results),
        _parity_check(stage_dir),
    ]
    if economics is not None:
        checks.append(
            GateCheck(
                f"unit economics: margin at least {economics.target_margin:.0%} at the test false-alert rate",
                economics.meets_target,
                f"{economics.unit.plan_id} plan margin {economics.unit.contribution_margin:.1%} at "
                f"{economics.unit.false_alerts_per_hour:.2f} false alerts/h "
                f"(break-even {economics.break_even_false_alerts_per_hour:.2f}/h)",
            )
        )

    manifest = StagingManifest(
        model_sha256=artifact.sha256,
        track=artifact.track,
        data_source=artifact.data_source,
        datasets=list(artifact.datasets),
        checks=checks,
        null_control=to_jsonable(null_control.to_json()),
    )
    write_json(stage_dir / MANIFEST_FILE, manifest.to_json())
    for check in checks:
        logger.info("%s %s: %s", "PASS" if check.passed else "FAIL", check.name, check.detail)
    logger.info("staging %s", "PASSED" if manifest.passed else "FAILED")
    return manifest


def load_manifest(workspace: Workspace, short_sha: str) -> StagingManifest | None:
    path = workspace.staging_root / short_sha / MANIFEST_FILE
    if not path.is_file():
        return None
    return StagingManifest.from_json(read_json(path, produced_by="stage_gate"))
