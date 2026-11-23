"""Promotion: the only path from a staged candidate to a released model, and it needs a person.

A model is promoted only when its staging manifest passed every check, it was trained on real data on the
commercial track from datasets cleared for commercial use (L9), and someone put their name on it. Promotion
copies files. It never uploads or deploys: releasing into the app writes files that a person commits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .compliance import check_product_flags
from .data import DATASETS
from .errors import ComplianceError, ConfigurationError
from .evaluation import load_test_report
from .modeling.artifact import ModelArtifact
from .modeling.store import load_candidate
from .product import Product
from .settings import DataSource, RunConfig, Track
from .staging.gate import StagingManifest, load_manifest
from .staging.packaging import FEATURE_SPEC_FILE, MODEL_FILE, render_model_card
from .workspace import Workspace, copy_text_atomic, write_json, write_text_atomic

logger = logging.getLogger(__name__)

APP_MODEL_FILE = "driven-current.json"
APP_RELEASE_FILE = "release.json"
APP_MODELS_DIR = Path("public") / "models"


@dataclass(frozen=True)
class Release:
    version: str
    model_path: Path
    notes_path: Path
    app_model_path: Path | None


def refusal_reasons(config: RunConfig, product: Product, manifest: StagingManifest) -> list[str]:
    reasons = []
    if not manifest.passed:
        reasons.append("staging gates failed: " + ", ".join(manifest.failed_checks))
    if manifest.track != Track.COMMERCIAL:
        reasons.append("only commercial-track models can ship (L9)")
    if manifest.data_source != DataSource.REAL:
        reasons.append("models trained on synthetic data can never ship")
    uncleared = [key for key in manifest.datasets if key not in DATASETS or DATASETS[key].commercial_use is not True]
    if uncleared:
        reasons.append(f"datasets not cleared for commercial use (L9): {', '.join(uncleared)}")
    if not config.approved_by:
        reasons.append("no named approver (--approved-by)")
    if check_product_flags(product.flags):
        reasons.append("product.json flags fail compliance")
    if config.write_to_app and not product.contact_email.strip():
        reasons.append(
            "product.json contact_email is empty; the privacy notice must say how to reach us "
            "(TDPSA, Tex. Bus. & Com. Code 541.102)"
        )
    return reasons


def promote(config: RunConfig, workspace: Workspace, product: Product) -> Release:
    artifact = load_candidate(workspace)
    manifest = load_manifest(workspace, artifact.short_sha)
    if manifest is None:
        raise ComplianceError("Promotion refused", ["the candidate has not been staged"])
    reasons = refusal_reasons(config, product, manifest)
    if reasons:
        raise ComplianceError("Promotion refused", reasons)

    stage_dir = workspace.staging_dir(artifact.short_sha)
    version = f"driven-v{len(list(workspace.production_dir.glob('driven-v*.json'))) + 1}"
    model_path = copy_text_atomic(
        stage_dir / MODEL_FILE,
        workspace.production_dir / f"{version}-{artifact.short_sha}.json",
    )
    card = render_model_card(artifact, load_test_report(workspace, artifact))
    notes = f"# {version}\n\nApproved by {config.approved_by} on {datetime.now(UTC).date().isoformat()}.\n\n{card}"
    notes_path = write_text_atomic(model_path.with_suffix(".RELEASE.md"), notes)
    logger.info(
        "promote: %s (%s) approved by %s",
        version,
        artifact.short_sha,
        config.approved_by,
    )
    app_model_path = release_to_app(config, stage_dir, version, artifact) if config.write_to_app else None
    return Release(
        version=version,
        model_path=model_path,
        notes_path=notes_path,
        app_model_path=app_model_path,
    )


def release_to_app(config: RunConfig, stage_dir: Path, version: str, artifact: ModelArtifact) -> Path:
    """Copy the released model and feature spec into the app's public folder. Nothing is uploaded."""
    if config.app_dir is None:
        raise ConfigurationError("no app directory: pass --app-dir")
    models_dir = config.app_dir / APP_MODELS_DIR
    app_model_path = copy_text_atomic(stage_dir / MODEL_FILE, models_dir / APP_MODEL_FILE)
    copy_text_atomic(stage_dir / FEATURE_SPEC_FILE, models_dir / FEATURE_SPEC_FILE)
    write_json(
        models_dir / APP_RELEASE_FILE,
        {
            "version": version,
            "sha256": artifact.sha256,
            "approved_by": config.approved_by,
            "date": datetime.now(UTC).date().isoformat(),
            "feature_spec_version": artifact.feature_spec_version,
        },
    )
    logger.info("promote: wrote %s into %s; commit and push to deploy", version, models_dir)
    return app_model_path
