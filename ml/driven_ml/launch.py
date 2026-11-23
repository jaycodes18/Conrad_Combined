"""Launch readiness: everything that must be true before real families use Driven.

``driven-launch-check`` prints one row per check with the fix for each failing one, then the planning
assumptions in product.json still to replace with measured pilot values. In this ML-only repository it exits
1, which is correct: there is no app and no released real model yet.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .app_fixtures import APP_FIXTURES_DIR, FIXTURE_FILES
from .compliance import check_product_flags, scan_repository
from .compliance.rules import BACKEND_PRODUCT_COPY, SOURCE_GLOBS, USER_FACING_TEXT_FILES
from .errors import DrivenError
from .modeling.artifact import ModelArtifact
from .product import UsageAssumptions, load_product
from .release import APP_MODEL_FILE, APP_MODELS_DIR
from .settings import find_repo_root
from .spec import FEATURE_SPEC

APP_DIR = Path("apps") / "web"
SELF_HOSTED_FACE_MODEL = Path("public") / "mediapipe" / "face_landmarker.task"


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    passed: bool
    detail: str
    how_to_fix: str


def _scanned_file_count(repo_root: Path) -> int:
    text_files = sum((repo_root / relative).is_file() for relative in USER_FACING_TEXT_FILES)
    return text_files + sum(1 for pattern in SOURCE_GLOBS for _ in repo_root.glob(pattern))


def _released_model(model_path: Path) -> tuple[ModelArtifact | None, str]:
    if not model_path.is_file():
        return None, f"{model_path} not found"
    try:
        return (
            ModelArtifact.from_json(json.loads(model_path.read_text(encoding="utf-8"))),
            "found",
        )
    except (json.JSONDecodeError, DrivenError) as exc:
        return None, f"unreadable: {exc}"


def readiness_checks(repo_root: Path) -> list[ReadinessCheck]:
    product = load_product(repo_root)
    app_dir = repo_root / APP_DIR
    findings = scan_repository(repo_root)
    code_findings = [finding for finding in findings if finding.rule_id != "config"]
    scanned = _scanned_file_count(repo_root)
    flag_findings = check_product_flags(product.flags)
    backend_copy = repo_root / BACKEND_PRODUCT_COPY
    model, model_status = _released_model(app_dir / APP_MODELS_DIR / APP_MODEL_FILE)
    face_model_found = (app_dir / SELF_HOSTED_FACE_MODEL).is_file()
    missing_fixtures = [name for name in FIXTURE_FILES if not (app_dir / APP_FIXTURES_DIR / name).is_file()]
    return [
        ReadinessCheck(
            "product flags pass compliance",
            not flag_findings,
            "; ".join(map(str, flag_findings)) or "L1-L7 and G1 satisfied",
            "set the flags in product.json back to the values in compliance/rules.py",
        ),
        ReadinessCheck(
            "landing page, app and backend scan clean",
            scanned > 0 and not code_findings,
            "; ".join(map(str, code_findings)) or f"{scanned} files scanned",
            "build the landing page and app in apps/web, then remove every flagged claim or API",
        ),
        ReadinessCheck(
            "backend copy of product.json in sync",
            backend_copy.is_file() and not any(finding.rule_id == "config" for finding in findings),
            "in sync" if backend_copy.is_file() else f"{BACKEND_PRODUCT_COPY} not found",
            "run `make sync` in the app repository",
        ),
        ReadinessCheck(
            "privacy contact email set",
            bool(product.contact_email.strip()),
            product.contact_email or "contact_email is empty",
            "set contact_email in product.json (TDPSA, Tex. Bus. & Com. Code 541.102)",
        ),
        ReadinessCheck(
            "Stripe Payment Link set",
            bool(product.stripe_payment_link.strip()),
            product.stripe_payment_link or "billing.stripe_payment_link is empty",
            "create a Payment Link for the Family plan in Stripe and set billing.stripe_payment_link",
        ),
        ReadinessCheck(
            "released model in the app",
            model is not None,
            model_status,
            "promote a staged model: python -m driven_ml --env production --track commercial "
            "--approved-by <name> --write-app",
        ),
        ReadinessCheck(
            "released model uses real, commercial-track data",
            model is not None and model.is_shippable_source,
            f"{model.data_source} data, {model.track} track" if model else "no released model",
            "train on consented team recordings or commercially licensed data with --track commercial",
        ),
        ReadinessCheck(
            "released model matches the feature spec version",
            model is not None and model.feature_spec_version == FEATURE_SPEC.version,
            f"model v{model.feature_spec_version}, spec v{FEATURE_SPEC.version}" if model else "no released model",
            "retrain and promote with the current feature spec",
        ),
        ReadinessCheck(
            "Face Landmarker model self-hosted",
            face_model_found,
            f"{APP_DIR / SELF_HOSTED_FACE_MODEL} {'found' if face_model_found else 'not found'}",
            "download face_landmarker.task into apps/web/public/mediapipe/ so no video frame leaves the phone",
        ),
        ReadinessCheck(
            "app parity fixtures present",
            not missing_fixtures,
            f"missing: {', '.join(missing_fixtures)}" if missing_fixtures else "all present",
            "run a staging or dev pipeline with --write-app and make the app's parity tests pass",
        ),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    repo_root = find_repo_root(Path.cwd()) or find_repo_root()
    if repo_root is None:
        print(
            "error: run driven-launch-check inside the Driven repository",
            file=sys.stderr,
        )
        return 1
    try:
        checks = readiness_checks(repo_root)
    except DrivenError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    width = max(len(check.name) for check in checks)
    print(f"Launch readiness for {repo_root}\n")
    for check in checks:
        print(f"{'PASS' if check.passed else 'FAIL'}  {check.name.ljust(width)}  {check.detail}")
        if not check.passed:
            print(f"      {'fix:'.ljust(width)}  {check.how_to_fix}")
    print(f"\nPlanning estimates still to replace with measured pilot values: {', '.join(UsageAssumptions.names())}")
    passed = sum(check.passed for check in checks)
    print(f"{passed}/{len(checks)} checks passed")
    return 0 if passed == len(checks) else 1
