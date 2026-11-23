"""Command line: ``python -m driven_ml --env <environment>`` runs that environment's stages.

Exit codes: 0 on success, 2 when a compliance or promotion check refuses the run, 1 for any other pipeline
error. Errors are printed to stderr; everything else goes through logging.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from .errors import ComplianceError, DrivenError
from .pipeline import run_pipeline
from .settings import Environment, RunConfig, Track

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
LOG_DATE_FORMAT = "%H:%M:%S"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="driven-ml", description="Driven drowsiness-model pipeline.")
    parser.add_argument(
        "--env",
        choices=[env.value for env in Environment],
        default=Environment.DEV.value,
    )
    parser.add_argument(
        "--track",
        choices=[track.value for track in Track],
        default=Track.RESEARCH.value,
    )
    parser.add_argument("--root", type=Path, default=Path("driven_work"), help="workspace directory")
    parser.add_argument("--approved-by", default="", help="name of the person approving a promotion")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--reuse-features",
        action="store_true",
        help="reuse frames.parquet instead of re-extracting",
    )
    parser.add_argument(
        "--no-discovery",
        action="store_true",
        help="skip the Hugging Face and Kaggle search",
    )
    parser.add_argument(
        "--write-app",
        action="store_true",
        help="copy fixtures or a released model into the app",
    )
    parser.add_argument(
        "--app-dir",
        type=Path,
        default=None,
        help="app directory (default: <repo>/apps/web)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
    )
    try:
        config = RunConfig.create(
            args.env,
            args.track,
            args.root,
            seed=args.seed,
            approved_by=args.approved_by,
            reuse_features=args.reuse_features,
            run_discovery=not args.no_discovery,
            app_dir=args.app_dir,
            write_to_app=args.write_app,
        )
        run_pipeline(config)
    except ComplianceError as exc:
        print(exc, file=sys.stderr)
        return 2
    except DrivenError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
