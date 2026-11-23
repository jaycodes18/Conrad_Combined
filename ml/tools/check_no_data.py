"""Refuse data, recordings and oversized files in git.

Usage:
    python ml/tools/check_no_data.py FILE...    check the given paths (the pre-commit hook passes staged files)
    python ml/tools/check_no_data.py --tracked  check every file git tracks (CI)

Face video and consent files are biometric records (Tex. Bus. & Com. Code 503.001), and pipeline workspaces
hold features derived from them, so none of it may enter git. `.gitignore` stops an ordinary `git add`; this
check also stops `git add -f` and anything already tracked. The patterns mirror `.gitignore`, and a test keeps
the two in step. Exits 1 and lists every offending path.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path, PurePosixPath

DATA_DIRECTORIES = frozenset({".venv", "driven_work", "recordings", ".ipynb_checkpoints"})
DATA_SUFFIXES = frozenset({".parquet", ".mp4", ".mov", ".avi", ".mkv"})
MAX_FILE_BYTES = 5 * 1024 * 1024


def data_reason(path: str) -> str | None:
    """Why `path` (relative to the repository root, `/`-separated) is data, or None if it is not."""
    parts = PurePosixPath(path).parts
    directory = next((part for part in parts[:-1] if part in DATA_DIRECTORIES), None)
    if directory is not None:
        return f"inside {directory}/"
    suffix = PurePosixPath(path).suffix.lower()
    if suffix in DATA_SUFFIXES:
        return f"{suffix} file"
    return None


def size_reason(path: Path) -> str | None:
    """Why the file at `path` is too large, or None. A path that no longer exists (a staged delete) passes."""
    if not path.is_file():
        return None
    size_bytes = path.stat().st_size
    if size_bytes > MAX_FILE_BYTES:
        return f"{size_bytes / 1024 / 1024:.1f} MB, over the {MAX_FILE_BYTES // 1024 // 1024} MB limit"
    return None


def find_violations(paths: Iterable[str], root: Path) -> list[str]:
    violations = []
    for path in paths:
        reason = data_reason(path) or size_reason(root / path)
        if reason is not None:
            violations.append(f"{path}: {reason}")
    return violations


def tracked_files(root: Path) -> list[str]:
    listing = subprocess.run(["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True, text=True)
    return [path for path in listing.stdout.split("\0") if path]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="*", help="paths relative to the repository root")
    parser.add_argument("--tracked", action="store_true", help="check every file git tracks")
    args = parser.parse_args(argv)

    root = Path.cwd()
    paths = tracked_files(root) if args.tracked else args.paths
    violations = find_violations(paths, root)
    for violation in violations:
        print(f"error: {violation}", file=sys.stderr)
    if violations:
        print(
            "Data, recordings and files over 5 MB never enter git (see CONTRIBUTING.md).",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
