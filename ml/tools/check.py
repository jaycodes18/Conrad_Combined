"""Every check CI runs, with Python only: no make, no shell scripts.

Run from the repository root or from ``ml/``, with the virtual environment's Python:

    .venv/bin/python ml/tools/check.py                 # ruff, ruff format, mypy, pytest
    .venv/bin/python ml/tools/check.py --dev-run       # ... then the full synthetic pipeline through staging
    .venv/bin/python ml/tools/check.py --notebooks     # ... then rebuild and execute both notebooks
    .venv/bin/python ml/tools/check.py --all           # everything CI runs

Each step runs as ``python -m <tool>`` with this same interpreter, so it always uses the venv's packages.
Stops at the first failing step and exits with its code.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ML = Path(__file__).resolve().parents[1]
NOTEBOOKS = ("Driven_ML_Pipeline.ipynb", "Driven_Product_and_Business.ipynb")


def run(*args: str) -> None:
    command = [sys.executable, *args]
    print(f"\n$ python {' '.join(args)}", flush=True)
    result = subprocess.run(command, cwd=ML, check=False)
    if result.returncode != 0:
        print(f"\nFAILED: python {' '.join(args)}", flush=True)
        sys.exit(result.returncode)


def static_checks() -> None:
    run("-m", "ruff", "check", ".")
    run("-m", "ruff", "format", "--check", ".")
    run("-m", "mypy", "driven_ml")
    run("-m", "pytest", "-q")


def dev_run() -> None:
    run("-m", "driven_ml", "--env", "dev", "--no-discovery")


def notebooks() -> None:
    run("tools/build_notebook.py")
    run("tools/build_business_notebook.py")
    with tempfile.TemporaryDirectory() as scratch:
        for name in NOTEBOOKS:
            run("tools/run_notebook.py", f"notebooks/{name}", scratch)
        for name in NOTEBOOKS:
            shutil.copy(Path(scratch) / name, ML / "notebooks" / name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--dev-run",
        action="store_true",
        help="also run the dev pipeline through staging",
    )
    parser.add_argument(
        "--notebooks",
        action="store_true",
        help="also rebuild and execute both notebooks",
    )
    parser.add_argument("--all", action="store_true", help="everything CI runs")
    args = parser.parse_args()
    static_checks()
    if args.dev_run or args.all:
        dev_run()
    if args.notebooks or args.all:
        notebooks()
    print("\nall checks passed", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
