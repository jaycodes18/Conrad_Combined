"""Execute a notebook from a clean kernel and save the executed copy.

Usage: python tools/run_notebook.py NOTEBOOK OUTPUT_DIR

The kernel runs with OUTPUT_DIR as its working directory, so the notebook's workspace lands there and never
in the repository. Exits 1 if any cell fails; the partly executed notebook is still saved for debugging.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbclient.exceptions import CellExecutionError

from driven_ml.workspace import write_text_atomic

KERNEL_NAME = "python3"
CELL_TIMEOUT_SECONDS = 1800


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("notebook", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args(argv)

    notebook = nbformat.read(args.notebook, as_version=nbformat.NO_CONVERT)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(
        notebook,
        kernel_name=KERNEL_NAME,
        timeout=CELL_TIMEOUT_SECONDS,
        resources={"metadata": {"path": str(output_dir)}},
    )
    output_path = output_dir / args.notebook.name
    try:
        client.execute()
    except CellExecutionError as exc:
        write_text_atomic(output_path, nbformat.writes(notebook))
        print(
            f"error: {args.notebook} failed; partial output in {output_path}\n{exc}",
            file=sys.stderr,
        )
        return 1
    write_text_atomic(output_path, nbformat.writes(notebook))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
