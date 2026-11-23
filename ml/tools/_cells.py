"""Small helpers shared by the notebook builders: dedented cells, and an atomic save formatted with ruff.

Running ``ruff format`` on every generated notebook keeps ``python tools/check.py`` clean however a cell's source was
indented in the builder.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import nbformat
from nbformat.notebooknode import NotebookNode

from driven_ml.workspace import write_text_atomic

NOTEBOOK_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def md(text: str) -> NotebookNode:
    return nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str) -> NotebookNode:
    return nbformat.v4.new_code_cell(textwrap.dedent(text).strip())


def save(cells: list[NotebookNode], file_name: str) -> Path:
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
    )
    nbformat.validate(notebook)
    path = write_text_atomic(NOTEBOOK_DIR / file_name, nbformat.writes(notebook))
    subprocess.run([sys.executable, "-m", "ruff", "format", "--quiet", str(path)], check=True)
    return path
