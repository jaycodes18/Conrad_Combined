"""Save and load the candidate model, the one ``train`` produced last.

``models/CANDIDATE`` names the candidate's file, so test, staging and production always act on the same model
and nobody passes a model path by hand.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..errors import MissingArtifactError
from ..workspace import Workspace, read_json, write_json, write_text_atomic
from .artifact import ModelArtifact


def save_candidate(workspace: Workspace, artifact: ModelArtifact, validation: pd.DataFrame) -> Path:
    path = workspace.candidate_path(artifact.short_sha)
    write_json(path, artifact.to_json())
    write_text_atomic(path.with_suffix(".validation.csv"), validation.to_csv(index=False))
    write_text_atomic(workspace.candidate_pointer, path.name)
    return path


def load_candidate(workspace: Workspace) -> ModelArtifact:
    if not workspace.candidate_pointer.is_file():
        raise MissingArtifactError("candidate model", produced_by="train")
    file_name = workspace.candidate_pointer.read_text(encoding="utf-8").strip()
    return ModelArtifact.from_json(read_json(workspace.models_dir / file_name, produced_by="train"))
