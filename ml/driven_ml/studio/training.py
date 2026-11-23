"""Training run: the real pipeline, one stage at a time, with each stage's result published as it lands.

Stages run through `pipeline.STAGE_RUNNERS`, exactly as `run_pipeline` does, so the model that comes
out is the pipeline's own model. Two extra views make training visible rather than a final number:

* epoch curves: after the split, an MLP and a logistic model are fit epoch by epoch on the development
  people (75/25 split by person) and their loss and AUC are streamed. This is a display refit: the
  shipped model is still the one the pipeline's `train` stage fits with grouped cross-validation.
* score distributions and the threshold sweep on the locked test people, computed after the official
  test stage has run once. They change nothing the pipeline decides.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import threading
import time
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..modeling.artifact import ModelArtifact, predict_probability
from ..settings import RunConfig, Stage
from ..spec import FEATURE_SPEC, INPUTS_WITH_BASELINE
from .events import EventHub
from .live import LiveEngine

logger = logging.getLogger(__name__)

DISPLAY_EPOCHS = 80
EPOCH_PAUSE_SECONDS = 0.04  # slow enough to watch, fast enough not to matter
SWEEP_POINTS = 60


class _HubLogHandler(logging.Handler):
    def __init__(self, hub: EventHub) -> None:
        super().__init__(level=logging.INFO)
        self.hub = hub

    def emit(self, record: logging.LogRecord) -> None:
        if record.name.startswith("driven_ml.studio"):
            return
        self.hub.publish(
            "log",
            {
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
            },
        )


class TrainingRun:
    def __init__(self, hub: EventHub, live: LiveEngine) -> None:
        self.hub = hub
        self.live = live
        self._thread: threading.Thread | None = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, environment: str, track: str, workspace_root: Path, seed: int) -> bool:
        if self.running:
            return False
        self._thread = threading.Thread(
            target=self._run,
            args=(environment, track, workspace_root, seed),
            name="driven-train",
            daemon=True,
        )
        self._thread.start()
        return True

    def _run(self, environment: str, track: str, workspace_root: Path, seed: int) -> None:
        from ..pipeline import STAGE_RUNNERS, start_state

        handler = _HubLogHandler(self.hub)
        package_logger = logging.getLogger("driven_ml")
        package_logger.addHandler(handler)
        started = time.monotonic()
        try:
            config = RunConfig.create(environment, track, workspace_root, seed=seed, run_discovery=False)
            state = start_state(config)
            stages = [stage.value for stage in config.stages]
            self.hub.publish(
                "train_plan",
                {
                    "environment": environment,
                    "track": track,
                    "stages": stages,
                    "workspace": str(state.workspace.base),
                },
            )
            for stage in config.stages:
                self.hub.publish("stage", {"stage": stage.value, "state": "running"})
                stage_started = time.monotonic()
                STAGE_RUNNERS[stage](state)
                self.hub.publish(
                    "stage",
                    {
                        "stage": stage.value,
                        "state": "done",
                        "seconds": time.monotonic() - stage_started,
                        "summary": _summary(stage, state),
                    },
                )
                if stage is Stage.SPLIT and state.split is not None:
                    self._epoch_curves(state.split.development, seed)
                if stage is Stage.TRAIN and state.training is not None:
                    self.live.set_model(
                        state.training.artifact,
                        origin=f"trained just now ({environment})",
                    )
                if stage is Stage.TEST and state.split is not None and state.training is not None:
                    self._test_views(state.training.artifact, state.split.test)
            manifest = getattr(state, "manifest", None)
            self.hub.publish(
                "train_done",
                {
                    "ok": True,
                    "seconds": time.monotonic() - started,
                    "passed": getattr(manifest, "passed", None),
                },
            )
        except Exception as error:  # a failing stage is shown on the dashboard
            logger.exception("training run failed")
            self.hub.publish(
                "train_done",
                {
                    "ok": False,
                    "error": f"{type(error).__name__}: {error}",
                    "seconds": time.monotonic() - started,
                },
            )
        finally:
            package_logger.removeHandler(handler)

    def _epoch_curves(self, development: pd.DataFrame, seed: int) -> None:
        from sklearn.linear_model import SGDClassifier
        from sklearn.metrics import log_loss, roc_auc_score
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler

        inputs = [name for name in INPUTS_WITH_BASELINE if name in development]
        frame = development[inputs]
        x = frame.fillna(frame.median(numeric_only=True)).fillna(0.0).to_numpy(dtype=float)
        y = development["y"].to_numpy(dtype=int)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
        train_index, held_index = next(splitter.split(x, y, groups=development["subject"]))
        if len(set(y[held_index])) < 2 or len(set(y[train_index])) < 2:
            self.hub.publish(
                "epoch_note",
                {"message": "not enough of both classes to draw epoch curves"},
            )
            return
        scaler = StandardScaler().fit(x[train_index])
        x_train, x_held = scaler.transform(x[train_index]), scaler.transform(x[held_index])
        y_train, y_held = y[train_index], y[held_index]
        models: dict[str, Any] = {
            "MLP (16 hidden units)": MLPClassifier(
                hidden_layer_sizes=(16,),
                alpha=1e-3,
                random_state=seed,
                learning_rate_init=1e-3,
            ),
            "Logistic (SGD)": SGDClassifier(loss="log_loss", alpha=1e-4, random_state=seed, learning_rate="optimal"),
        }
        people = {
            "train": int(development["subject"].iloc[train_index].nunique()),
            "held_out": int(development["subject"].iloc[held_index].nunique()),
        }
        self.hub.publish(
            "epoch_start",
            {"models": list(models), "epochs": DISPLAY_EPOCHS, "people": people},
        )
        rng = np.random.default_rng(seed)
        for epoch in range(1, DISPLAY_EPOCHS + 1):
            order = rng.permutation(len(x_train))
            for name, model in models.items():
                model.partial_fit(x_train[order], y_train[order], classes=np.array([0, 1]))
                p_train = model.predict_proba(x_train)[:, 1]
                p_held = model.predict_proba(x_held)[:, 1]
                self.hub.publish(
                    "epoch",
                    {
                        "model": name,
                        "epoch": epoch,
                        "train_loss": float(log_loss(y_train, p_train, labels=[0, 1])),
                        "held_loss": float(log_loss(y_held, p_held, labels=[0, 1])),
                        "held_auc": float(roc_auc_score(y_held, p_held)),
                    },
                )
            time.sleep(EPOCH_PAUSE_SECONDS)

    def _test_views(self, artifact: ModelArtifact, test: pd.DataFrame) -> None:
        probabilities = predict_probability(artifact, test[artifact.inputs].to_numpy(dtype=float))
        labels = test["y"].to_numpy(dtype=int)
        bins = np.linspace(0, 1, 21)
        alert_counts, _ = np.histogram(probabilities[labels == 0], bins=bins)
        drowsy_counts, _ = np.histogram(probabilities[labels == 1], bins=bins)
        hours = max(int((labels == 0).sum()) * FEATURE_SPEC.hop_seconds / 3600.0, 1e-9)
        sweep = []
        for threshold in np.linspace(0.02, 0.99, SWEEP_POINTS):
            fired = probabilities >= threshold
            sweep.append(
                {
                    "threshold": float(threshold),
                    "false_windows_per_hour": float(fired[labels == 0].sum() / hours),
                    "drowsy_windows_caught": float(fired[labels == 1].mean()) if (labels == 1).any() else math.nan,
                }
            )
        self.hub.publish(
            "test_views",
            {
                "bins": bins.tolist(),
                "alert": alert_counts.tolist(),
                "drowsy": drowsy_counts.tolist(),
                "threshold": artifact.threshold,
                "sweep": sweep,
                "people": int(test["subject"].nunique()),
                "note": "Windows, before the 120 s cooldown; the pipeline's false-alert rate applies the cooldown.",
            },
        )


def _summary(stage: Stage, state: Any) -> dict[str, Any]:
    builder = _SUMMARIES.get(stage)
    if builder is None:
        return {}
    try:
        return builder(state)
    except Exception as error:  # noqa: BLE001 - a summary must never fail the run
        return {"summary_error": f"{type(error).__name__}: {error}"}


def _features_summary(state: Any) -> dict[str, Any]:
    frames = pd.read_parquet(state.workspace.frames_path, columns=["subject"])
    return {"frames": len(frames), "people": int(frames["subject"].nunique())}


def _windows_summary(state: Any) -> dict[str, Any]:
    windows = pd.read_parquet(state.workspace.windows_path, columns=["subject"])
    return {"windows": len(windows), "people": int(windows["subject"].nunique())}


def _split_summary(state: Any) -> dict[str, Any]:
    development, test = state.split.development, state.split.test
    return {
        "development_windows": len(development),
        "development_people": int(development["subject"].nunique()),
        "test_windows": len(test),
        "test_people": int(test["subject"].nunique()),
    }


def _train_summary(state: Any) -> dict[str, Any]:
    artifact = state.training.artifact
    winner = state.training.winner
    table = state.validation.to_frame() if state.validation is not None else None
    return {
        "sha": artifact.short_sha,
        "chosen": artifact.weights.type,
        "threshold": artifact.threshold,
        "inputs": len(artifact.inputs),
        "auc": winner.auc,
        "auc_eyes_masked": winner.auc_eyes_masked,
        "balanced_accuracy_eyes_masked": winner.balanced_accuracy_eyes_masked,
        "validation": [] if table is None else table.to_dict(orient="records"),
    }


def _gate_summary(state: Any) -> dict[str, Any]:
    return {
        "passed": state.manifest.passed,
        "checks": [_plain(check) for check in state.manifest.checks],
    }


_SUMMARIES: dict[Stage, Callable[[Any], dict[str, Any]]] = {
    Stage.FEATURES: _features_summary,
    Stage.WINDOWS: _windows_summary,
    Stage.SPLIT: _split_summary,
    Stage.TRAIN: _train_summary,
    Stage.TEST: lambda state: _plain(state.test_report),
    Stage.REPLAY: lambda state: {"sessions": [_plain(result) for result in state.replay]},
    Stage.ECONOMICS: lambda state: _plain(state.economics),
    Stage.STAGE_GATE: _gate_summary,
}


def _plain(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _plain(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value
