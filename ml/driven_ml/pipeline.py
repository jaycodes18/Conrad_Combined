"""Run an environment's stages in order, passing results between them in one ``PipelineState``.

Each stage is one small function in ``STAGE_RUNNERS``. The environment decides which stages run (see
``settings.STAGES_BY_ENVIRONMENT``); compliance always runs first, so no environment can skip it.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .app_fixtures import (
    FixtureExport,
    copy_fixtures_into_app,
    export_fixtures,
    fixtures_directory,
)
from .compliance import enforce_compliance
from .data.discovery import DiscoveryResult, run_discovery
from .economics import EconomicsReport, run_economics
from .errors import ConfigurationError
from .evaluation import TestReport, evaluate_on_test, load_test_report
from .features import build_frame_table
from .modeling import TrainingResult, ValidationReport, load_candidate, train, validate
from .modeling.training import write_validation_report
from .product import Product, load_product
from .release import Release, promote
from .settings import Environment, RunConfig, Stage
from .splits import DatasetSplit, split_windows
from .staging import (
    ReplayResult,
    StagingManifest,
    package_candidate,
    parity_inputs,
    replay_recordings,
    run_stage_gate,
)
from .windows import build_window_table
from .workspace import Workspace

logger = logging.getLogger(__name__)


@dataclass
class PipelineState:
    config: RunConfig
    workspace: Workspace
    product: Product
    discovery: DiscoveryResult | None = None
    split: DatasetSplit | None = None
    training: TrainingResult | None = None
    validation: ValidationReport | None = None
    test_report: TestReport | None = None
    fixtures: FixtureExport | None = None
    economics: EconomicsReport | None = None
    manifest: StagingManifest | None = None
    release: Release | None = None
    replay: list[ReplayResult] = field(default_factory=list)

    def require_split(self) -> DatasetSplit:
        if self.split is None:
            raise ConfigurationError("this stage needs the split stage earlier in the same run")
        return self.split

    def current_test_report(self) -> TestReport:
        if self.test_report is not None:
            return self.test_report
        return load_test_report(self.workspace, load_candidate(self.workspace))


def _compliance(state: PipelineState) -> None:
    enforce_compliance(state.product, state.config.require_repo(), state.workspace.extra_strings_path)


def _discover(state: PipelineState) -> None:
    if not state.config.run_discovery:
        logger.info("discover: skipped (--no-discovery)")
        return
    state.discovery = run_discovery(state.workspace)


def _features(state: PipelineState) -> None:
    build_frame_table(state.config, state.workspace)


def _windows(state: PipelineState) -> None:
    build_window_table(state.workspace)


def _split(state: PipelineState) -> None:
    state.split = split_windows(state.config, state.workspace)


def _train(state: PipelineState) -> None:
    state.training = train(state.config, state.workspace, state.require_split().development)
    state.validation = state.training.validation


def _validate(state: PipelineState) -> None:
    if state.validation is None:
        state.validation = validate(state.workspace, state.require_split().development, state.config.seed)
    else:
        write_validation_report(state.workspace, state.validation)


def _test(state: PipelineState) -> None:
    state.test_report = evaluate_on_test(state.workspace, state.require_split().test)


def _package(state: PipelineState) -> None:
    package_candidate(state.workspace, state.require_split().test, state.config.seed)


def _export_app_fixtures(state: PipelineState) -> None:
    config, workspace = state.config, state.workspace
    artifact = load_candidate(workspace)
    inputs = parity_inputs(artifact, state.require_split().test, config.seed)
    state.fixtures = export_fixtures(artifact, inputs, fixtures_directory(workspace, artifact.short_sha), config.seed)
    if config.write_to_app and config.app_dir is not None:
        copy_fixtures_into_app(
            state.fixtures,
            config.app_dir,
            include_dev_model=config.environment is Environment.DEV,
        )


def _replay(state: PipelineState) -> None:
    state.replay = replay_recordings(state.config, state.workspace)


def _economics(state: PipelineState) -> None:
    report = state.current_test_report()
    state.economics = run_economics(
        state.workspace,
        state.product,
        report.driven.false_alerts_per_hour,
        report.model_sha256[:12],
    )


def _stage_gate(state: PipelineState) -> None:
    state.manifest = run_stage_gate(state.workspace, state.product, state.replay, state.economics)


def _promote(state: PipelineState) -> None:
    state.release = promote(state.config, state.workspace, state.product)


STAGE_RUNNERS: dict[Stage, Callable[[PipelineState], None]] = {
    Stage.COMPLIANCE: _compliance,
    Stage.DISCOVER: _discover,
    Stage.FEATURES: _features,
    Stage.WINDOWS: _windows,
    Stage.SPLIT: _split,
    Stage.TRAIN: _train,
    Stage.VALIDATE: _validate,
    Stage.TEST: _test,
    Stage.PACKAGE: _package,
    Stage.EXPORT_APP_FIXTURES: _export_app_fixtures,
    Stage.REPLAY: _replay,
    Stage.ECONOMICS: _economics,
    Stage.STAGE_GATE: _stage_gate,
    Stage.PROMOTE: _promote,
}


def start_state(config: RunConfig) -> PipelineState:
    return PipelineState(
        config=config,
        workspace=Workspace(config).ensure(),
        product=load_product(config.require_repo()),
    )


def run_pipeline(config: RunConfig) -> PipelineState:
    stages = config.stages
    logger.info(
        "%s run on %s data, %s track: %s",
        config.environment,
        config.data_source,
        config.track,
        " -> ".join(stages),
    )
    state = start_state(config)
    for number, stage in enumerate(stages, start=1):
        logger.info("[%d/%d] %s", number, len(stages), stage)
        started = time.perf_counter()
        STAGE_RUNNERS[stage](state)
        logger.debug("%s finished in %.2f s", stage, time.perf_counter() - started)
    return state
