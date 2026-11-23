"""Run settings: which environment runs which stages, on which data, for which track.

The six environments (dev, train, validate, test, staging, production) are fixed stage lists, so a run can
never skip the compliance gate or promote from the wrong place. This module stays free of pandas and numpy so
the CLI starts quickly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import TypeVar

from .errors import ConfigurationError

PRODUCT_FILE_NAME = "product.json"

_EnumT = TypeVar("_EnumT", bound=StrEnum)


class Environment(StrEnum):
    DEV = "dev"
    TRAIN = "train"
    VALIDATE = "validate"
    TEST = "test"
    STAGING = "staging"
    PRODUCTION = "production"


class Track(StrEnum):
    RESEARCH = "research"
    COMMERCIAL = "commercial"


class DataSource(StrEnum):
    SYNTHETIC = "synthetic"
    REAL = "real"


class Stage(StrEnum):
    """Pipeline stages, in the order they run."""

    COMPLIANCE = "compliance"
    DISCOVER = "discover"
    FEATURES = "features"
    WINDOWS = "windows"
    SPLIT = "split"
    TRAIN = "train"
    VALIDATE = "validate"
    TEST = "test"
    PACKAGE = "package"
    EXPORT_APP_FIXTURES = "export_app_fixtures"
    REPLAY = "replay"
    ECONOMICS = "economics"
    STAGE_GATE = "stage_gate"
    PROMOTE = "promote"


STAGES_BY_ENVIRONMENT: dict[Environment, tuple[Stage, ...]] = {
    Environment.DEV: tuple(stage for stage in Stage if stage is not Stage.PROMOTE),
    Environment.TRAIN: (
        Stage.COMPLIANCE,
        Stage.FEATURES,
        Stage.WINDOWS,
        Stage.SPLIT,
        Stage.TRAIN,
    ),
    Environment.VALIDATE: (
        Stage.COMPLIANCE,
        Stage.WINDOWS,
        Stage.SPLIT,
        Stage.VALIDATE,
    ),
    Environment.TEST: (Stage.COMPLIANCE, Stage.WINDOWS, Stage.SPLIT, Stage.TEST),
    Environment.STAGING: (
        Stage.COMPLIANCE,
        Stage.WINDOWS,
        Stage.SPLIT,
        Stage.PACKAGE,
        Stage.EXPORT_APP_FIXTURES,
        Stage.REPLAY,
        Stage.ECONOMICS,
        Stage.STAGE_GATE,
    ),
    Environment.PRODUCTION: (Stage.COMPLIANCE, Stage.PROMOTE),
}


def find_repo_root(start: Path | None = None) -> Path | None:
    """Return the first directory at or above ``start`` that holds product.json, or None."""
    origin = (start or Path(__file__)).resolve()
    if origin.is_file():
        origin = origin.parent
    for directory in (origin, *origin.parents):
        if (directory / PRODUCT_FILE_NAME).is_file():
            return directory
    return None


def _coerce(enum_type: type[_EnumT], value: _EnumT | str, label: str) -> _EnumT:
    try:
        return enum_type(value)
    except ValueError as exc:
        choices = ", ".join(member.value for member in enum_type)
        raise ConfigurationError(f"unknown {label} {value!r}; expected one of: {choices}") from exc


@dataclass(frozen=True)
class RunConfig:
    """Everything one pipeline run needs to know. Build it with ``RunConfig.create``."""

    environment: Environment = Environment.DEV
    track: Track = Track.RESEARCH
    workspace_root: Path = field(default_factory=lambda: Path("driven_work"))
    seed: int = 7
    approved_by: str = ""
    reuse_features: bool = False
    run_discovery: bool = True
    repo_root: Path | None = None
    app_dir: Path | None = None
    write_to_app: bool = False

    @classmethod
    def create(
        cls,
        environment: Environment | str = Environment.DEV,
        track: Track | str = Track.RESEARCH,
        workspace_root: Path | str = "driven_work",
        *,
        seed: int = 7,
        approved_by: str = "",
        reuse_features: bool = False,
        run_discovery: bool = True,
        repo_root: Path | str | None = None,
        app_dir: Path | str | None = None,
        write_to_app: bool = False,
    ) -> RunConfig:
        repo = Path(repo_root).resolve() if repo_root is not None else find_repo_root()
        if app_dir is not None:
            resolved_app_dir: Path | None = Path(app_dir).expanduser().resolve()
        elif repo is not None and (repo / "apps" / "web").is_dir():
            resolved_app_dir = repo / "apps" / "web"
        else:
            resolved_app_dir = None
        if write_to_app and resolved_app_dir is None:
            raise ConfigurationError("--write-app needs an app directory: pass --app-dir or create apps/web")
        return cls(
            environment=_coerce(Environment, environment, "environment"),
            track=_coerce(Track, track, "track"),
            workspace_root=Path(workspace_root).expanduser(),
            seed=seed,
            approved_by=approved_by.strip(),
            reuse_features=reuse_features,
            run_discovery=run_discovery,
            repo_root=repo,
            app_dir=resolved_app_dir,
            write_to_app=write_to_app,
        )

    @property
    def stages(self) -> tuple[Stage, ...]:
        return STAGES_BY_ENVIRONMENT[self.environment]

    @property
    def data_source(self) -> DataSource:
        """Dev always runs on synthetic people; every other environment needs real, labeled video."""
        return DataSource.SYNTHETIC if self.environment is Environment.DEV else DataSource.REAL

    def require_repo(self) -> Path:
        if self.repo_root is None:
            raise ConfigurationError(f"no {PRODUCT_FILE_NAME} found; run from inside the Driven repository")
        return self.repo_root
