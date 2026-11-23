"""Typed errors for the Driven pipeline.

Every failure a person can act on has its own class, so the CLI can map it to an exit code and a clear
message without parsing strings.
"""

from __future__ import annotations

from collections.abc import Iterable


class DrivenError(Exception):
    """Base class for every error this package raises on purpose."""


class ConfigurationError(DrivenError):
    """A run setting or a product.json value is missing or invalid."""


class DataError(DrivenError):
    """Input data is missing, malformed or not allowed for this run."""


class MissingArtifactError(DrivenError):
    """A file that an earlier stage writes does not exist yet."""

    def __init__(self, artifact: str, produced_by: str) -> None:
        super().__init__(f"{artifact} not found. Run the {produced_by} stage first.")
        self.artifact = artifact
        self.produced_by = produced_by


class ComplianceError(DrivenError):
    """A legal or product guardrail failed; the message lists every reason, one per line."""

    def __init__(self, title: str, reasons: Iterable[str]) -> None:
        self.title = title
        self.reasons = list(reasons)
        super().__init__("\n".join([f"{title}:", *(f"- {reason}" for reason in self.reasons)]))
