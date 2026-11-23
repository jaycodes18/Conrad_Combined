"""The compliance gate: product.json flags, banned claims in user-facing text, and forbidden code patterns.

It runs first in every environment. A single finding fails the run, and the error lists every finding so one
fix-up pass is enough.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ComplianceError
from ..product import Product
from ..settings import PRODUCT_FILE_NAME
from .rules import (
    BACKEND_PRODUCT_COPY,
    BANNED_CLAIM_PATTERNS,
    FORBIDDEN_CODE_PATTERNS,
    GATED_SETTINGS,
    LEGAL_RULES,
    SOURCE_GLOBS,
    USER_FACING_TEXT_FILES,
)

logger = logging.getLogger(__name__)

_SKIPPED_DIRECTORIES = frozenset({"node_modules", "dist", ".git"})
_CLAIM_REGEXES = tuple(re.compile(pattern, re.IGNORECASE) for pattern in BANNED_CLAIM_PATTERNS)
_CODE_REGEXES = tuple((rule, re.compile(rule.pattern)) for rule in FORBIDDEN_CODE_PATTERNS)


@dataclass(frozen=True)
class Finding:
    rule_id: str
    location: str
    message: str

    def __str__(self) -> str:
        return f"{self.location}: {self.message} ({self.rule_id})"


def check_product_flags(flags: Mapping[str, Any]) -> list[Finding]:
    findings = []
    for rule in LEGAL_RULES:
        if rule.flag is None:
            continue
        value = flags.get(rule.flag)
        if value is not rule.required_value:
            findings.append(
                Finding(
                    rule.rule_id,
                    f"{PRODUCT_FILE_NAME} flags.{rule.flag}",
                    f"must be {json.dumps(rule.required_value)}, is {json.dumps(value)}: {rule.description}",
                )
            )
    for gate in GATED_SETTINGS:
        value = flags.get(gate.setting)
        if value not in gate.allowed_without_opinion and flags.get(gate.opinion_flag) is not True:
            findings.append(
                Finding(
                    gate.rule_id,
                    f"{PRODUCT_FILE_NAME} flags.{gate.setting}",
                    f"{json.dumps(value)} needs a legal opinion first ({gate.opinion_flag}): {gate.description}",
                )
            )
    return findings


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def scan_claims(text: str, location: str) -> list[Finding]:
    """Flag every banned marketing claim (L8), case-insensitively, with its line number."""
    return [
        Finding(
            "L8",
            f"{location}:{_line_of(text, match.start())}",
            f"banned claim '{match.group(0)}'",
        )
        for regex in _CLAIM_REGEXES
        for match in regex.finditer(text)
    ]


def scan_source(text: str, location: str) -> list[Finding]:
    """Flag every forbidden API or tracker in app or backend source, case-sensitively."""
    return [
        Finding(
            rule.rule_id,
            f"{location}:{_line_of(text, match.start())}",
            f"{rule.meaning} '{match.group(0)}'",
        )
        for rule, regex in _CODE_REGEXES
        for match in regex.finditer(text)
    ]


def _source_files(root: Path) -> Iterable[Path]:
    for pattern in SOURCE_GLOBS:
        for path in sorted(root.glob(pattern)):
            relative = path.relative_to(root)
            if _SKIPPED_DIRECTORIES.isdisjoint(relative.parts) and not path.name.endswith(".test.ts"):
                yield path


def _backend_copy_finding(root: Path) -> Finding | None:
    copy_path = root / BACKEND_PRODUCT_COPY
    if not copy_path.is_file():
        return None
    try:
        in_sync = json.loads(copy_path.read_text(encoding="utf-8")) == json.loads(
            (root / PRODUCT_FILE_NAME).read_text(encoding="utf-8")
        )
    except json.JSONDecodeError:
        in_sync = False
    if in_sync:
        return None
    return Finding("config", BACKEND_PRODUCT_COPY, "differs from product.json; run `make sync`")


def scan_repository(root: Path) -> list[Finding]:
    findings: list[Finding] = []
    for relative in USER_FACING_TEXT_FILES:
        path = root / relative
        if path.is_file():
            findings.extend(scan_claims(path.read_text(encoding="utf-8"), relative))
    for path in _source_files(root):
        findings.extend(scan_source(path.read_text(encoding="utf-8"), path.relative_to(root).as_posix()))
    backend_finding = _backend_copy_finding(root)
    if backend_finding is not None:
        findings.append(backend_finding)
    return findings


def enforce_compliance(product: Product, repo_root: Path, extra_text: Path | None = None) -> None:
    """Raise one ComplianceError listing every finding; ``extra_text`` is an optional file of app strings."""
    findings = check_product_flags(product.flags) + scan_repository(repo_root)
    if extra_text is not None and extra_text.is_file():
        findings.extend(scan_claims(extra_text.read_text(encoding="utf-8"), extra_text.name))
    if findings:
        raise ComplianceError("Compliance gate failed", [str(finding) for finding in findings])
    logger.info(
        "compliance gate passed: %d product flags and %d gated setting checked, repository scan clean",
        sum(rule.flag is not None for rule in LEGAL_RULES),
        len(GATED_SETTINGS),
    )
