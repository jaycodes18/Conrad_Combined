"""DriverRiskSignal v1: fixtures validate and round-trip, stay golden, carry no banned claims and no raw data."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from driven_ml.compliance.checks import scan_claims
from driven_ml.compliance.rules import BANNED_CLAIM_PATTERNS
from driven_ml.errors import DataError
from driven_ml.indicators import Situation
from driven_ml.risk import DISCLAIMER, Condition, DriverRiskSignal, RiskLevel
from driven_ml.risk.fixtures import CASES, build_fixtures, render

ML = Path(__file__).resolve().parents[1]
CONTRACTS = ML / "contracts"
SCHEMA = json.loads((CONTRACTS / "driver_risk_signal.v1.schema.json").read_text(encoding="utf-8"))
FIXTURES = {path.stem: json.loads(path.read_text(encoding="utf-8")) for path in (CONTRACTS / "fixtures").glob("*.json")}
NEW_CODE = [
    *sorted((ML / "driven_ml" / "indicators").glob("*.py")),
    *sorted((ML / "driven_ml" / "risk").glob("*.py")),
    CONTRACTS / "driver_risk_signal.v1.ts",
    CONTRACTS / "driver_risk_signal.v1.schema.json",
    *sorted((CONTRACTS / "fixtures").glob("*.json")),
]
# User-facing wording: the product reports observable conditions, never these words (AGENTS.md invariant 4).
USER_FACING_BANNED = re.compile(r"\b(drunk|intoxicat\w*|unconscious\w*|BAC|diagnos\w*)\b", re.IGNORECASE)
RAW_DATA_TOKENS = frozenset(
    {"pixels", "pixel", "image", "jpeg", "rgb", "frame", "landmarks", "points", "audio", "pcm", "samples",
     "transcript", "latitude", "longitude", "lat", "lng", "gps", "coordinates"}
)  # fmt: skip


def validator() -> Draft202012Validator:
    return Draft202012Validator(SCHEMA, format_checker=Draft202012Validator.FORMAT_CHECKER)


def keys(document: Any) -> Iterator[str]:
    if isinstance(document, dict):
        for key, value in document.items():
            yield str(key)
            yield from keys(value)
    elif isinstance(document, list):
        for item in document:
            yield from keys(item)


def test_there_is_a_fixture_for_every_condition_and_the_sunglasses_case() -> None:
    assert set(FIXTURES) == {case.name for case in CASES}
    assert {document["condition"] for document in FIXTURES.values()} == {condition.value for condition in Condition}
    sunglasses = FIXTURES["degraded_sunglasses"]["monitoring_degraded"]
    assert sunglasses == {"active": True, "situations": ["sunglasses_or_eye_occlusion"]}


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_fixture_validates_and_round_trips(name: str) -> None:
    document = FIXTURES[name]
    Draft202012Validator.check_schema(SCHEMA)
    errors = [error.message for error in validator().iter_errors(document)]
    assert errors == []
    assert DriverRiskSignal.from_json(document).to_json() == document


def test_fixtures_are_golden() -> None:
    produced = build_fixtures()
    for name, document in produced.items():
        assert render(document) == (CONTRACTS / "fixtures" / f"{name}.json").read_text(encoding="utf-8"), (
            f"{name}.json is stale: run python -m driven_ml.risk.fixtures"
        )


def test_schema_rejects_what_the_contract_forbids() -> None:
    base = FIXTURES["possible_unresponsiveness"]
    for path, bad in (
        (("escalation", "emergency_services"), "automated"),
        (("condition",), "drunk"),
        (("driver_ref",), "jane.doe@example.com"),
        (("indicators", "eyes", "severity"), 1.5),
    ):
        document = json.loads(json.dumps(base))
        target = document
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = bad
        assert list(validator().iter_errors(document)), f"schema accepted {path}={bad!r}"
    extra = {**json.loads(json.dumps(base)), "latitude": 32.9}
    assert list(validator().iter_errors(extra))
    unavailable_but_severe = json.loads(json.dumps(base))
    unavailable_but_severe["indicators"]["breathing"]["severity"] = 0.5
    assert list(validator().iter_errors(unavailable_but_severe))


def test_dataclass_refuses_names_and_automated_dispatch() -> None:
    document = json.loads(json.dumps(FIXTURES["drowsy"]))
    document["driver_ref"] = "Jane Doe"
    with pytest.raises(DataError, match="pseudonymous"):
        DriverRiskSignal.from_json(document)
    document = json.loads(json.dumps(FIXTURES["drowsy"]))
    document["escalation"]["emergency_services"] = "call_911"
    with pytest.raises(DataError, match="never_automated"):
        DriverRiskSignal.from_json(document)
    with pytest.raises(DataError, match="not a valid"):
        DriverRiskSignal.from_json({"schema_version": "driver_risk_signal.v1"})


def test_no_banned_claims_in_new_code_contracts_or_fixtures() -> None:
    assert BANNED_CLAIM_PATTERNS
    findings = [finding for path in NEW_CODE for finding in scan_claims(path.read_text(encoding="utf-8"), path.name)]
    assert findings == []
    user_facing = [DISCLAIMER]
    for document in FIXTURES.values():
        user_facing.append(document["escalation"]["reason"])
        for indicator in document["indicators"].values():
            user_facing.extend(indicator["evidence"])
    hits = [text for text in user_facing if USER_FACING_BANNED.search(text)]
    assert hits == []


def test_no_pixel_landmark_audio_or_gps_fields() -> None:
    schema_keys = set(keys(SCHEMA["properties"]))
    for name, document in FIXTURES.items():
        for key in {*keys(document), *schema_keys}:
            tokens = set(key.lower().split("_"))
            assert not tokens & RAW_DATA_TOKENS, f"{name}: field {key!r} looks like raw sensor data"


def test_schema_and_typescript_list_every_value_the_code_can_emit() -> None:
    properties = SCHEMA["properties"]
    assert properties["condition"]["enum"] == [condition.value for condition in Condition]
    assert properties["level"]["enum"] == [level.value for level in RiskLevel]
    situations = properties["monitoring_degraded"]["properties"]["situations"]["items"]["enum"]
    assert set(situations) == {situation.value for situation in Situation} - {"mic_blocked_by_policy"}
    assert properties["disclaimer"]["const"] == DISCLAIMER
    typescript = (CONTRACTS / "driver_risk_signal.v1.ts").read_text(encoding="utf-8")
    literals = set(situations) | set(properties["condition"]["enum"]) | set(properties["level"]["enum"])
    for indicator in properties["indicators"]["properties"].values():
        literals |= set(indicator["properties"]["state"]["enum"])
    missing = sorted(value for value in literals if f"'{value}'" not in typescript)
    assert missing == []
    for field in properties:
        assert re.search(rf"\b{field}\b", typescript), field
