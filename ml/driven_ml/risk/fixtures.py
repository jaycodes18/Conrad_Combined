"""Golden ``DriverRiskSignal`` fixtures for the app team, produced by running real sessions, not written by hand.

Each fixture replays a scripted synthetic session through ``RiskSession`` and keeps the signal at one telling
moment. ``python -m driven_ml.risk.fixtures`` rewrites ``ml/contracts/fixtures``; ``--check`` exits non-zero if
the files differ from what the code produces now, so the contract cannot drift silently.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from ..features.synthetic import (
    GAZE_ERRATIC_10S,
    HEAD_DROP_NO_RESPONSE,
    NODDING_RUN,
    SUNGLASSES_ON,
    scripted_synthetic_session,
)
from ..indicators import CameraHealth, FrameSignals
from ..product import load_product
from ..workspace import write_text_atomic
from .escalation import ConsentState, ContactBook, EscalationConfig
from .session import RiskSession
from .signal import DriverRiskSignal

logger = logging.getLogger(__name__)

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "contracts" / "fixtures"
FIXTURE_EPOCH = datetime(2026, 11, 23, 15, 0, 0, tzinfo=UTC)
FIXTURE_SEED = 20
CALIBRATION_SECONDS = 120.0
CLEAR_CAMERA = CameraHealth(mean_luma=0.5, face_luma=0.5, sharpness=100.0)
LOOK_AWAY_YAW_DEGREES = 40.0


@dataclass(frozen=True)
class FixtureCase:
    name: str
    script: tuple[tuple[float, float, str], ...]
    pick: Callable[[DriverRiskSignal], bool]
    look_away: tuple[float, float] | None = None  # (start, seconds) of a 40 degree head turn
    until_seconds: float = 400.0


CASES: tuple[FixtureCase, ...] = (
    FixtureCase("none", (), lambda signal: signal.t_seconds >= 150.0),
    FixtureCase(
        "distracted",
        (),
        lambda signal: signal.condition.value == "distracted" and signal.escalation.in_car_alert,
        look_away=(150.0, 4.0),
    ),
    FixtureCase(
        "drowsy",
        ((150.0, 5.0, "closure_5s"),),
        lambda signal: signal.condition.value == "drowsy" and signal.escalation.in_car_alert,
    ),
    FixtureCase(
        "possible_impairment",
        ((140.0, 40.0, NODDING_RUN), (150.0, 12.0, GAZE_ERRATIC_10S)),
        lambda signal: signal.condition.value == "possible_impairment" and signal.escalation.notify_contacts,
    ),
    FixtureCase(
        "possible_unresponsiveness",
        ((150.0, 30.0, HEAD_DROP_NO_RESPONSE),),
        lambda signal: signal.escalation.voice_call_recommended,
    ),
    FixtureCase(
        "degraded_sunglasses",
        ((140.0, 60.0, SUNGLASSES_ON),),
        lambda signal: signal.t_seconds >= 150.0 and signal.monitoring_degraded.active,
    ),
)


def fixture_contacts() -> ContactBook:
    book = ContactBook()
    book.add(
        "contact-1",
        "Parent",
        "+12145550100",
        consent=ConsentState.RECORDED,
        consent_date="2026-11-01",
        calls_consented=True,
    )
    return book


def iter_case_signals(case: FixtureCase) -> Iterator[DriverRiskSignal]:
    frames, _events = scripted_synthetic_session(FIXTURE_SEED, script=case.script, minutes=7)
    session = RiskSession(
        session_id=f"fixture-{case.name.replace('_', '-')}",
        driver_ref="drv_fixture01",
        escalation=EscalationConfig.from_product(load_product()),
        contacts=fixture_contacts(),
    )
    for record in frames.to_dict(orient="records"):
        row = {str(key): value for key, value in record.items()}
        t = float(row["t"])
        if t > case.until_seconds:
            break
        if case.look_away is not None and case.look_away[0] <= t < case.look_away[0] + case.look_away[1]:
            row["yaw"] = float(row["yaw"]) + LOOK_AWAY_YAW_DEGREES
        frame = FrameSignals.from_features(
            t, row, eye_visibility=float(row["eye_visibility"]), mouth_visibility=float(row["mouth_visibility"])
        )
        yield session.push_frame(
            frame,
            CLEAR_CAMERA,
            calibrating=t < CALIBRATION_SECONDS,
            driving=True,
            emitted_at=FIXTURE_EPOCH + timedelta(seconds=t),
        )


def run_case(case: FixtureCase) -> DriverRiskSignal:
    for signal in iter_case_signals(case):
        if signal.t_seconds >= CALIBRATION_SECONDS and case.pick(signal):
            return signal
    raise RuntimeError(f"fixture {case.name}: the scripted session never produced the expected signal")


def build_fixtures() -> dict[str, dict[str, Any]]:
    return {case.name: run_case(case).to_json() for case in CASES}


def render(document: dict[str, Any]) -> str:
    return json.dumps(document, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m driven_ml.risk.fixtures", description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare with the files instead of writing them")
    parser.add_argument("--out", type=Path, default=FIXTURES_DIR)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    stale = []
    for name, document in build_fixtures().items():
        path = args.out / f"{name}.json"
        text = render(document)
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                stale.append(path.name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_text_atomic(path, text)
            logger.info("wrote %s", path)
    if stale:
        print(f"stale contract fixtures: {', '.join(stale)}; run python -m driven_ml.risk.fixtures", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
