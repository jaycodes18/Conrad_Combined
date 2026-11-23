"""Honest inventory, weakness audit, and compatibility matrix for the ML repository and its studio.

Statuses describe this repository as it is. A passing unit test is not a camera bench test, and a finished
training job is not a production deployment. ``VERIFIED — SYNTHETIC ONLY`` means a test exercises it on synthetic
frames or synthetic audio and nothing else. The stationary pilot in the research brief has not been run here.
Run ``python -m driven_ml.studio.status`` to print the report as markdown.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any

from .cameras import DEVICES
from .catalog import CARDS


@dataclass(frozen=True)
class StatusRow:
    feature_id: str
    feature: str
    location: str
    status: str
    dependencies: str
    verification: str

    def to_json(self) -> dict[str, str]:
        return {
            "feature_id": self.feature_id,
            "feature": self.feature,
            "location": self.location,
            "status": self.status,
            "dependencies": self.dependencies,
            "verification": self.verification,
        }


@dataclass(frozen=True)
class Weakness:
    issue_id: str
    module: str
    description: str
    severity: str
    steps: str
    expected: str
    actual: str
    root_cause: str
    correction: str
    status: str
    verification: str

    def to_json(self) -> dict[str, str]:
        return {
            "issue_id": self.issue_id,
            "module": self.module,
            "description": self.description,
            "severity": self.severity,
            "steps": self.steps,
            "expected": self.expected,
            "actual": self.actual,
            "root_cause": self.root_cause,
            "correction": self.correction,
            "status": self.status,
            "verification": self.verification,
        }


MATRIX: tuple[StatusRow, ...] = (
    StatusRow(
        "F1",
        "Phone-camera live path",
        "`studio/live.py`, `studio/sources.py`",
        "IMPLEMENTED — NOT VERIFIED",
        "Camera permission, MediaPipe model",
        "Unit tests cover retry, gap invalidation and timing; no physical camera was opened.",
    ),
    StatusRow(
        "F2",
        "Drowsiness pipeline (60 s windows, baseline, person-grouped training, locked test, staging)",
        "`features/`, `modeling/`, `pipeline.py`, `staging/`",
        "IMPLEMENTED — NOT VERIFIED",
        "Licensed data, trained artifact",
        "`tools/check.py --dev-run` passes staging on synthetic data; no real-data training or promotion.",
    ),
    StatusRow(
        "F3",
        "2-second eyes-off rule (NHTSA 78 FR 24818)",
        "`studio/live.py`, `indicators/suite.py`",
        "VERIFIED — SYNTHETIC ONLY",
        "Face landmarks, calibration",
        "`test_studio.py`, `test_fusion_rules.py`.",
    ),
    StatusRow(
        "F4",
        "Eyes indicator: closure, 5 s critical, PERCLOS, winks, gap bridging",
        "`indicators/eyes.py`",
        "VERIFIED — SYNTHETIC ONLY",
        "Eye visibility, calibrated open level",
        "`test_indicator_eyes.py`; robustness suite under eight conditions.",
    ),
    StatusRow(
        "F5",
        "Ocular indicator: 5 s / 10 s erratic gaze against calibration",
        "`indicators/ocular.py`",
        "VERIFIED — SYNTHETIC ONLY",
        "Iris landmarks, calibration",
        "`test_indicator_ocular.py`. Measures dispersion and reversals, not nystagmus.",
    ),
    StatusRow(
        "F6",
        "Head indicator: nods, head drop, slump",
        "`indicators/head.py`",
        "VERIFIED — SYNTHETIC ONLY",
        "Head pose",
        "`test_indicator_head.py`.",
    ),
    StatusRow(
        "F7",
        "Breathing indicator: audio with quiet-cabin rules, gasp pattern, visual fallback",
        "`indicators/breathing.py`",
        "BLOCKED BY POLICY (implemented; synthetic only)",
        "Both legal flags for the microphone; shoulder tracking for the visual path",
        (
            "`test_indicator_breathing.py` on synthetic PCM. No microphone opened; no shoulder tracker exists, so"
            " the visual path never runs live."
        ),
    ),
    StatusRow(
        "F8",
        "Responsiveness indicator: 5 s visual response check and latency",
        "`indicators/responsiveness.py`",
        "VERIFIED — SYNTHETIC ONLY",
        "Face tracking during the check",
        "`test_indicator_responsiveness.py`.",
    ),
    StatusRow(
        "F9",
        "Sensor quality: coverage and named situations",
        "`indicators/sensor_quality.py`",
        "VERIFIED — SYNTHETIC ONLY",
        "Camera statistics, eye and mouth visibility",
        "`test_indicator_sensor_quality.py`; situations come from heuristics, not a validated quality model.",
    ),
    StatusRow(
        "F10",
        "Risk fusion: four rules, per-condition hysteresis, coverage-scaled confidence",
        "`risk/fusion.py`",
        "VERIFIED — SYNTHETIC ONLY",
        "Indicator readings; hand-set scores",
        "`test_fusion_rules.py`. `fit_fusion` is tested on synthetic labels only; no real labelled sessions exist.",
    ),
    StatusRow(
        "F11",
        "Escalation tiers 1-3, consent, daily cap, no automated 911",
        "`risk/escalation.py`",
        "VERIFIED — SIMULATED",
        "Contact consent records; the app's SMS path",
        "`test_escalation.py`; nothing is sent from this repository.",
    ),
    StatusRow(
        "F12",
        "DriverRiskSignal v1 contract",
        "`risk/signal.py`, `ml/contracts/`",
        "VERIFIED — NOT CONSUMED BY APP",
        "App integration",
        (
            "`test_contract.py`: schema, round-trip, golden fixtures, no banned claims or raw data. The app does "
            "not read it yet."
        ),
    ),
    StatusRow(
        "F13",
        "Studio: Drive, Test and Train tabs",
        "`studio/server.py`, `studio/static/index.html`",
        "VERIFIED — SYNTHETIC ONLY",
        "Local server, browser",
        (
            "`test_studio_simplified.py` checks routes and the risk_signal stream; see the report for the browser"
            " smoke test."
        ),
    ),
    StatusRow(
        "F14",
        "Eye-feature dropout training and eyes-masked validation",
        "`modeling/training.py`",
        "VERIFIED — SYNTHETIC ONLY",
        "Development windows",
        "`test_robustness.py`; the dev run reports both metrics.",
    ),
    StatusRow(
        "F15",
        "Condition regression suite (eight conditions)",
        "`studio/conditions.py`, `studio/scenarios.py`",
        "VERIFIED — SYNTHETIC ONLY",
        "Feature-level condition effects",
        "`test_robustness.py`; conditions are modelled effects, not cabin recordings.",
    ),
    StatusRow(
        "F16",
        "Camera adapter contract and hardware registry",
        "`studio/cameras.py`",
        "IMPLEMENTED — NOT VERIFIED",
        "Hardware-specific stream and bench testing",
        "No dashcam, USB camera or phone attached.",
    ),
    StatusRow(
        "F17",
        "Dataset registry, cards and approval gate",
        "`data/registry.py`, `studio/catalog.py`",
        "VERIFIED",
        "Human licence review",
        "`test_product_surface.py`; no dataset downloaded. No Studio tab after the simplification.",
    ),
    StatusRow(
        "F18",
        "MobileNetV3 comparison head",
        "Not present",
        "NOT STARTED",
        "Approved data and driver-held-out evaluation",
        "No trained model or comparison.",
    ),
    StatusRow(
        "F19",
        "Stationary pilot (10 volunteers)",
        "Not present",
        "NOT STARTED",
        "Consent, volunteers, protocol",
        "No volunteers recorded; targets are not results.",
    ),
    StatusRow(
        "F20",
        "Production deployment and rollback",
        "Existing staging and promotion",
        "IMPLEMENTED — NOT VERIFIED",
        "Passing staging manifest, named approver",
        "No deployment or rollback performed.",
    ),
    StatusRow(
        "F21",
        "Intoxication detection",
        "Not present by policy",
        "BLOCKED",
        "Independently measured reference labels; FTC Act section 5",
        "Reported only as possible_impairment from observable indicators.",
    ),
    StatusRow(
        "F22",
        "Camera recovery and freshness",
        "`studio/live.py`, `studio/sources.py`",
        "IMPLEMENTED — NOT VERIFIED",
        "Selected stream, OS permission",
        "Mocked reconnect and long-gap invalidation pass; no physical stream.",
    ),
    StatusRow(
        "F23",
        "Research microphone capture and reconnect",
        "`studio/microphone.py`",
        "BLOCKED BY POLICY (implemented; mocked and synthetic tested)",
        "`sounddevice`, both legal flags, `record_audio=false`",
        "Device enumeration stays off before authorization; no physical microphone opened.",
    ),
)

WEAKNESSES: tuple[Weakness, ...] = (
    Weakness(
        "AD-001",
        "Camera",
        (
            "No physical camera was exercised. Permission, image quality, recovery on a real device, latency and "
            "mount performance remain unverified."
        ),
        "High",
        "Start the studio with no camera permission and press Start.",
        "A distinct camera fault, reconnect, unavailable readings, and no alert from missing frames.",
        "Mocked reconnect and gap cases pass; the camera path was not run on hardware.",
        "Hardware was not available to this change.",
        "Finish physical bench tests before claiming compatibility.",
        "IMPLEMENTED — NOT VERIFIED",
        "Run with camera permission on and off, then disconnect and reconnect during monitoring.",
    ),
    Weakness(
        "AD-002",
        "Models",
        (
            "The drowsiness head is the only trained model. Every indicator and the fusion are rules with hand-"
            "set thresholds."
        ),
        "High",
        "Look for a fitted fusion weight file or a validated indicator threshold.",
        "Weights fitted on person-grouped, labelled Driven sessions.",
        "Named constants in `risk/fusion.py`; `fit_fusion` exists but has no real labels to learn from.",
        "No labelled Driven sessions exist.",
        "Record consented sessions, fit with `fit_fusion`, and keep research weights out of staging.",
        "IMPLEMENTED — NOT VERIFIED",
        "A future fit must report out-of-fold AUC on people outside the locked test.",
    ),
    Weakness(
        "AD-003",
        "Validation",
        (
            "The 10-volunteer stationary pilot has not been run. Recall, nuisance rate and response latency are "
            "unmeasured."
        ),
        "Critical",
        "Search reports for 10 volunteers and 60 scripted events.",
        "A report of recall, nuisance rate, delay and availability.",
        "No such report exists.",
        "The pilot was specified, not executed.",
        "Run the protocol before quoting any recall or false-alert figure.",
        "NOT STARTED",
        "Record the protocol results, failures included.",
    ),
    Weakness(
        "AD-004",
        "Escalation",
        "Tiers are recommendations in the signal. Nothing is sent, and the app does not read the signal yet.",
        "Medium",
        "Trigger a head drop with no response in the Drive tab.",
        "Tier 2 then Tier 3 recommended, with no message sent.",
        "The ladder lights; the contact log records what would be sent.",
        "47 CFR 64.1200 and the studio privacy rule keep sending out of this process.",
        "Wire the app's SMS path to `notify_contacts`; measure delivery there.",
        "VERIFIED — SIMULATED",
        "`test_escalation.py`; `emergency_services` is always never_automated.",
    ),
    Weakness(
        "AD-005",
        "Dashcam",
        "VIOFO and other dashcams are documented, not supported.",
        "High",
        "Read the hardware registry row for VIOFO A229 Plus.",
        "A compatibility report after stream, lighting, heat and disconnect tests.",
        "Status is documented_only. No stream was opened.",
        "The front camera faces the road; an interior camera is a different product.",
        "Do not mark a device tested until the bench list is run.",
        "NOT STARTED",
        "A hardware report must name firmware, channel, resolution and failure cases.",
    ),
    Weakness(
        "AD-006",
        "Data",
        "No dataset was ingested. Kaggle sets stay blocked until a named review.",
        "Medium",
        "Call ingest_plan on state_farm.",
        "No download.",
        "allowed false and executed false.",
        "Licences are not automatically safe; State Farm is never_use.",
        "Keep reviewer names out of git if they should not be published.",
        "VERIFIED",
        "Unit test covers rejection of never_use and missing commercial licence.",
    ),
    Weakness(
        "AD-007",
        "Science",
        (
            "Sunglasses, IR, low light, backlight, glare, blur, fisheye and a hand are modelled twice: as OpenCV "
            "image transforms and as feature-level effects. Neither is a cabin recording."
        ),
        "High",
        "Read `studio/conditions.py` and run the Test tab.",
        "In-car recordings under each condition.",
        "Both suites pass on synthetic or captured frames only.",
        "No consented in-car clips exist.",
        "Collect consented clips per condition before changing a verdict.",
        "IMPLEMENTED — NOT VERIFIED",
        "Re-run both suites on real cabin clips and keep the as-captured column.",
    ),
    Weakness(
        "AD-008",
        "Business",
        "Prices live in product.json. This repository does not bill or send leads.",
        "Low",
        "Look for a checkout call in the studio server.",
        "No payment capture in the ML studio.",
        "No payment code.",
        "The ML repo is not the billing app.",
        "Leave billing in the app.",
        "VERIFIED",
        "Studio routes do not reference payments.",
    ),
    Weakness(
        "AD-009",
        "Microphone privacy",
        "Real microphone capture is blocked because both legal-opinion flags are false.",
        "High",
        "Open Drive with the default product.json and inspect the breathing card.",
        "No device enumeration or capture before authorization.",
        "Breathing shows unavailable: mic_blocked_by_policy; the API returns an empty device list.",
        "The legal gates are false by design.",
        "Obtain written review before enabling both flags; never record or transcribe audio.",
        "BLOCKED BY POLICY",
        "Unit tests assert gated enumeration is skipped.",
    ),
    Weakness(
        "AD-010",
        "Breathing",
        (
            "Agonal-pattern and quiet-cabin readings can confirm possible unresponsiveness with closed eyes, but "
            "never trigger it alone."
        ),
        "High",
        "Feed gasp-like synthetic audio to a parked session with open eyes.",
        "Breathing flagged experimental; no condition raised.",
        "As expected on synthetic PCM.",
        "The detector is hand-built after Chan et al. (2019); no Driven data validates it.",
        "Validate on consented, independently labelled audio before relying on it.",
        "IMPLEMENTED — EXPERIMENTAL",
        "`test_robustness.py` audio cases.",
    ),
    Weakness(
        "AD-011",
        "Ocular",
        (
            "A 30 fps phone camera cannot resolve saccades; the ocular indicator measures gaze dispersion and "
            "reversals only."
        ),
        "Medium",
        "Compare with Koch et al. (CHI 2023, CHI 2025).",
        "Comparable metrics measured on Driven data.",
        "No comparison exists.",
        "Their systems used dedicated eye trackers.",
        "Do not quote their accuracy for Driven.",
        "IMPLEMENTED — EXPERIMENTAL",
        "Measure on consented Driven sessions.",
    ),
    Weakness(
        "AD-012",
        "Head tracking",
        (
            "MediaPipe can lose the face in a steep head-down pose. A lost face is unavailable, never a head "
            "drop, so a real slump that loses tracking is not confirmed."
        ),
        "High",
        "Drop the head until the face is lost.",
        "Tier escalation on a confirmed slump.",
        "Readings go unavailable after 0.3 s and fusion cannot confirm.",
        "The unavailable invariant takes priority over recall.",
        "Bench-test head-down poses per mount position.",
        "IMPLEMENTED — NOT VERIFIED",
        "Physical test with a mannequin or a parked volunteer.",
    ),
    Weakness(
        "AD-013",
        "Camera quality",
        (
            "Degraded images now lower confidence and name a situation instead of zeroing every behaviour score; "
            "the 60 s model window is still skipped when camera quality is degraded."
        ),
        "Medium",
        "Run the low-light condition in the Test tab.",
        "No false alarms and closure still caught.",
        "Robustness suite passes on modelled conditions.",
        "Eye readings gate on eye visibility, so a dark frame no longer needs a global veto.",
        "Confirm on real low-light clips.",
        "VERIFIED — SYNTHETIC ONLY",
        "`test_robustness.py`.",
    ),
)


def matrix_json() -> list[dict[str, str]]:
    return [row.to_json() for row in MATRIX]


def weaknesses_json() -> list[dict[str, str]]:
    return [item.to_json() for item in WEAKNESSES]


def overview() -> dict[str, Any]:
    return {
        "product": "Driven",
        "studio_role": "Local model studio. Not the shipped phone app and not a deployment.",
        "preserves": [
            "drowsiness model on licensed or synthetic research data",
            "distraction as a 2-second eyes-off rule",
            "possible impairment and possible unresponsiveness as observable indicators, never a claim about alcohol",
            "no automated 911 call",
            "product.json as the price and flag source",
            "human approval before promotion",
        ],
        "datasets_listed": len(CARDS),
        "devices_listed": len(DEVICES),
        "pilot_executed": False,
        "hardware_bench_executed": False,
        "download_executed": False,
        "deployment_executed": False,
    }


def render_report() -> str:
    """Markdown the studio and the repository report share, so the two cannot drift."""
    lines = [
        "# Driven ML implementation report",
        "",
        "The compatibility matrix and weakness audit are generated from `driven_ml.studio.status`",
        "(`python -m driven_ml.studio.status`). This report claims no camera bench test, no microphone capture,",
        "no Kaggle download, no MobileNet run, no pilot and no Driven-recorded impairment or unresponsiveness data.",
        "",
        "## A. What exists",
        "",
        "- Dataset registry, licence filter, and discovery that does not download.",
        "- Landmark features, 60-second windows, personal baseline, person-grouped training with eye-feature dropout.",
        "- Locked test, staging gates, human promotion.",
        "- Six indicator submodels (`driven_ml/indicators`), risk fusion and escalation tiers (`driven_ml/risk`).",
        "- The `DriverRiskSignal` v1 contract with schema, TypeScript type and golden fixtures (`ml/contracts`).",
        "- Studio with three tabs: Drive, Test and Train.",
        "- Eyes off the road for 2 seconds as a rule, not a model.",
        "",
        "## B. Weakness audit",
        "",
    ]
    for item in WEAKNESSES:
        lines.extend(
            [
                f"### {item.issue_id} ({item.severity}) — {item.module}",
                "",
                item.description,
                "",
                f"- Steps: {item.steps}",
                f"- Expected: {item.expected}",
                f"- Actual: {item.actual}",
                f"- Root cause: {item.root_cause}",
                f"- Correction: {item.correction}",
                f"- Status: {item.status}",
                f"- Verification: {item.verification}",
                "",
            ]
        )
    lines.extend(
        [
            "## C. Architecture",
            "",
            (
                "Camera -> one-frame capture -> landmarks -> frame signals -> six indicators -> fusion -> escalation "
                "tiers -> DriverRiskSignal (SSE `risk_signal` in Studio). The 60-second drowsiness model feeds fusion "
                "as one family. The gated microphone feeds the breathing indicator in memory only."
            ),
            "Training, locked evaluation, staging and human promotion stay in the pipeline; Studio does not deploy.",
            "",
            "## D. Compatibility matrix",
            "",
            "| ID | Feature | Implementation | Status | Dependencies | Verification |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in MATRIX:
        lines.append(
            f"| {row.feature_id} | {row.feature} | {row.location} | {row.status} | "
            f"{row.dependencies} | {row.verification} |"
        )
    lines.extend(
        [
            "",
            "## E. Limitations",
            "",
            "- No physical camera, dashcam or microphone was opened.",
            "- No real glasses on a real face; conditions are modelled effects on synthetic frames.",
            "- Breathing is unverified on road audio; the microphone stays behind both legal flags.",
            "- No Driven-recorded impairment or unresponsiveness data; fusion scores are hand-set.",
            "- No intoxication model, and no claim about alcohol.",
            "- No real SMS, call or emergency dispatch.",
            "- Pilot targets are not results.",
            "",
            "## F. How to demonstrate what is done",
            "",
            "1. `cd ml && ../.venv/bin/python -m driven_ml.studio --source synthetic` and open http://127.0.0.1:8765.",
            "2. Drive shows the fused condition, six indicators, the escalation ladder and the signal JSON.",
            "3. Test runs the scripted condition suite and the scenario lab; hardware rows show NOT RUN.",
            "4. Train runs the pipeline and shows validation with full and eyes-masked metrics.",
            "5. `../.venv/bin/python -m pytest -q`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    sys.stdout.write(render_report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
