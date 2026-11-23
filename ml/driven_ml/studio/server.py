"""The studio's local web server: one dashboard page, an MJPEG video stream, server-sent events and a small JSON API.

Three tabs, and only their routes: Drive (live stream, fused condition, indicators, escalation ladder, the
``DriverRiskSignal`` JSON, contact consent), Test (condition capture, scenario lab, scripted condition suite) and
Train (the pipeline). It binds to 127.0.0.1 by default and serves nothing but this dashboard. Camera frames and
microphone chunks stay in memory on this computer: nothing is written to disk and nothing leaves it (rule L6).
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import threading
import time
from collections.abc import Callable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, ClassVar

from ..errors import DrivenError
from ..features.preprocess import Preprocess, parse_region
from ..modeling.artifact import ModelArtifact
from ..product import load_product
from ..risk import ConsentState, ContactBook, EscalationConfig
from ..settings import RunConfig
from ..workspace import Workspace
from .conditions import FEATURE_CONDITION_KEYS, ImageLandmarker, assess, to_json
from .events import EventHub
from .live import LiveEngine
from .microphone import ResearchMicrophone, list_input_devices, microphone_backend_available
from .profiles import DriverBook
from .research_audio import research_audio_allowed, research_audio_block_reason
from .scenarios import ROBUSTNESS_SCRIPT, run_audio_scenarios, run_scripted_scenarios, run_vision_condition_scenarios
from .sources import NOVATEK_DEFAULT_IP, describe, start_novatek_live_view
from .training import TrainingRun

logger = logging.getLogger(__name__)
STATIC = Path(__file__).with_name("static")
MAX_BODY_BYTES = 64 * 1024
FRAME_INTERVAL_SECONDS = 0.08


class Studio:
    def __init__(self, workspace_root: Path, models_dir: Path) -> None:
        self.hub = EventHub()
        self.models_dir = models_dir
        self._conditions_thread: threading.Thread | None = None
        self.workspace_root = workspace_root
        self.live = LiveEngine(self.hub, models_dir)
        self.training = TrainingRun(self.hub, self.live)
        self.drivers = DriverBook(workspace_root / "driver_profiles.json")
        self.contacts = ContactBook()
        product = load_product()
        require_consent = product.flags.get("require_contact_consent") is not False
        allow_911 = product.flags.get("auto_call_911") is True
        self.escalation = EscalationConfig.from_product(product)  # refuses auto_call_911 = true
        flags = dict(product.flags)
        self.live.configure_risk(self.escalation, self.contacts, audio_allowed=research_audio_allowed(flags))
        self.live.on_baseline = self._save_baseline
        self.microphone = ResearchMicrophone()
        self.microphone.configure_flags(flags)
        self.microphone.on_update = self._audio_update
        self.microphone.on_chunk = self.live.push_audio
        self.scenario_results: list[dict[str, Any]] = []
        self.product_flags = {
            "require_contact_consent": require_consent,
            "auto_call_911": False,
            "auto_call_911_requested": allow_911,
            "record_audio": bool(flags.get("record_audio", False)),
            "studio_research_audio": bool(flags.get("studio_research_audio", False)),
            "legal_opinion_studio_audio": bool(flags.get("legal_opinion_studio_audio", False)),
        }

    def load_model_file(self, path: Path) -> None:
        document = json.loads(path.read_text(encoding="utf-8"))
        self.live.set_model(ModelArtifact.from_json(document), origin=str(path))

    def load_latest_candidate(self) -> bool:
        from ..modeling.store import load_candidate

        for environment in ("dev", "train"):
            try:
                workspace = Workspace(RunConfig.create(environment, workspace_root=self.workspace_root))
                self.live.set_model(
                    load_candidate(workspace),
                    origin=f"last {environment} candidate in {workspace.base}",
                )
                return True
            except DrivenError:
                continue
        return False

    def run_conditions(self) -> bool:
        if self._conditions_thread is not None and self._conditions_thread.is_alive():
            return False
        samples = list(self.live.captured)
        if not any(not sample.eyes_closed for sample in samples):
            raise ValueError("capture eyes-open frames first")

        def work() -> None:
            self.hub.publish("conditions_status", {"state": "running", "frames": len(samples)})
            landmarker = None
            try:
                landmarker = ImageLandmarker(self.models_dir)
                results = assess(
                    samples,
                    landmarker,
                    progress=lambda name: self.hub.publish("conditions_status", {"state": "running", "now": name}),
                )
                self.hub.publish(
                    "conditions",
                    {
                        "results": to_json(results),
                        "closed_frames": sum(s.eyes_closed for s in samples),
                        "open_frames": sum(not s.eyes_closed for s in samples),
                    },
                )
            except Exception as error:  # noqa: BLE001 - shown on the dashboard
                self.hub.publish(
                    "conditions_status",
                    {"state": "error", "message": f"{type(error).__name__}: {error}"},
                )
            finally:
                if landmarker is not None:
                    landmarker.close()

        self._conditions_thread = threading.Thread(target=work, name="driven-conditions", daemon=True)
        self._conditions_thread.start()
        return True

    def state(self) -> dict[str, Any]:
        model = self.live.model
        return {
            "live_running": self.live.running,
            "training_running": self.training.running,
            "driving": self.live.driving,
            "calibration_seconds": self.live.calibration_seconds,
            "model": None if model is None else self.live.model_summary(),
            "workspace": str(self.workspace_root),
            "preprocess": {
                "enhance_low_light": self.live.preprocess.enhance_low_light,
                "region": self.live.preprocess.region,
                "fisheye_fov_degrees": self.live.preprocess.fisheye_fov_degrees,
            },
            "captured": {
                "open": sum(not s.eyes_closed for s in self.live.captured),
                "closed": sum(s.eyes_closed for s in self.live.captured),
            },
            "contacts": self.contacts.to_json(),
            "drivers": self.drivers.to_json(),
            "camera": {
                "dropped_frames": self.live.dropped_frames,
                "camera_faults": self.live.camera_faults,
                "latency_ms": self.live.last_latency_ms,
                "source_fps_hz": self.live.last_source_fps_hz,
                "processing_fps_hz": self.live.last_processing_fps_hz,
            },
            "product_flags": self.product_flags,
            "signal": None if self.live.risk.last_signal is None else self.live.risk.last_signal.to_json(),
            "tier": self.live.risk.escalation.tier,
            "microphone": self.microphone_state(),
            "scenario_results": self.scenario_results,
        }

    def microphone_state(self) -> dict[str, Any]:
        flags = {
            **self.product_flags,
            "record_audio": self.product_flags["record_audio"],
        }
        allowed = research_audio_allowed(flags)
        return {
            "running": self.microphone.running,
            "allowed": allowed,
            "block_reason": "" if allowed else research_audio_block_reason(flags),
            "backend_available": microphone_backend_available() if allowed else False,
            "devices": [device.to_json() for device in list_input_devices()] if allowed else [],
            "health": None if self.microphone.last_health is None else self.microphone.last_health.to_json(),
            "breathing": None if self.microphone.last_breathing is None else self.microphone.last_breathing.to_json(),
        }

    def _audio_update(self, health: Any, estimate: Any) -> None:
        self.live.publish_audio_metrics(health, estimate)

    def _save_baseline(self, baseline: dict[str, float] | None, centre: dict[str, float]) -> None:
        driver_id = self.live.active_driver_id or self.drivers.active_id
        if not driver_id:
            return
        self.drivers.save_calibration(driver_id, baseline, centre)


class Handler(BaseHTTPRequestHandler):
    studio: ClassVar[Studio]
    server_version = "DrivenStudio/1"

    def log_message(self, format: str, *args: Any) -> None:
        logger.debug("%s - %s", self.address_string(), format % args)

    # ---- GET -----------------------------------------------------------------------------------------
    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send_bytes((STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/events":
            self._events()
        elif self.path in _GET_ROUTES:
            self._send_json(_GET_ROUTES[self.path](self.studio))
        elif self.path.startswith("/stream.mjpg"):
            self._mjpeg()
        else:
            self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)

    def _events(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        inbox = self.studio.hub.subscribe()
        self.studio.hub.publish("hello", self.studio.state())
        try:
            while True:
                try:
                    message = inbox.get(timeout=15)
                    self.wfile.write(f"data: {message}\n\n".encode())
                except queue.Empty:
                    self.wfile.write(b": keep-alive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.studio.hub.unsubscribe(inbox)

    def _mjpeg(self) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        last: bytes | None = None
        try:
            while True:
                jpeg = self.studio.live.latest_jpeg()
                if jpeg is not None and jpeg is not last:
                    last = jpeg
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                    self.wfile.write(jpeg + b"\r\n")
                    self.wfile.flush()
                time.sleep(FRAME_INTERVAL_SECONDS)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # ---- POST ----------------------------------------------------------------------------------------
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            self._send_json({"error": "too_large"}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return
        try:
            body: dict[str, Any] = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._send_json({"error": "invalid_json"}, HTTPStatus.BAD_REQUEST)
            return
        route = POST_ROUTES.get(self.path)
        if route is None:
            self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND)
            return
        try:
            self._send_json(route(self.studio, body))
        except (OSError, ValueError, KeyError, DrivenError) as error:
            self._send_json(
                {"ok": False, "error": f"{type(error).__name__}: {error}"},
                HTTPStatus.BAD_REQUEST,
            )

    # ---- helpers -------------------------------------------------------------------------------------
    def _send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(json.dumps(payload).encode(), "application/json", status)

    def _send_bytes(self, data: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


_GET_ROUTES: dict[str, Callable[[Studio], dict[str, Any]]] = {
    "/api/state": lambda studio: studio.state(),
    "/api/signal": lambda studio: {
        "signal": None if studio.live.risk.last_signal is None else studio.live.risk.last_signal.to_json()
    },
    "/api/microphone": lambda studio: studio.microphone_state(),
    "/api/scenarios": lambda studio: {
        "results": studio.scenario_results,
        "conditions": ["clean", *FEATURE_CONDITION_KEYS],
        "events": [event for _start, _seconds, event in ROBUSTNESS_SCRIPT],
    },
}


# ---- POST routes: each takes the studio and the JSON body and returns the JSON reply ---------------------
def _live_start(studio: Studio, body: dict[str, Any]) -> dict[str, Any]:
    source = str(body.get("source") or "0").strip()
    if body.get("novatek_live_view"):
        reply = start_novatek_live_view(str(body.get("dashcam_ip") or NOVATEK_DEFAULT_IP))
        studio.hub.publish(
            "log",
            {"level": "INFO", "logger": "studio", "message": f"dashcam: {reply[:200]}"},
        )
    studio.live.start(source, speed=float(body.get("speed") or 1.0))
    return {"ok": True, "source": describe(source)}


def _live_stop(studio: Studio, body: dict[str, Any]) -> dict[str, Any]:
    studio.live.stop()
    return {"ok": True}


def _live_driving(studio: Studio, body: dict[str, Any]) -> dict[str, Any]:
    studio.live.driving = bool(body.get("on", True))
    return {"ok": True, "driving": studio.live.driving}


def _live_recalibrate(studio: Studio, body: dict[str, Any]) -> dict[str, Any]:
    seconds = body.get("calibration_seconds")
    if seconds is not None:
        studio.live.calibration_seconds = max(60.0, float(seconds))
    studio.live.recalibrate()
    return {"ok": True, "calibration_seconds": studio.live.calibration_seconds}


def _live_preprocess(studio: Studio, body: dict[str, Any]) -> dict[str, Any]:
    region_text = str(body.get("region") or "").strip()
    fov = body.get("fisheye_fov_degrees")
    studio.live.preprocess = Preprocess(
        enhance_low_light=bool(body.get("enhance_low_light")),
        region=parse_region(region_text) if region_text else None,
        fisheye_fov_degrees=float(fov) if fov not in (None, "", 0) else None,
    )
    return {"ok": True, **studio.state()["preprocess"]}


def _conditions_capture(studio: Studio, body: dict[str, Any]) -> dict[str, Any]:
    studio.live.start_capture(bool(body.get("eyes_closed")), float(body.get("seconds") or 4.0))
    return {"ok": True}


def _conditions_clear(studio: Studio, body: dict[str, Any]) -> dict[str, Any]:
    studio.live.clear_captures()
    return {"ok": True}


def _conditions_run(studio: Studio, body: dict[str, Any]) -> dict[str, Any]:
    started = studio.run_conditions()
    return {
        "ok": started,
        "error": None if started else "a condition test is already running",
    }


def _train_start(studio: Studio, body: dict[str, Any]) -> dict[str, Any]:
    started = studio.training.start(
        environment=str(body.get("environment") or "dev"),
        track=str(body.get("track") or "research"),
        workspace_root=studio.workspace_root,
        seed=int(body.get("seed") or 7),
    )
    return {
        "ok": started,
        "error": None if started else "a training run is already going",
    }


def _model_load(studio: Studio, body: dict[str, Any]) -> dict[str, Any]:
    studio.load_model_file(Path(str(body["path"])).expanduser())
    return {"ok": True}


def _contact_add(studio: Studio, body: dict[str, Any]) -> dict[str, Any]:
    consent = ConsentState(str(body.get("consent") or ConsentState.MISSING.value))
    contact = studio.contacts.add(
        str(body["contact_id"]),
        str(body["display_name"]),
        str(body["phone_e164"]),
        consent=consent,
        consent_date=str(body.get("consent_date") or ""),
        sms_enabled=bool(body.get("sms_enabled", True)),
        calls_consented=bool(body.get("calls_consented", False)),
    )
    return {"ok": True, "contact": contact.to_json()}


def _contact_consent(studio: Studio, body: dict[str, Any]) -> dict[str, Any]:
    contact = studio.contacts.set_consent(
        str(body["contact_id"]),
        ConsentState(str(body["consent"])),
        str(body.get("consent_date") or ""),
        calls_consented=None if body.get("calls_consented") is None else bool(body["calls_consented"]),
    )
    return {"ok": True, "contact": contact.to_json()}


def _driver_add(studio: Studio, body: dict[str, Any]) -> dict[str, Any]:
    profile = studio.drivers.add(str(body["driver_id"]), str(body["display_name"]))
    studio.live.active_driver_id = profile.driver_id
    return {"ok": True, "drivers": studio.drivers.to_json()}


def _driver_activate(studio: Studio, body: dict[str, Any]) -> dict[str, Any]:
    profile = studio.drivers.activate(str(body["driver_id"]))
    studio.live.active_driver_id = profile.driver_id
    studio.live.recalibrate()
    return {"ok": True, "active": profile.to_json()}


def _microphone_start(studio: Studio, body: dict[str, Any]) -> dict[str, Any]:
    selected = body.get("device_index")
    device_index = None if selected in (None, "") else int(selected)
    studio.microphone.start(device_index=device_index)
    return {"ok": True, "microphone": studio.microphone_state()}


def _microphone_stop(studio: Studio, body: dict[str, Any]) -> dict[str, Any]:
    studio.microphone.stop()
    return {"ok": True, "microphone": studio.microphone_state()}


def _scenario_audio(studio: Studio, body: dict[str, Any]) -> dict[str, Any]:
    results = run_audio_scenarios()
    studio.scenario_results = [item.to_json() for item in results]
    studio.hub.publish("scenario_results", {"results": studio.scenario_results})
    return {"ok": True, "results": studio.scenario_results}


def _scenario_scripted(studio: Studio, body: dict[str, Any]) -> dict[str, Any]:
    """The scripted condition suite: each chosen condition through the robustness script, in memory."""
    keys = [str(key) for key in body.get("conditions") or ["clean"]]
    unknown = [key for key in keys if key != "clean" and key not in FEATURE_CONDITION_KEYS]
    if unknown:
        raise ValueError(f"unknown conditions: {', '.join(unknown)}")
    conditions = [None if key == "clean" else FEATURE_CONDITION_KEYS[key] for key in keys]
    events = [str(event) for event in body["events"]] if body.get("events") else None
    results = run_scripted_scenarios(conditions, escalation=studio.escalation, events=events)
    studio.scenario_results = [item.to_json() for item in results]
    studio.hub.publish("scenario_results", {"results": studio.scenario_results, "kind": "scripted"})
    return {"ok": True, "results": studio.scenario_results}


def _scenario_vision(studio: Studio, body: dict[str, Any]) -> dict[str, Any]:
    samples = list(studio.live.captured)
    if not any(not sample.eyes_closed for sample in samples):
        raise ValueError("capture eyes-open frames from the selected camera before running vision scenarios")
    landmarker = ImageLandmarker(studio.models_dir)
    try:
        results = run_vision_condition_scenarios(samples, landmarker)
    finally:
        landmarker.close()
    studio.scenario_results = [item.to_json() for item in results]
    studio.hub.publish("scenario_results", {"results": studio.scenario_results})
    return {"ok": True, "results": studio.scenario_results}


POST_ROUTES: dict[str, Callable[[Studio, dict[str, Any]], dict[str, Any]]] = {
    # Drive
    "/api/live/start": _live_start,
    "/api/live/stop": _live_stop,
    "/api/live/driving": _live_driving,
    "/api/live/recalibrate": _live_recalibrate,
    "/api/live/preprocess": _live_preprocess,
    "/api/contacts/add": _contact_add,
    "/api/contacts/consent": _contact_consent,
    "/api/drivers/add": _driver_add,
    "/api/drivers/activate": _driver_activate,
    "/api/microphone/start": _microphone_start,
    "/api/microphone/stop": _microphone_stop,
    # Test
    "/api/conditions/capture": _conditions_capture,
    "/api/conditions/clear": _conditions_clear,
    "/api/conditions/run": _conditions_run,
    "/api/scenarios/audio": _scenario_audio,
    "/api/scenarios/vision": _scenario_vision,
    "/api/scenarios/scripted": _scenario_scripted,
    # Train
    "/api/train/start": _train_start,
    "/api/model/load": _model_load,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m driven_ml.studio", description=__doc__)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="interface to bind (default: this computer only)",
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--workspace",
        default="driven_work/studio",
        help="workspace the Training tab writes to",
    )
    parser.add_argument(
        "--model",
        default="",
        help="model.json to load (default: the last trained candidate)",
    )
    parser.add_argument(
        "--source",
        default="",
        help="start live at once: 0 (webcam), rtsp://..., http://..., a file, synthetic",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="playback speed for files and synthetic data",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _serve(args)
    except DrivenError as error:
        print(f"driven studio: {error}", file=sys.stderr)
        return 2


def _serve(args: argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    workspace_root = Path(args.workspace).expanduser().resolve()
    studio = Studio(workspace_root, models_dir=workspace_root / "mediapipe")
    if args.model:
        studio.load_model_file(Path(args.model).expanduser())
    elif not studio.load_latest_candidate():
        logger.info("no trained model yet: open the Training tab and press Train")
    Handler.studio = studio
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    if args.source:
        studio.live.start(args.source, speed=args.speed)
    logger.info("Driven Studio on http://%s:%d", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        studio.live.stop()
        server.server_close()
    return 0
