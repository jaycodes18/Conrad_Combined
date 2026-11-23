"""The simplified Studio: exactly the Drive, Test and Train routes plus SSE, a streaming risk_signal, and 404s."""

from __future__ import annotations

import http.client
import json
import re
import threading
import time
from collections.abc import Iterator
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from driven_ml.studio.server import _GET_ROUTES, POST_ROUTES, STATIC, Handler, Studio

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "contracts" / "driver_risk_signal.v1.schema.json").read_text()
)
DRIVE_GET = {"/api/state", "/api/signal", "/api/microphone"}
TEST_GET = {"/api/scenarios"}
DRIVE_POST = {
    "/api/live/start",
    "/api/live/stop",
    "/api/live/driving",
    "/api/live/recalibrate",
    "/api/live/preprocess",
    "/api/contacts/add",
    "/api/contacts/consent",
    "/api/drivers/add",
    "/api/drivers/activate",
    "/api/microphone/start",
    "/api/microphone/stop",
}
TEST_POST = {
    "/api/conditions/capture",
    "/api/conditions/clear",
    "/api/conditions/run",
    "/api/scenarios/audio",
    "/api/scenarios/vision",
    "/api/scenarios/scripted",
}
TRAIN_POST = {"/api/train/start", "/api/model/load"}
REMOVED_GET = ["/api/catalog", "/api/hardware", "/api/status", "/api/contacts", "/api/metrics", "/api/drivers"]
REMOVED_POST = ["/api/contacts/ack", "/api/monitor", "/api/datasets/review", "/api/datasets/plan"]


@pytest.fixture
def server(tmp_path: Path) -> Iterator[tuple[str, int, Studio]]:
    studio = Studio(tmp_path, models_dir=tmp_path / "mediapipe")
    Handler.studio = studio
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield "127.0.0.1", httpd.server_address[1], studio
    finally:
        studio.live.stop()
        httpd.shutdown()
        httpd.server_close()


def request(host: str, port: int, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, Any]:
    connection = http.client.HTTPConnection(host, port, timeout=20)
    payload = None if body is None else json.dumps(body)
    connection.request(method, path, body=payload, headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    data = response.read()
    connection.close()
    kind = response.getheader("Content-Type") or ""
    return response.status, json.loads(data) if "json" in kind else data.decode()


def test_the_server_routes_are_exactly_the_three_tabs() -> None:
    assert set(_GET_ROUTES) == DRIVE_GET | TEST_GET
    assert set(POST_ROUTES) == DRIVE_POST | TEST_POST | TRAIN_POST


def test_the_page_has_three_tabs_and_stays_small() -> None:
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    assert len(page.splitlines()) < 500
    assert re.findall(r'data-tab="(\w+)"', page) == ["drive", "test", "train"]
    for removed in ("Datasets", "Hardware", "Coverage", 'data-tab="status"', 'data-tab="contacts"'):
        assert removed not in page
    for route in ("/api/catalog", "/api/hardware", "/api/status", '/api/contacts"', "/api/monitor"):
        assert route not in page


def test_removed_routes_return_404_and_kept_ones_answer(server: tuple[str, int, Studio]) -> None:
    host, port, _studio = server
    for path in REMOVED_GET:
        assert request(host, port, "GET", path)[0] == 404, path
    for path in REMOVED_POST:
        assert request(host, port, "POST", path, {})[0] == 404, path
    for path in sorted(DRIVE_GET | TEST_GET):
        status, body = request(host, port, "GET", path)
        assert status == 200, path
        assert isinstance(body, dict)
    status, page = request(host, port, "GET", "/")
    assert status == 200
    assert "Driven Studio" in page


def test_contact_consent_runs_through_the_drive_drawer_routes(server: tuple[str, int, Studio]) -> None:
    host, port, studio = server
    status, reply = request(
        host,
        port,
        "POST",
        "/api/contacts/add",
        {"contact_id": "p1", "display_name": "Parent", "phone_e164": "+12145550100", "consent": "recorded",
         "consent_date": "2026-11-01", "calls_consented": True},
    )  # fmt: skip
    assert status == 200
    assert reply["contact"]["calls_consented"] is True
    assert studio.live.risk.escalation.contacts.call_recipients()[0].contact_id == "p1"
    status, reply = request(
        host, port, "POST", "/api/contacts/add", {"contact_id": "p2", "display_name": "X", "phone_e164": "12"}
    )
    assert status == 400


def test_drive_streams_a_valid_risk_signal(server: tuple[str, int, Studio]) -> None:
    host, port, _studio = server
    connection = http.client.HTTPConnection(host, port, timeout=20)
    connection.request("GET", "/events")
    stream = connection.getresponse()
    assert stream.status == 200
    assert request(host, port, "POST", "/api/live/start", {"source": "synthetic", "speed": 50})[0] == 200
    deadline = time.monotonic() + 20.0
    signal = None
    while time.monotonic() < deadline and signal is None:
        line = stream.fp.readline().decode().strip()
        if line.startswith("data: "):
            message = json.loads(line[len("data: ") :])
            if message.get("type") == "risk_signal":
                signal = {key: value for key, value in message.items() if key != "type"}
    connection.close()
    assert signal is not None, "no risk_signal event within 20 s"
    assert list(Draft202012Validator(SCHEMA).iter_errors(signal)) == []
    assert signal["escalation"]["emergency_services"] == "never_automated"
    status, body = request(host, port, "GET", "/api/signal")
    assert status == 200
    assert body["signal"]["schema_version"] == "driver_risk_signal.v1"
