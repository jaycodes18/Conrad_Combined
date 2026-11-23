"""Live engine: camera frames -> face landmarks -> frame signals -> 60 s windows -> baseline -> model -> alerts.

It calls the same package functions as the pipeline (`frame_features`, `summarize_window`,
`predict_probability`), so what you see here is what the model computes, not a re-implementation.

Phases, as on the phone:
1. calibrating: the first `baseline_seconds` (2 minutes) build the driver's personal baseline and
   the driver's own straight-ahead head and gaze centre;
2. monitoring: every hop (15 s) a 60 s window is scored; an alert fires when the probability passes the
   model's threshold, the car is moving, and the cooldown has passed.

Eyes off the road for longer than `eyes_off_seconds` (2 s, NHTSA's guideline) is a fixed rule, not the
model. Every frame also runs the product's ``risk.RiskSession``: the six indicator submodels, fusion and the
escalation tiers, published as a ``risk_signal`` event carrying the exact ``DriverRiskSignal`` the app would
receive. "Repeated alerts go unanswered" is escalation Tier 2; the studio never sends a text or places a call.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..features.geometry import frame_features, no_face_frame
from ..features.preprocess import Preprocess
from ..indicators import CameraHealth, EyesOffTimer, FrameSignals
from ..modeling.artifact import (
    LogisticRegressionWeights,
    ModelArtifact,
    predict_probability,
)
from ..product import load_product
from ..risk import Condition, ContactBook, EscalationConfig, RiskSession
from ..spec import BASELINE_DELTA_PREFIX, FEATURE_SPEC, INDICATOR_SPEC, WINDOW_FEATURES, FeatureSpec
from ..windows import summarize_window
from .audio_health import AudioHealthReport
from .cameras import CameraKind, classify_source, expected_step_seconds, frame_gap_dropped, timestamp_advances
from .conditions import Sample, _eye_contrast, _mouth_contrast
from .events import EventHub
from .monitor import MonitorConfig, TemporalTracker
from .observations import ObservationBus, SensorObservation, SensorStatus, SourceKind, unavailable
from .sources import SYNTHETIC, SourceFrame, describe, iter_camera, iter_synthetic

logger = logging.getLogger(__name__)

CAPTURE_EVERY_FRAMES = 3
OBSERVATION_MAX_AGE_SECONDS = INDICATOR_SPEC.fusion.max_reading_age_seconds
DRIVER_REF = "drv_studio"
MAX_CAPTURED_FRAMES = 120
CAMERA_RECONNECT_INITIAL_SECONDS = 0.5
CAMERA_RECONNECT_MAX_SECONDS = 5.0
DRAW_EYES = (33, 133, 160, 158, 153, 144, 362, 263, 385, 387, 373, 380)
DRAW_MOUTH = (78, 308, 13, 14)
DRAW_IRIS = (468, 473)


class LiveEngine:
    """Runs one live session at a time on a background thread and publishes everything it computes."""

    def __init__(self, hub: EventHub, models_dir: Path, spec: FeatureSpec = FEATURE_SPEC) -> None:
        self.hub = hub
        self.models_dir = models_dir
        self.spec = spec
        self.model: ModelArtifact | None = None
        self.model_origin = ""
        self.driving = True
        self.calibration_seconds = float(spec.baseline_seconds)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._frame_lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._recalibrate = threading.Event()
        self.preprocess = Preprocess()
        self._capture_lock = threading.Lock()
        self._capture: tuple[bool, float] | None = None  # (eyes_closed, stop at monotonic time)
        self.captured: list[Sample] = []
        self.monitor_config = MonitorConfig.research_brief()
        self.active_driver_id = ""
        self.observations = ObservationBus()
        self.contacts = ContactBook()
        self.audio_allowed = False
        self._risk_lock = threading.Lock()
        self.risk = self._new_risk_session(EscalationConfig.from_product(load_product()))
        self.last_latency_ms: float | None = None
        self.last_source_fps_hz: float | None = None
        self.last_processing_fps_hz: float | None = None
        self.dropped_frames = 0
        self.camera_faults = 0
        self.on_baseline: Callable[[dict[str, float] | None, dict[str, float]], None] | None = None
        self._initialize_metrics()

    # ---- control -------------------------------------------------------------------------------------
    def _new_risk_session(self, escalation: EscalationConfig) -> RiskSession:
        return RiskSession(
            session_id=f"studio-{int(time.time())}",
            driver_ref=DRIVER_REF,
            escalation=escalation,
            contacts=self.contacts,
            audio_allowed=self.audio_allowed,
            model=self.model,
        )

    def configure_risk(self, escalation: EscalationConfig, contacts: ContactBook, *, audio_allowed: bool) -> None:
        """Use the product's escalation rules, the studio's contact book and the research-audio gate."""
        with self._risk_lock:
            self.contacts = contacts
            self.audio_allowed = audio_allowed
            self.risk = self._new_risk_session(escalation)

    def set_model(self, model: ModelArtifact, origin: str) -> None:
        self.model = model
        self.model_origin = origin
        with self._risk_lock:
            self.risk.set_model(model)
        self.hub.publish("model", self.model_summary())

    def model_summary(self) -> dict[str, Any]:
        model = self.model
        if model is None:
            return {}
        return {
            "origin": self.model_origin,
            "sha": model.short_sha,
            "kind": model.weights.type,
            "inputs": len(model.inputs),
            "threshold": model.threshold,
            "cooldown_seconds": model.cooldown_seconds,
            "data_source": model.data_source,
            "track": model.track,
            "feature_spec_version": model.feature_spec_version,
            "weights": _input_weights(model),
        }

    def start(self, source: str, speed: float = 1.0) -> None:
        self.stop()
        self._stop.clear()
        self.last_latency_ms = None
        self.last_source_fps_hz = None
        self.last_processing_fps_hz = None
        self._thread = threading.Thread(target=self._run, args=(source, speed), name="driven-live", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=5)
            if thread.is_alive():
                raise TimeoutError("camera stream did not stop within 5 seconds")
        self._thread = None

    def recalibrate(self) -> None:
        self._recalibrate.set()

    def start_capture(self, eyes_closed: bool, seconds: float) -> None:
        """Keep raw frames for the condition test: eyes open (looking ahead) or eyes closed."""
        with self._capture_lock:
            self._capture = (eyes_closed, time.monotonic() + seconds)

    def clear_captures(self) -> None:
        with self._capture_lock:
            self.captured = []
            self._capture = None

    def _maybe_capture(self, frame: SourceFrame, index: int) -> None:
        if frame.rgb is None or index % CAPTURE_EVERY_FRAMES:
            return
        with self._capture_lock:
            if self._capture is None:
                return
            eyes_closed, until = self._capture
            if time.monotonic() > until:
                self._capture = None
                counts = {
                    "open": sum(not s.eyes_closed for s in self.captured),
                    "closed": sum(s.eyes_closed for s in self.captured),
                }
                self.hub.publish("capture_done", counts)
                return
            if len(self.captured) < MAX_CAPTURED_FRAMES:
                self.captured.append(Sample(frame.rgb.copy(), eyes_closed))

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def latest_jpeg(self) -> bytes | None:
        with self._frame_lock:
            return self._jpeg

    def metrics_snapshot(self) -> dict[str, Any]:
        return self.observations.snapshot()

    def publish_audio_metrics(self, health: Any, estimate: Any) -> None:
        """Publish microphone health and breathing estimates without storing raw audio."""
        self.hub.publish("audio_health", health.to_json())
        self.hub.publish("breathing", estimate.to_json())
        for observation in (
            health.observation(),
            estimate.rate_observation(),
            estimate.regularity_observation(),
            estimate.confidence_observation(),
        ):
            self.observations.publish(observation)
        self.observations.expire(health.t_seconds, OBSERVATION_MAX_AGE_SECONDS)
        self._publish_metric_state()

    def push_audio(self, t_seconds: float, samples: np.ndarray, health: AudioHealthReport) -> None:
        """One analyzed microphone chunk into the breathing indicator. It stays in memory and is dropped after."""
        with self._risk_lock:
            self.risk.push_audio(t_seconds, samples, health, driving=self.driving)

    def _initialize_metrics(self) -> None:
        t_seconds = time.monotonic()
        for name, detail in (
            ("eye_movement", "waiting for a camera frame"),
            ("gaze_direction", "waiting for a camera frame"),
            ("gaze_duration", "waiting for a camera frame"),
            ("eye_closure", "waiting for a camera frame"),
            ("head_position", "waiting for a camera frame"),
            ("face_visibility", "waiting for a camera frame"),
            ("camera_quality", "waiting for a camera frame"),
            ("eye_tracking", "waiting for a camera frame"),
            ("microphone_quality", "research microphone is disabled or not yet started"),
            ("breathing_rate", "breathing signal unavailable"),
            ("breathing_regularity", "breathing signal unavailable"),
            ("respiratory_confidence", "breathing signal unavailable"),
            ("driver_responsiveness", "no prompt-response sensor is configured"),
            ("driving_stability", "requires outward-camera or vehicle telemetry input"),
            ("drowsiness_probability", "no current model score"),
        ):
            self.observations.publish(unavailable(name, t_seconds, detail=detail))

    def _mark_camera_unavailable(self, source_kind: SourceKind, detail: str) -> None:
        t_seconds = time.monotonic()
        for name in (
            "face_visibility",
            "eye_movement",
            "gaze_direction",
            "gaze_duration",
            "eye_closure",
            "head_position",
            "camera_quality",
            "eye_tracking",
            "drowsiness_probability",
        ):
            self.observations.publish(
                unavailable(
                    name,
                    t_seconds,
                    detail=detail,
                    source_kind=source_kind,
                )
            )
        self._publish_metric_state()
        self.publish_risk(FrameSignals.from_features(t_seconds, no_face_frame()), CameraHealth(connected=False))

    def _publish_frame_metrics(
        self,
        features: dict[str, float],
        monitor: dict[str, Any],
        rgb: np.ndarray | None,
        source_kind: SourceKind,
    ) -> None:
        t_seconds = time.monotonic()
        face_visible = float(features.get("face", 0.0)) > 0.0
        face_observation = SensorObservation(
            name="face_visibility",
            t_seconds=t_seconds,
            value=1.0 if face_visible else 0.0,
            confidence=1.0,
            status=SensorStatus.AVAILABLE if face_visible else SensorStatus.UNAVAILABLE,
            valid=face_visible,
            source_kind=source_kind,
            detail=(
                "face landmarks returned; MediaPipe does not expose a calibrated detection confidence"
                if face_visible
                else "face not visible; behavior is not assessed"
            ),
        )
        self.observations.publish(face_observation)
        eye_visible = (
            face_visible
            and math.isfinite(float(features.get("ear", math.nan)))
            and monitor.get("monitoring") == "available"
        )
        if eye_visible:
            gaze = float(features.get("gaze_horizontal", math.nan))
            yaw = float(features.get("yaw", math.nan))
            looking_away = bool(monitor.get("looking_away", False))
            self.observations.publish(
                SensorObservation(
                    name="eye_movement",
                    t_seconds=t_seconds,
                    value=gaze if math.isfinite(gaze) else None,
                    confidence=1.0 if math.isfinite(gaze) else 0.0,
                    status=SensorStatus.AVAILABLE if math.isfinite(gaze) else SensorStatus.UNAVAILABLE,
                    valid=math.isfinite(gaze),
                    source_kind=source_kind,
                    detail="landmark-derived horizontal eye movement; not a calibrated probability",
                )
            )
            self.observations.publish(
                SensorObservation(
                    name="gaze_direction",
                    t_seconds=t_seconds,
                    value="off_forward" if looking_away else "forward",
                    confidence=1.0,
                    status=SensorStatus.AVAILABLE,
                    valid=True,
                    source_kind=source_kind,
                    detail="direction is relative to the driver's calibrated centre",
                )
            )
            self.observations.publish(
                SensorObservation(
                    name="gaze_duration",
                    t_seconds=t_seconds,
                    value=float(monitor.get("gaze_for_seconds", 0.0)),
                    confidence=1.0,
                    status=SensorStatus.AVAILABLE,
                    valid=True,
                    source_kind=source_kind,
                    unit="seconds",
                    detail="continuous off-forward gaze duration",
                )
            )
            self.observations.publish(
                SensorObservation(
                    name="eye_closure",
                    t_seconds=t_seconds,
                    value=float(monitor.get("closure_for_seconds", 0.0)),
                    confidence=1.0,
                    status=SensorStatus.AVAILABLE,
                    valid=True,
                    source_kind=source_kind,
                    unit="seconds",
                    detail="continuous eyelid closure duration from EAR landmarks",
                )
            )
            self.observations.publish(
                SensorObservation(
                    name="head_position",
                    t_seconds=t_seconds,
                    value=yaw if math.isfinite(yaw) else None,
                    confidence=1.0 if math.isfinite(yaw) else 0.0,
                    status=SensorStatus.AVAILABLE if math.isfinite(yaw) else SensorStatus.UNAVAILABLE,
                    valid=math.isfinite(yaw),
                    source_kind=source_kind,
                    unit="degrees_yaw",
                    detail="head yaw from face landmarks; confidence is not calibrated",
                )
            )
            self.observations.publish(
                SensorObservation(
                    name="eye_tracking",
                    t_seconds=t_seconds,
                    value="available",
                    confidence=1.0,
                    status=SensorStatus.AVAILABLE,
                    valid=True,
                    source_kind=source_kind,
                    detail="eye aspect ratio is measurable; occlusion confidence is not exposed",
                )
            )
        else:
            for name in (
                "eye_movement",
                "gaze_direction",
                "gaze_duration",
                "eye_closure",
                "head_position",
                "eye_tracking",
            ):
                self.observations.publish(
                    unavailable(
                        name,
                        t_seconds,
                        detail="eye or face landmarks unavailable; no behavior value inferred",
                        source_kind=source_kind,
                    )
                )

        self.observations.publish(_camera_quality_observation(rgb, t_seconds, source_kind))
        self.observations.publish(
            unavailable(
                "driver_responsiveness",
                t_seconds,
                detail="no prompt-response sensor is configured",
                source_kind=source_kind,
            )
        )
        self.observations.publish(
            unavailable(
                "driving_stability",
                t_seconds,
                detail="requires outward-camera or vehicle telemetry input",
                source_kind=source_kind,
            )
        )
        self.observations.expire(t_seconds, OBSERVATION_MAX_AGE_SECONDS)
        self._publish_metric_state()

    def publish_drowsiness_probability(
        self,
        probability: float,
        t_seconds: float,
        source_kind: SourceKind = SourceKind.HARDWARE,
    ) -> None:
        self.observations.publish(
            SensorObservation(
                name="drowsiness_probability",
                t_seconds=time.monotonic(),
                value=probability,
                confidence=1.0,
                status=SensorStatus.AVAILABLE,
                valid=True,
                source_kind=source_kind,
                detail=f"existing drowsiness model window ending at drive time {t_seconds:.2f} s",
            )
        )

    def _publish_metric_state(self) -> None:
        self.hub.publish("metrics", self.observations.snapshot())

    def publish_risk(
        self,
        frame: FrameSignals,
        camera: CameraHealth,
        *,
        calibrating: bool = False,
        eyes_off_seconds: float = 0.0,
    ) -> dict[str, Any]:
        """Run the product risk session on one frame and stream the ``DriverRiskSignal`` it produces."""
        with self._risk_lock:
            signal = self.risk.push_frame(
                frame,
                camera,
                calibrating=calibrating,
                driving=self.driving,
                emitted_at=datetime.now(UTC),
                eyes_off_seconds=eyes_off_seconds,
            )
            decision = self.risk.last_decision
        document = signal.to_json()
        self.hub.publish("risk_signal", document)
        for event in () if decision is None else decision.events:
            self.hub.publish("escalation", event.to_json())
        return document

    # ---- the loop ------------------------------------------------------------------------------------
    def _run(self, source: str, speed: float) -> None:
        self.hub.publish(
            "live_status",
            {
                "state": "starting",
                "source": describe(source),
                "camera_kind": classify_source(source).value,
            },
        )
        try:
            if source == SYNTHETIC:
                self._loop(
                    iter_synthetic(speed=max(speed, 1.0), spec=self.spec),
                    detector=None,
                    source_kind=SourceKind.SIMULATED,
                )
            else:
                camera_kind = classify_source(source)
                source_kind = SourceKind.PRERECORDED if camera_kind is CameraKind.FILE else SourceKind.HARDWARE
                with _open_detector(self.models_dir, self.spec) as detect:
                    retry = 0
                    while not self._stop.is_set():
                        try:
                            self._loop(iter_camera(source, self.spec, speed), detector=detect, source_kind=source_kind)
                            break
                        except OSError as error:
                            if camera_kind in {CameraKind.FILE, CameraKind.SYNTHETIC} or self._stop.is_set():
                                raise
                            self.camera_faults += 1
                            detail = f"camera disconnected or unavailable: {error}"
                            self._mark_camera_unavailable(source_kind, detail)
                            delay = min(
                                CAMERA_RECONNECT_INITIAL_SECONDS * (2 ** min(retry, 4)),
                                CAMERA_RECONNECT_MAX_SECONDS,
                            )
                            retry += 1
                            self.hub.publish(
                                "camera_fault",
                                {"kind": "disconnected", "message": detail, "retry_in_seconds": delay},
                            )
                            self.hub.publish(
                                "live_status",
                                {
                                    "state": "recovering",
                                    "source": describe(source),
                                    "message": f"{detail}; retrying in {delay:.1f} seconds",
                                },
                            )
                            if self._stop.wait(delay):
                                break
            self.hub.publish("live_status", {"state": "finished", "source": describe(source)})
        except OSError as error:
            logger.exception("camera failed")
            self.camera_faults += 1
            camera_kind = classify_source(source)
            source_kind = SourceKind.PRERECORDED if camera_kind is CameraKind.FILE else SourceKind.HARDWARE
            self._mark_camera_unavailable(source_kind, f"camera unavailable: {error}")
            self.hub.publish(
                "live_status",
                {
                    "state": "error",
                    "kind": "camera_failure",
                    "message": f"camera_failure: {error}",
                },
            )
        except Exception as error:  # any failure must reach the dashboard, not kill the server
            logger.exception("live session failed")
            self.hub.publish(
                "live_status",
                {"state": "error", "kind": "engine_error", "message": f"{type(error).__name__}: {error}"},
            )

    def _loop(
        self,
        frames: Iterator[SourceFrame],
        detector: Callable[[SourceFrame], Any] | None,
        source_kind: SourceKind = SourceKind.HARDWARE,
    ) -> None:
        session = _Session(self)
        previous_t: float | None = None
        previous_processing_started: float | None = None
        step = expected_step_seconds(self.spec)
        for index, raw in enumerate(frames):
            if self._stop.is_set():
                return
            if not timestamp_advances(previous_t, raw.t):
                self.camera_faults += 1
                self.hub.publish(
                    "camera_fault",
                    {
                        "kind": "non_monotonic",
                        "t": raw.t,
                        "message": "timestamp did not increase; frame dropped, not scored as behavior",
                    },
                )
                continue
            source_fps_hz = None if previous_t is None else 1.0 / (raw.t - previous_t)
            if previous_t is not None and frame_gap_dropped(previous_t, raw.t, step):
                self.dropped_frames += 1
                self.hub.publish("camera_fault", {"kind": "dropped_frame", "t": raw.t})
                gap_seconds = raw.t - previous_t
                if gap_seconds > self.monitor_config.invalid_after_seconds:
                    session.camera_gap(previous_t, raw.t, source_kind)
            previous_t = raw.t
            processing_started = time.perf_counter()
            if previous_processing_started is not None:
                elapsed = processing_started - previous_processing_started
                self.last_processing_fps_hz = 1.0 / elapsed if elapsed > 0 else None
            previous_processing_started = processing_started
            self.last_source_fps_hz = source_fps_hz
            self._maybe_capture(raw, index)
            frame = raw
            if raw.rgb is not None and self.preprocess.active:
                frame = SourceFrame(t=raw.t, rgb=self.preprocess.apply(raw.rgb), features=None)
            if self._recalibrate.is_set():
                self._recalibrate.clear()
                session.reset(frame.t)
                self.hub.publish("live_status", {"state": "recalibrating"})
            if frame.features is not None:
                features, landmarks = dict(frame.features), None
            else:
                result = detector(frame) if detector else None
                features, landmarks = result if result else (no_face_frame(), None)
            camera = None
            if frame.rgb is not None:
                fisheye = self.preprocess.fisheye_fov_degrees is not None
                camera = CameraHealth.from_rgb(frame.rgb, landmarks, fisheye=fisheye)
            alerts = session.push(frame.t, features, rgb=frame.rgb, source_kind=source_kind, camera=camera)
            self.last_latency_ms = (time.perf_counter() - processing_started) * 1000.0
            self._render(
                frame,
                features,
                landmarks=landmarks,
                calibrating=session.baseline is None,
                alerting=bool(alerts),
                t=features["t"],
            )
            self.hub.publish(
                "camera_performance",
                {
                    "inference_latency_ms": self.last_latency_ms,
                    "source_fps_hz": self.last_source_fps_hz,
                    "processing_fps_hz": self.last_processing_fps_hz,
                    "dropped_frames": self.dropped_frames,
                    "camera_faults": self.camera_faults,
                },
            )

    def _render(
        self,
        frame: SourceFrame,
        features: dict[str, float],
        *,
        landmarks: Any,
        calibrating: bool,
        alerting: bool,
        t: float,
    ) -> None:
        import cv2

        if frame.rgb is not None:
            image = cv2.cvtColor(frame.rgb, cv2.COLOR_RGB2BGR)
        else:
            image = np.zeros((360, 640, 3), dtype=np.uint8)
            image[:] = (32, 26, 22)
            cv2.putText(
                image,
                "Synthetic driver: no camera, signals only",
                (24, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (230, 230, 230),
                2,
            )
            ear = features.get("ear", math.nan)
            if math.isfinite(ear):
                opening = int(max(0.0, min(1.0, ear / 0.35)) * 120)
                cv2.ellipse(
                    image,
                    (240, 190),
                    (70, max(2, opening // 2)),
                    0,
                    0,
                    360,
                    (240, 240, 240),
                    -1,
                )
                cv2.ellipse(
                    image,
                    (400, 190),
                    (70, max(2, opening // 2)),
                    0,
                    0,
                    360,
                    (240, 240, 240),
                    -1,
                )
        if landmarks is not None:
            for index in (*DRAW_EYES, *DRAW_MOUTH):
                x, y = landmarks[index]
                cv2.circle(image, (int(x), int(y)), 2, (80, 220, 120), -1)
            for index in DRAW_IRIS:
                if index < len(landmarks):
                    x, y = landmarks[index]
                    cv2.circle(image, (int(x), int(y)), 3, (255, 180, 60), -1)
        status = "CALIBRATING" if calibrating else ("ALERT" if alerting else "MONITORING")
        colour = (0, 200, 255) if calibrating else ((40, 40, 230) if alerting else (90, 200, 90))
        cv2.rectangle(
            image,
            (0, 0),
            (image.shape[1] - 1, image.shape[0] - 1),
            colour,
            6 if alerting else 2,
        )
        cv2.putText(
            image,
            f"{status}  t={t:6.1f}s  EAR={features.get('ear', math.nan):.3f}",
            (16, image.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colour,
            2,
        )
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if ok:
            with self._frame_lock:
                self._jpeg = encoded.tobytes()


class _Session:
    """Per-drive state: frame buffer, calibration, baseline, cooldowns and the escalation history."""

    def __init__(self, engine: LiveEngine) -> None:
        self.engine = engine
        self.spec = engine.spec
        self.buffer: deque[dict[str, float]] = deque(maxlen=self.spec.frames_per_window)
        self.reset(0.0)

    def reset(self, origin: float) -> None:
        self.origin = origin
        self.buffer.clear()
        self.calibration_windows: list[dict[str, float]] = []
        self.calibration_frames: list[dict[str, float]] = []
        self.baseline: dict[str, float] | None = None
        self.centre = {"yaw": 0.0, "gaze_horizontal": 0.0}
        self.since_hop = 0
        self.last_drowsy_start = -math.inf
        self.eyes_off = EyesOffTimer()
        self.last_eyes_off = -math.inf
        self.visibility_reference: tuple[float, float] | None = None
        self._contrasts: list[tuple[float, float]] = []
        self.engine.risk.reset()
        self.tracker = TemporalTracker(self.engine.monitor_config)
        self.open_ear: float | None = None
        self.last_monitor: dict[str, Any] = {}
        self._skip_tracker_t: float | None = None

    def camera_gap(
        self,
        previous_source_t: float,
        source_t: float,
        source_kind: SourceKind = SourceKind.HARDWARE,
    ) -> None:
        """Break continuous events and discard any model window spanning a long camera gap."""
        self.eyes_off.reset()
        self.buffer.clear()
        self.since_hop = 0
        t_seconds = time.monotonic()
        for name in (
            "eye_movement",
            "gaze_direction",
            "gaze_duration",
            "eye_closure",
            "head_position",
            "eye_tracking",
            "drowsiness_probability",
        ):
            self.engine.observations.publish(
                unavailable(
                    name,
                    t_seconds,
                    detail="camera gap; behavior and model score are unavailable until tracking recovers",
                    source_kind=source_kind,
                )
            )
        self.engine._publish_metric_state()
        self.engine.risk.clear_probability()
        gap_frame = FrameSignals.from_features(source_t - self.origin, no_face_frame())
        self.engine.publish_risk(gap_frame, CameraHealth(frame_fault=True), calibrating=self.baseline is None)
        start_t = previous_source_t - self.origin + 1e-6
        end_t = source_t - self.origin
        invalid_values = {
            "face": 0.0,
            "ear": math.nan,
            "yaw": math.nan,
            "gaze_horizontal": math.nan,
        }
        self.tracker.push(
            start_t,
            **invalid_values,
            centre_yaw=self.centre["yaw"],
            centre_gaze=self.centre["gaze_horizontal"],
            open_ear=self.open_ear,
            camera_connected=False,
            eye_visible=False,
        )
        snapshot = self.tracker.push(
            end_t,
            **invalid_values,
            centre_yaw=self.centre["yaw"],
            centre_gaze=self.centre["gaze_horizontal"],
            open_ear=self.open_ear,
            camera_connected=False,
            eye_visible=False,
        )
        self.last_monitor = snapshot.to_json()
        self._skip_tracker_t = end_t

    def push(
        self,
        source_t: float,
        features: dict[str, float],
        *,
        rgb: np.ndarray | None = None,
        source_kind: SourceKind = SourceKind.HARDWARE,
        camera: CameraHealth | None = None,
    ) -> list[dict[str, Any]]:
        t = source_t - self.origin
        features["t"] = t
        self.buffer.append(features)
        self.since_hop += 1
        if self.baseline is None and features.get("face", 0.0) > 0:
            self.calibration_frames.append(features)
        alerts = self._eyes_off(t, features)
        alerts += self._monitor(t, features)
        if len(self.buffer) == self.spec.frames_per_window and self.since_hop >= self.spec.frames_per_hop:
            self.since_hop = 0
            alerts += self._score_window(t, source_t, source_kind)
        self.engine._publish_frame_metrics(features, self.last_monitor, rgb, source_kind)
        eye_visibility, mouth_visibility = self._visibility(features)
        signal = self.engine.publish_risk(
            FrameSignals.from_features(t, features, eye_visibility=eye_visibility, mouth_visibility=mouth_visibility),
            camera or CameraHealth(),
            calibrating=self.baseline is None,
            eyes_off_seconds=self.eyes_off_for(t),
        )
        hub = self.engine.hub
        for alert in alerts:
            hub.publish("alert", alert)
        hub.publish(
            "frame",
            {
                "t": t,
                "phase": "calibrating" if self.baseline is None else "monitoring",
                "calibration_left": max(0.0, self.engine.calibration_seconds - t) if self.baseline is None else 0.0,
                "driving": self.engine.driving,
                "looking_away": self.eyes_off.since is not None,
                "eyes_off_for": self.eyes_off_for(t),
                "condition": signal["condition"],
                "level": signal["level"],
                "monitoring": self.last_monitor.get("monitoring", "available"),
                "closure_for_seconds": self.last_monitor.get("closure_for_seconds", 0.0),
                "warning_delay_seconds": self.last_monitor.get("warning_delay_seconds", 0.0),
                "source_fps_hz": self.engine.last_source_fps_hz,
                "processing_fps_hz": self.engine.last_processing_fps_hz,
                "camera_faults": self.engine.camera_faults,
                "dropped_frames": self.engine.dropped_frames,
                **{key: features.get(key) for key in ("ear", "mar", "yaw", "pitch", "gaze_horizontal", "face")},
            },
        )
        return alerts

    def _monitor(self, t: float, features: dict[str, float]) -> list[dict[str, Any]]:
        """Research timers. They do not replace the 2-second eyes-off rule or the 60-second model."""
        if self._skip_tracker_t is not None and abs(t - self._skip_tracker_t) <= 1e-6:
            self._skip_tracker_t = None
            return []
        self.tracker.config = self.engine.monitor_config
        snap = self.tracker.push(
            t,
            face=float(features.get("face", 0.0)),
            ear=float(features.get("ear", math.nan)),
            yaw=float(features.get("yaw", math.nan)),
            gaze_horizontal=float(features.get("gaze_horizontal", math.nan)),
            centre_yaw=self.centre["yaw"],
            centre_gaze=self.centre["gaze_horizontal"],
            open_ear=self.open_ear,
            camera_connected=True,
            eye_visible=True,
        )
        self.last_monitor = snap.to_json()
        if self.baseline is None or not self.engine.driving:
            return []
        return [event.to_json() for event in snap.events if event.kind in {"research_gaze", "research_closure"}]

    def eyes_off_for(self, t: float) -> float:
        return 0.0 if self.eyes_off.since is None else t - self.eyes_off.since

    def _visibility(self, features: dict[str, float]) -> tuple[float, float]:
        """Eye and mouth visibility, 0..1. Synthetic frames carry them; camera frames carry region contrasts,
        which are scored against the calibration's median contrast for this driver and camera."""
        if "eye_visibility" in features:
            return float(features["eye_visibility"]), float(features.get("mouth_visibility", 1.0))
        eye, mouth = features.get("eye_contrast"), features.get("mouth_contrast")
        if eye is None or mouth is None or not (math.isfinite(eye) and math.isfinite(mouth)):
            return 1.0, 1.0
        if self.baseline is None:
            self._contrasts.append((eye, mouth))
            return 1.0, 1.0
        if self.visibility_reference is None and self._contrasts:
            self.visibility_reference = (
                float(np.median([item[0] for item in self._contrasts])),
                float(np.median([item[1] for item in self._contrasts])),
            )
        if self.visibility_reference is None:
            return 1.0, 1.0
        eye_ref, mouth_ref = self.visibility_reference
        return min(1.0, eye / max(eye_ref, 1e-6)), min(1.0, mouth / max(mouth_ref, 1e-6))

    def _eyes_off(self, t: float, features: dict[str, float]) -> list[dict[str, Any]]:
        """NHTSA's 2-second rule, from the driver's calibrated centre. Not the model."""
        rules = self.spec.live_alerts
        away = (
            self.engine.driving
            and self.baseline is not None
            and self.last_monitor.get("monitoring") == "available"
            and _looking_away(features, self.centre, self.spec)
        )
        away_for = self.eyes_off.push(t, away, tracked=features.get("face", 0.0) > 0)
        if not away_for:
            return []
        if away_for >= rules.eyes_off_seconds and t - self.last_eyes_off >= rules.eyes_off_cooldown_seconds:
            self.last_eyes_off = t
            return [{"kind": "eyes_off", "t": t, "score": away_for}]
        return []

    def _score_window(self, t: float, source_t: float, source_kind: SourceKind) -> list[dict[str, Any]]:
        spec, engine = self.spec, self.engine
        window = pd.DataFrame(list(self.buffer))
        summary = summarize_window(window, spec)
        window_start = float(window["t"].iloc[0])
        face_fraction = float((window["face"] > 0).mean())
        event: dict[str, Any] = {
            "t0": window_start,
            "t1": t,
            "features": summary,
            "face_fraction": face_fraction,
        }
        alerts: list[dict[str, Any]] = []
        camera_quality = engine.observations.latest("camera_quality")
        if self.baseline is None:
            self._calibrate(summary, window_start, t, source_t)
        elif engine.model is None:
            event["note"] = "no model loaded: train one on the Training tab or pass --model"
        elif camera_quality is not None and camera_quality.status is SensorStatus.DEGRADED:
            event["note"] = "camera image quality is degraded; this model window is not scored"
        elif face_fraction >= spec.thresholds.min_face_fraction:
            model = engine.model
            inputs = _model_inputs(summary, self.baseline, model.inputs)
            probability = float(predict_probability(model, np.array([inputs]))[0])
            event.update(
                probability=probability,
                threshold=model.threshold,
                deltas={n: v for n, v in zip(model.inputs, inputs, strict=True) if n.startswith(BASELINE_DELTA_PREFIX)},
            )
            engine.publish_drowsiness_probability(probability, t, source_kind=source_kind)
            with engine._risk_lock:
                engine.risk.set_probability(probability, t, model.threshold)
            cooled = window_start - self.last_drowsy_start >= model.cooldown_seconds
            if engine.driving and probability >= model.threshold and cooled:
                self.last_drowsy_start = window_start
                alerts.append(
                    {
                        "kind": "drowsy",
                        "t": window_start + spec.window_seconds,
                        "score": probability,
                    }
                )
                with engine._risk_lock:
                    engine.risk.escalation.note_alert(t, Condition.DROWSY)
        else:
            event["note"] = "face visible less than the minimum share of this window: not scored"
        engine.hub.publish("window", event)
        return alerts

    def _calibrate(self, summary: dict[str, float], window_start: float, t: float, source_t: float) -> None:
        engine = self.engine
        if window_start + self.spec.window_seconds <= engine.calibration_seconds + 1e-9:
            self.calibration_windows.append(summary)
        if t + 1.0 / self.spec.target_fps < engine.calibration_seconds - 1e-9:  # the window ending at 2:00 counts
            return
        self.baseline, self.centre = _finish_calibration(self.calibration_windows, self.calibration_frames)
        self.open_ear = _median_ear(self.calibration_frames)
        if engine.on_baseline is not None and engine.active_driver_id:
            engine.on_baseline(self.baseline, self.centre)
        engine.hub.publish(
            "baseline",
            {
                "ok": self.baseline is not None,
                "baseline": self.baseline or {},
                "centre": self.centre,
                "windows": len(self.calibration_windows),
            },
        )
        if self.baseline is None:
            self.reset(source_t)


class _DetectorContext:
    """Opens MediaPipe Face Landmarker (VIDEO mode) and turns a frame into (features, pixel landmarks)."""

    def __init__(self, models_dir: Path, spec: FeatureSpec) -> None:
        self.models_dir = models_dir
        self.spec = spec
        self._detector: Any = None
        self._last_ms = -1

    def __enter__(self) -> Callable[[SourceFrame], tuple[dict[str, float], Any] | None]:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        from ..features.extraction import download_face_model

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(download_face_model(self.models_dir))),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._detector = vision.FaceLandmarker.create_from_options(options)

        def detect(frame: SourceFrame) -> tuple[dict[str, float], Any] | None:
            if frame.rgb is None:
                return None
            timestamp_ms = max(int(frame.t * 1000), self._last_ms + 1)  # VIDEO mode needs increasing timestamps
            self._last_ms = timestamp_ms
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frame.rgb))
            result = self._detector.detect_for_video(image, timestamp_ms)
            if not result.face_landmarks:
                return None
            height, width = frame.rgb.shape[:2]
            points = np.array([[p.x * width, p.y * height] for p in result.face_landmarks[0]])
            blendshapes = (
                {c.category_name: c.score for c in result.face_blendshapes[0]} if result.face_blendshapes else None
            )
            matrix = result.facial_transformation_matrixes[0] if result.facial_transformation_matrixes else None
            features = frame_features(points, blendshapes, matrix, self.spec)
            features["eye_contrast"] = _eye_contrast(frame.rgb, points)
            features["mouth_contrast"] = _mouth_contrast(frame.rgb, points)
            return features, points

        return detect

    def __exit__(self, *exc: object) -> None:
        if self._detector is not None:
            self._detector.close()


def _open_detector(models_dir: Path, spec: FeatureSpec) -> _DetectorContext:
    return _DetectorContext(models_dir, spec)


def _camera_quality_observation(
    rgb: np.ndarray | None,
    t_seconds: float,
    source_kind: SourceKind,
) -> SensorObservation:
    if rgb is None:
        return unavailable(
            "camera_quality",
            t_seconds,
            detail="no image input in this session",
            source_kind=source_kind,
        )
    import cv2

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mean_luma = float(np.mean(gray)) / 255.0
    contrast = float(np.std(gray)) / 255.0
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    issues: list[str] = []
    if mean_luma < 0.12:
        issues.append("low_light")
    elif mean_luma > 0.92:
        issues.append("overexposed")
    if contrast < 0.06:
        issues.append("low_contrast")
    if sharpness < 12.0:
        issues.append("possible_motion_blur_or_soft_focus")
    state = "degraded" if issues else "usable"
    return SensorObservation(
        name="camera_quality",
        t_seconds=t_seconds,
        value=state,
        confidence=0.5 if issues else 0.8,
        status=SensorStatus.DEGRADED if issues else SensorStatus.AVAILABLE,
        valid=True,
        source_kind=source_kind,
        detail=(
            f"heuristic image quality; mean_luma={mean_luma:.3f}, contrast={contrast:.3f}, "
            f"laplacian_variance={sharpness:.1f}; "
            + (", ".join(issues) if issues else "no configured quality issue detected")
        ),
    )


def _median_ear(frames: list[dict[str, float]]) -> float | None:
    if not frames:
        return None
    values = [float(item["ear"]) for item in frames if "ear" in item and math.isfinite(float(item["ear"]))]
    if not values:
        return None
    return float(np.nanmedian(np.asarray(values, dtype=float)))


def _looking_away(features: dict[str, float], centre: dict[str, float], spec: FeatureSpec) -> bool:
    if features.get("face", 0.0) <= 0:
        return False
    gaze = features.get("gaze_horizontal", math.nan) - centre["gaze_horizontal"]
    yaw = features.get("yaw", math.nan) - centre["yaw"]
    thresholds = spec.thresholds
    return bool(
        (math.isfinite(gaze) and abs(gaze) > thresholds.gaze_off_ratio)
        or (math.isfinite(yaw) and abs(yaw) > thresholds.yaw_off_degrees)
    )


def _finish_calibration(
    windows: list[dict[str, float]], frames: list[dict[str, float]]
) -> tuple[dict[str, float] | None, dict[str, float]]:
    centre = {"yaw": 0.0, "gaze_horizontal": 0.0}
    if frames:
        table = pd.DataFrame(frames)
        for key in centre:
            if key in table:
                median = float(np.nanmedian(table[key].to_numpy(dtype=float)))
                centre[key] = median if math.isfinite(median) else 0.0
    if not windows:
        return None, centre
    table = pd.DataFrame(windows)
    baseline = {name: float(np.nanmean(table[name].to_numpy(dtype=float))) for name in WINDOW_FEATURES if name in table}
    return baseline, centre


def _model_inputs(summary: dict[str, float], baseline: dict[str, float], names: list[str]) -> list[float]:
    """Same rule as the app: `delta_<f>` is value minus baseline, any other name is the raw value."""
    values = []
    for name in names:
        if name.startswith(BASELINE_DELTA_PREFIX):
            feature = name[len(BASELINE_DELTA_PREFIX) :]
            values.append(summary.get(feature, math.nan) - baseline.get(feature, math.nan))
        else:
            values.append(summary.get(name, math.nan))
    return values


def _input_weights(model: ModelArtifact) -> list[dict[str, Any]]:
    """Per-input influence: logistic coefficients, or summed |first-layer weights| for an MLP."""
    weights = model.weights
    if isinstance(weights, LogisticRegressionWeights):
        values = [float(value) for value in weights.coefficients]
    else:
        first = np.asarray(weights.layers[0].weights, dtype=float)
        values = [float(value) for value in np.abs(first).sum(axis=1)]
    return [{"input": name, "weight": float(value)} for name, value in zip(model.inputs, values, strict=True)]


def wait_for(predicate: Callable[[], bool], timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()
