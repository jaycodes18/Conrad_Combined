"""The feature spec shared by training and the phone.

Everything the on-device port must reproduce lives here: MediaPipe landmark indices, signal thresholds,
window and baseline lengths, and the exact order of every feature list. ``feature_spec.json`` is
``FEATURE_SPEC.to_json()``, so the app and the model can never disagree about an input. No pandas or numpy
imports: this module is loaded by light tools too.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class LandmarkIndices:
    """Indices into MediaPipe Face Landmarker's 478-point mesh."""

    left_eye: tuple[int, ...] = (362, 385, 387, 263, 373, 380)
    right_eye: tuple[int, ...] = (33, 160, 158, 133, 153, 144)
    mouth: tuple[int, ...] = (78, 308, 13, 14)
    right_iris_center: int = 468
    left_iris_center: int = 473
    right_eye_corners: tuple[int, int] = (33, 133)
    left_eye_corners: tuple[int, int] = (362, 263)


@dataclass(frozen=True)
class SignalThresholds:
    """Closure is measured against the eye's own open level in each window, so camera distance cancels out."""

    open_eye_percentile: float = 90.0
    closed_fraction: float = 0.5
    perclos_fraction: float = 0.8
    microsleep_seconds: float = 1.0
    yawn_mar: float = 0.60
    yawn_min_seconds: float = 1.0
    gaze_off_ratio: float = 0.15
    yaw_off_degrees: float = 25.0
    nod_degrees: float = 10.0
    nod_min_seconds: float = 0.5
    nod_max_seconds: float = 3.0
    min_face_fraction: float = 0.6


@dataclass(frozen=True)
class LiveAlertRules:
    """Rules the phone applies frame by frame, outside the model (eyes-off distraction and motion state)."""

    eyes_off_seconds: float = 2.0
    eyes_off_cooldown_seconds: float = 15.0
    moving_speed_mps: float = 4.5
    moving_hold_seconds: float = 5.0
    parked_speed_mps: float = 1.0
    parked_hold_seconds: float = 60.0


@dataclass(frozen=True)
class FeatureSpec:
    version: int = 4
    target_fps: int = 10
    window_seconds: int = 60
    hop_seconds: int = 15
    baseline_seconds: int = 120
    landmarks: LandmarkIndices = field(default_factory=LandmarkIndices)
    thresholds: SignalThresholds = field(default_factory=SignalThresholds)
    live_alerts: LiveAlertRules = field(default_factory=LiveAlertRules)

    @property
    def frames_per_window(self) -> int:
        return self.window_seconds * self.target_fps

    @property
    def frames_per_hop(self) -> int:
        return self.hop_seconds * self.target_fps

    def to_json(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "frame_features": list(FRAME_FEATURES),
            "window_features": list(WINDOW_FEATURES),
        }


FEATURE_SPEC = FeatureSpec()


# ---- Indicator submodels -------------------------------------------------------------------------------------
# Thresholds for the short-horizon indicators in ``driven_ml.indicators``. They are kept out of ``FeatureSpec``,
# so adding or tuning one never changes ``feature_spec.json`` or the drowsiness model's inputs. Where a value
# already exists in ``SignalThresholds``, the default below reads it from there instead of copying the number.


@dataclass(frozen=True)
class EyesConfig:
    """Eyelid closure. Closure is relative to the driver's calibrated open level, per eye."""

    closed_fraction: float = SignalThresholds.closed_fraction
    perclos_fraction: float = SignalThresholds.perclos_fraction
    closure_warning_seconds: float = 2.0
    closure_critical_seconds: float = 5.0  # the team's line for a clear indication
    eye_visibility_cutoff: float = 0.25  # below this, MediaPipe is guessing where the eyelids are
    eye_agreement_tolerance: float = 0.35  # closure difference between the eyes beyond which it is a wink
    bridge_gap_seconds: float = 0.3
    perclos_window_seconds: float = 60.0


@dataclass(frozen=True)
class OcularConfig:
    """Gaze instability over 5 s and 10 s windows, scored against the driver's calibration."""

    short_window_seconds: float = 5.0
    long_window_seconds: float = 10.0
    min_reversal_amplitude: float = 0.04  # iris position units (eye widths); smaller swings are landmark jitter
    off_centre_ratio: float = SignalThresholds.gaze_off_ratio
    erratic_instability: float = 3.0
    scanning_instability: float = 1.0
    min_coverage: float = 0.8  # share of the window with a valid gaze sample
    mirror_glance_max_seconds: float = 1.5
    mirror_zone_tolerance: float = 0.08
    min_mirror_glances: int = 2
    dispersion_std_floor: float = 0.01
    reversal_std_floor: float = 0.1
    off_centre_std_floor: float = 0.05


@dataclass(frozen=True)
class HeadConfig:
    """Head control, relative to the calibrated head pose. Down is negative pitch."""

    nod_drop_degrees: float = 15.0
    nod_recovery_seconds: float = 2.0
    nod_min_seconds: float = SignalThresholds.nod_min_seconds
    nodding_min_per_minute: int = 3
    head_drop_degrees: float = 25.0
    slump_roll_degrees: float = 30.0
    hold_seconds: float = 2.0
    bridge_gap_seconds: float = 0.3


@dataclass(frozen=True)
class BreathingConfig:
    """Audio breathing gating and the experimental gasp-pattern detector."""

    quiet_noise_floor_rms: float = 0.02
    speech_quiet_seconds: float = 10.0
    mic_warmup_seconds: float = 20.0
    cabin_window_seconds: float = 10.0
    cabin_fault_fraction: float = 0.3
    gasp_rms_ratio: float = 6.0  # burst RMS over the cabin's median RMS
    gasp_min_flatness: float = 0.3  # spectral flatness: broadband, not a tone or a voice
    gasp_max_seconds: float = 1.0
    gasp_min_interval_seconds: float = 6.0  # longer than any normal resting breath cycle
    gasp_pattern_window_seconds: float = 60.0
    gasp_min_count: int = 2
    irregular_interval_cv: float = 0.35
    audio_max_age_seconds: float = 2.0
    breath_min_peak_ratio: float = 1.3  # a breath candidate must rise this far over the median envelope


@dataclass(frozen=True)
class ResponsivenessConfig:
    """The visual response check that follows an in-car alert."""

    response_window_seconds: float = 5.0
    open_fraction: float = 0.7
    hold_seconds: float = 0.5
    head_return_degrees: float = 10.0
    gaze_centre_ratio: float = SignalThresholds.gaze_off_ratio
    max_untracked_fraction: float = 0.5


@dataclass(frozen=True)
class SensorQualityConfig:
    """Camera and microphone conditions that limit what the indicators can support."""

    low_light_luma: float = 0.12
    backlit_luma_delta: float = 0.30
    glare_fraction: float = 0.05
    blur_sharpness: float = 12.0  # variance of the Laplacian
    mouth_visibility_cutoff: float = 0.4
    degraded_confidence: float = 0.6


@dataclass(frozen=True)
class FusionConfig:
    """Fusion thresholds. Score levels use the same hysteresis as the Wave B risk engine."""

    elevated_enter: float = 0.45
    elevated_exit: float = 0.35
    high_enter: float = 0.70
    high_exit: float = 0.55
    max_reading_age_seconds: float = 2.0
    probability_max_age_seconds: float = 15.0  # one 15 s hop: a window score stays current until the next one
    repeated_closure_warnings: int = 2
    repeated_closure_window_seconds: float = 60.0
    unresponsive_head_hold_seconds: float = 5.0


@dataclass(frozen=True)
class IndicatorSpec:
    eyes: EyesConfig = field(default_factory=EyesConfig)
    ocular: OcularConfig = field(default_factory=OcularConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    breathing: BreathingConfig = field(default_factory=BreathingConfig)
    responsiveness: ResponsivenessConfig = field(default_factory=ResponsivenessConfig)
    sensor_quality: SensorQualityConfig = field(default_factory=SensorQualityConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)


INDICATOR_SPEC = IndicatorSpec()

FRAME_FEATURES: tuple[str, ...] = (
    "ear_left",
    "ear_right",
    "ear",
    "mar",
    "yaw",
    "pitch",
    "roll",
    "gaze_horizontal",
    "blink_left",
    "blink_right",
    "jaw_open",
    "look_down",
    "face",
)

WINDOW_INPUT_SIGNALS: tuple[str, ...] = (
    "ear",
    "mar",
    "yaw",
    "pitch",
    "gaze_horizontal",
    "face",
)

# Feature spec v4: only quantities a phone mount cannot shift. Absolute eye opening, head pitch and look-down
# depend on where the phone sits; closure relative to the eye's own open level, blink timing, nods and the
# spread of head angle do not. PERCLOS is P80 (Wierwille et al., NHTSA DOT HS 808 247, 1994; Dinges and Grace,
# FHWA-MCRT-98-006, 1998).
WINDOW_FEATURES: tuple[str, ...] = (
    "closure_mean",
    "closure_std",
    "perclos",
    "blink_rate",
    "blink_duration",
    "microsleeps",
    "yawns",
    "pitch_std",
    "yaw_std",
    "head_nods",
    "gaze_off",
    "face_fraction",
)

# Features that come from the eyes. With sunglasses or glare they are NaN at inference and are filled with the
# development medians, so training masks them the same way on a share of windows (feature dropout).
EYE_DERIVED_FEATURES: tuple[str, ...] = (
    "closure_mean",
    "closure_std",
    "perclos",
    "blink_rate",
    "blink_duration",
    "microsleeps",
    "gaze_off",
)
EYE_MASK_TRAINING_FRACTION = 0.3

BASELINE_DELTA_PREFIX = "delta_"
RAW_INPUTS: tuple[str, ...] = WINDOW_FEATURES
INPUTS_WITH_BASELINE: tuple[str, ...] = WINDOW_FEATURES + tuple(
    f"{BASELINE_DELTA_PREFIX}{name}" for name in WINDOW_FEATURES
)

# Team recordings have two clips per session: a calibration clip for the baseline, then the scripted drive.
CLIP_CALIBRATION = "calibration"
CLIP_DRIVE = "drive"
UNLABELED = -1  # drive frames of a team session; their windows are labeled from the script

SCRIPT_COLUMNS: tuple[str, ...] = ("start_s", "end_s", "event")
ACTED_DROWSINESS = "acted_drowsiness"
ACTED_YAWN = "acted_yawn"
ACTED_NOD = "acted_nod"
ACTED_EYES_OFF = "acted_eyes_off"
DROWSY_EVENTS = frozenset({ACTED_DROWSINESS, ACTED_YAWN, ACTED_NOD})
SCRIPT_EVENTS = DROWSY_EVENTS | {ACTED_EYES_OFF}
ALERT_MARGIN_SECONDS = 30  # a drive window is alert only if it is at least this far from every event

FALSE_ALERT_BUDGET_PER_HOUR = 0.2
ALERT_COOLDOWN_SECONDS = 120
