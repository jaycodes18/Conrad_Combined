"""Optional driving-stability inputs from vehicle telemetry or outward cameras.

A driver-facing camera alone cannot establish lane position. When no outward or
telemetry input is provided, driving stability is reported unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .observations import SensorObservation, SensorStatus, SourceKind


@dataclass(frozen=True)
class DrivingSample:
    t_seconds: float
    lateral_accel_mps2: float | None = None
    yaw_rate_dps: float | None = None
    speed_mps: float | None = None
    lane_offset_m: float | None = None
    source_kind: SourceKind = SourceKind.HARDWARE


@dataclass
class DrivingStabilityTracker:
    """Scores instability only from explicit telemetry fields."""

    last: SensorObservation | None = None

    def unavailable(self, t_seconds: float) -> SensorObservation:
        obs = SensorObservation(
            name="driving_stability",
            t_seconds=t_seconds,
            value=None,
            confidence=0.0,
            status=SensorStatus.UNAVAILABLE,
            valid=False,
            source_kind=SourceKind.HARDWARE,
            detail=("driving stability unavailable; requires outward-facing camera or vehicle telemetry"),
        )
        self.last = obs
        return obs

    def push(self, sample: DrivingSample | None) -> SensorObservation:
        if sample is None:
            return self.unavailable(0.0)
        values = [
            sample.lateral_accel_mps2,
            sample.yaw_rate_dps,
            sample.lane_offset_m,
        ]
        present = [value for value in values if value is not None]
        if not present:
            return self.unavailable(sample.t_seconds)

        # Normalize crude magnitudes into 0..1 instability.
        score = 0.0
        parts: list[str] = []
        if sample.lateral_accel_mps2 is not None:
            lateral = min(1.0, abs(sample.lateral_accel_mps2) / 3.0)
            score = max(score, lateral)
            parts.append(f"lateral_accel={sample.lateral_accel_mps2:.2f}")
        if sample.yaw_rate_dps is not None:
            yaw = min(1.0, abs(sample.yaw_rate_dps) / 30.0)
            score = max(score, yaw)
            parts.append(f"yaw_rate={sample.yaw_rate_dps:.2f}")
        if sample.lane_offset_m is not None:
            lane = min(1.0, abs(sample.lane_offset_m) / 0.8)
            score = max(score, lane)
            parts.append(f"lane_offset={sample.lane_offset_m:.2f}")
        obs = SensorObservation(
            name="driving_stability",
            t_seconds=sample.t_seconds,
            value=float(score),
            confidence=0.7 if sample.lane_offset_m is not None else 0.5,
            status=SensorStatus.AVAILABLE,
            valid=True,
            source_kind=sample.source_kind,
            detail="telemetry: " + ", ".join(parts),
        )
        self.last = obs
        return obs

    def to_json(self) -> dict[str, Any]:
        if self.last is None:
            return {"status": "unavailable"}
        return self.last.to_json()
