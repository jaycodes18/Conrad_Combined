"""Named driver profiles and the calibration saved for each one.

The live engine still calibrates one session at a time. This book remembers that result so a family
can switch drivers without pretending an uncalibrated person has a baseline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import DataError
from ..workspace import write_text_atomic


@dataclass
class DriverProfile:
    driver_id: str
    display_name: str
    calibrated: bool = False
    baseline: dict[str, float] | None = None
    centre: dict[str, float] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "driver_id": self.driver_id,
            "display_name": self.display_name,
            "calibrated": self.calibrated,
            "baseline": self.baseline,
            "centre": self.centre,
        }


class DriverBook:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.profiles: dict[str, DriverProfile] = {}
        self.active_id = ""
        if path.is_file():
            self._load()

    def add(self, driver_id: str, display_name: str) -> DriverProfile:
        if not driver_id.strip() or not display_name.strip():
            raise DataError("driver id and name are required")
        if driver_id in self.profiles:
            raise DataError(f"driver {driver_id} already exists")
        profile = DriverProfile(driver_id.strip(), display_name.strip())
        self.profiles[profile.driver_id] = profile
        if not self.active_id:
            self.active_id = profile.driver_id
        self._save()
        return profile

    def activate(self, driver_id: str) -> DriverProfile:
        profile = self._get(driver_id)
        self.active_id = profile.driver_id
        self._save()
        return profile

    def save_calibration(
        self, driver_id: str, baseline: dict[str, float] | None, centre: dict[str, float]
    ) -> DriverProfile:
        current = self._get(driver_id)
        current.baseline = None if baseline is None else {key: float(value) for key, value in baseline.items()}
        current.centre = {key: float(value) for key, value in centre.items()}
        current.calibrated = baseline is not None
        self._save()
        return current

    def active(self) -> DriverProfile | None:
        if not self.active_id:
            return None
        return self.profiles.get(self.active_id)

    def to_json(self) -> dict[str, Any]:
        return {
            "active_id": self.active_id,
            "profiles": [profile.to_json() for profile in self.profiles.values()],
        }

    def _get(self, driver_id: str) -> DriverProfile:
        try:
            return self.profiles[driver_id]
        except KeyError as exc:
            raise DataError(f"unknown driver {driver_id}") from exc

    def _load(self) -> None:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.active_id = str(raw.get("active_id", ""))
        for item in raw.get("profiles", []):
            profile = DriverProfile(
                str(item["driver_id"]),
                str(item["display_name"]),
                bool(item.get("calibrated", False)),
                item.get("baseline"),
                item.get("centre"),
            )
            self.profiles[profile.driver_id] = profile

    def _save(self) -> None:
        write_text_atomic(self.path, json.dumps(self.to_json(), indent=2) + "\n")
