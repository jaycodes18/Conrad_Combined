"""Typed view of product.json, the single source of prices, rates, flags and planning assumptions.

The pipeline never hard-codes a price or a legal flag: the compliance gate reads the flags from here and the
unit-economics check reads the costs and assumptions from here. No pandas or numpy imports.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from .errors import ConfigurationError
from .settings import PRODUCT_FILE_NAME, find_repo_root


@dataclass(frozen=True)
class Plan:
    plan_id: str
    name: str
    price_usd_month: float | None
    parent_sms: bool
    summary: str
    features: tuple[str, ...]


@dataclass(frozen=True)
class NotifyRules:
    drowsy_sms: bool
    eyes_off_sms: bool
    min_gap_seconds: int
    max_sms_per_driver_per_day: int
    stale_event_seconds: int


@dataclass(frozen=True)
class EscalationRules:
    """When the risk signal recommends each escalation tier; the gap and the daily cap come from ``notify``."""

    tier1_min_level: str
    tier2_impairment_level: str
    tier3_condition: str
    tier3_hold_seconds: float
    episode_end_seconds: float


@dataclass(frozen=True)
class CostModel:
    sms_segment_usd: float
    sms_carrier_fee_usd: float
    phone_number_usd_month: float
    supabase_pro_usd_month: float
    cloudflare_pages_usd_month: float
    stripe_percent: float
    stripe_fixed_usd: float

    @property
    def sms_cost_per_segment(self) -> float:
        return self.sms_segment_usd + self.sms_carrier_fee_usd

    @property
    def fixed_usd_month(self) -> float:
        return self.supabase_pro_usd_month + self.phone_number_usd_month + self.cloudflare_pages_usd_month


@dataclass(frozen=True)
class UsageAssumptions:
    """Team planning estimates; each is replaced by the measured pilot value before launch."""

    drive_hours_per_month: float
    true_drowsy_alerts_per_drive_hour: float
    confirmed_contacts_per_family: float
    segments_per_sms: float
    paying_families: int

    @staticmethod
    def names() -> tuple[str, ...]:
        return tuple(item.name for item in fields(UsageAssumptions))


@dataclass(frozen=True)
class Product:
    name: str
    version: int
    contact_email: str
    flags: Mapping[str, Any]
    plans: tuple[Plan, ...]
    notify: NotifyRules
    escalation: EscalationRules
    costs: CostModel
    assumptions: UsageAssumptions
    min_contribution_margin: float
    stripe_payment_link: str
    raw: Mapping[str, Any]

    def plan(self, plan_id: str) -> Plan:
        for plan in self.plans:
            if plan.plan_id == plan_id:
                return plan
        raise ConfigurationError(f"product.json has no plan {plan_id!r}")


def _optional_price(value: Any) -> float | None:
    return None if value is None else float(value)


def parse_product(raw: Mapping[str, Any]) -> Product:
    try:
        notify, costs, assumptions = raw["notify"], raw["costs"], raw["assumptions"]
        escalation = raw["escalation"]
        return Product(
            name=str(raw["name"]),
            version=int(raw["version"]),
            contact_email=str(raw.get("contact_email", "")),
            flags=dict(raw["flags"]),
            plans=tuple(
                Plan(
                    plan_id=str(plan["id"]),
                    name=str(plan["name"]),
                    price_usd_month=_optional_price(plan["price_usd_month"]),
                    parent_sms=bool(plan["parent_sms"]),
                    summary=str(plan["summary"]),
                    features=tuple(str(feature) for feature in plan["features"]),
                )
                for plan in raw["plans"]
            ),
            notify=NotifyRules(
                drowsy_sms=bool(notify["drowsy_sms"]),
                eyes_off_sms=bool(notify["eyes_off_sms"]),
                min_gap_seconds=int(notify["min_gap_s"]),
                max_sms_per_driver_per_day=int(notify["max_sms_per_driver_per_day"]),
                stale_event_seconds=int(notify["stale_event_s"]),
            ),
            escalation=EscalationRules(
                tier1_min_level=str(escalation["tier1_min_level"]),
                tier2_impairment_level=str(escalation["tier2_impairment_level"]),
                tier3_condition=str(escalation["tier3_condition"]),
                tier3_hold_seconds=float(escalation["tier3_hold_s"]),
                episode_end_seconds=float(escalation["episode_end_s"]),
            ),
            costs=CostModel(
                sms_segment_usd=float(costs["sms_segment_usd"]),
                sms_carrier_fee_usd=float(costs["sms_carrier_fee_usd"]),
                phone_number_usd_month=float(costs["phone_number_usd_month"]),
                supabase_pro_usd_month=float(costs["supabase_pro_usd_month"]),
                cloudflare_pages_usd_month=float(costs["cloudflare_pages_usd_month"]),
                stripe_percent=float(costs["stripe_pct"]),
                stripe_fixed_usd=float(costs["stripe_fixed_usd"]),
            ),
            assumptions=UsageAssumptions(
                drive_hours_per_month=float(assumptions["drive_hours_per_month"]),
                true_drowsy_alerts_per_drive_hour=float(assumptions["true_drowsy_alerts_per_drive_hour"]),
                confirmed_contacts_per_family=float(assumptions["confirmed_contacts_per_family"]),
                segments_per_sms=float(assumptions["segments_per_sms"]),
                paying_families=int(assumptions["paying_families"]),
            ),
            min_contribution_margin=float(raw["targets"]["min_contribution_margin"]),
            stripe_payment_link=str(raw.get("billing", {}).get("stripe_payment_link", "")),
            raw=raw,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"product.json is missing or has a bad value: {exc!r}") from exc


def load_product(repo_root: Path | None = None) -> Product:
    root = repo_root or find_repo_root()
    if root is None:
        raise ConfigurationError(f"no {PRODUCT_FILE_NAME} found; run from inside the Driven repository")
    path = root / PRODUCT_FILE_NAME
    if not path.is_file():
        raise ConfigurationError(f"{path} does not exist")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigurationError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigurationError(f"{path} must hold a JSON object")
    return parse_product(raw)
