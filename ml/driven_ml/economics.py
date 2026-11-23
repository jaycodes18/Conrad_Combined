"""Unit economics of one paying family, and how many false alerts the price can absorb.

    texts          = drive hours x alerts per hour x confirmed contacts x segments per text
    SMS cost       = texts x (Twilio segment price + carrier fee)
    infrastructure = fixed monthly costs / max(paying families, 1)
    payments       = price x Stripe percent + Stripe fixed fee
    margin         = (price - SMS cost - infrastructure - payments) / price

Each false alert texts every confirmed contact, so it costs money. Every input comes from product.json.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .product import Product, UsageAssumptions
from .workspace import Workspace, write_json

logger = logging.getLogger(__name__)

DEFAULT_PLAN_ID = "family"
_BISECTION_UPPER_PER_HOUR = 100.0
_BISECTION_STEPS = 60


@dataclass(frozen=True)
class UnitEconomics:
    plan_id: str
    price_usd_month: float
    false_alerts_per_hour: float
    texts_per_month: float
    sms_usd_month: float
    infrastructure_usd_month: float
    payments_usd_month: float
    cost_usd_month: float
    contribution_margin: float


@dataclass(frozen=True)
class EconomicsReport:
    unit: UnitEconomics
    target_margin: float
    meets_target: bool
    break_even_false_alerts_per_hour: float
    estimated_assumptions: tuple[str, ...]


def unit_economics(product: Product, false_alerts_per_hour: float, plan_id: str = DEFAULT_PLAN_ID) -> UnitEconomics:
    price = product.plan(plan_id).price_usd_month
    if not price:
        raise ValueError(f"plan {plan_id!r} has no monthly price")
    usage, costs = product.assumptions, product.costs
    texts = 0.0
    if product.notify.drowsy_sms:
        alerts_per_hour = false_alerts_per_hour + usage.true_drowsy_alerts_per_drive_hour
        texts = (
            usage.drive_hours_per_month * alerts_per_hour * usage.confirmed_contacts_per_family * usage.segments_per_sms
        )
    sms = texts * costs.sms_cost_per_segment
    infrastructure = costs.fixed_usd_month / max(usage.paying_families, 1)
    payments = price * costs.stripe_percent + costs.stripe_fixed_usd
    cost = sms + infrastructure + payments
    return UnitEconomics(
        plan_id=plan_id,
        price_usd_month=price,
        false_alerts_per_hour=false_alerts_per_hour,
        texts_per_month=texts,
        sms_usd_month=sms,
        infrastructure_usd_month=infrastructure,
        payments_usd_month=payments,
        cost_usd_month=cost,
        contribution_margin=(price - cost) / price,
    )


def break_even_false_alert_rate(product: Product, plan_id: str = DEFAULT_PLAN_ID) -> float:
    """Highest false-alert rate per drive hour that still meets the margin target; 0.0 if none does."""
    target = product.min_contribution_margin
    if unit_economics(product, 0.0, plan_id).contribution_margin < target:
        return 0.0
    low, high = 0.0, _BISECTION_UPPER_PER_HOUR
    for _ in range(_BISECTION_STEPS):
        middle = (low + high) / 2
        if unit_economics(product, middle, plan_id).contribution_margin >= target:
            low = middle
        else:
            high = middle
    return low


def run_economics(
    workspace: Workspace,
    product: Product,
    false_alerts_per_hour: float,
    short_sha: str,
    plan_id: str = DEFAULT_PLAN_ID,
) -> EconomicsReport:
    unit = unit_economics(product, false_alerts_per_hour, plan_id)
    report = EconomicsReport(
        unit=unit,
        target_margin=product.min_contribution_margin,
        meets_target=unit.contribution_margin >= product.min_contribution_margin,
        break_even_false_alerts_per_hour=break_even_false_alert_rate(product, plan_id),
        estimated_assumptions=UsageAssumptions.names(),
    )
    write_json(workspace.economics_report_path(short_sha), report)
    logger.info(
        "economics: %s plan at %.2f false alerts/h -> margin %.1f%% (target %.0f%%, break-even %.2f/h)",
        plan_id,
        false_alerts_per_hour,
        unit.contribution_margin * 100,
        report.target_margin * 100,
        report.break_even_false_alerts_per_hour,
    )
    return report
