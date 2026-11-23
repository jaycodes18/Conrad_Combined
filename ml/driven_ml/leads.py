"""Summarize the landing page's lead export for the weekly go-to-market review.

Parents are the first customers; dashcam makers, insurers and driver-education programs are partner leads
that get a personal reply. The export comes from the backend's leads table as a CSV.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .errors import DataError

LEAD_COLUMNS: tuple[str, ...] = (
    "created_at",
    "email",
    "role",
    "state",
    "utm_source",
    "teen_drivers",
    "sms_marketing_consent",
)
PARENT_ROLE = "parent"
PARTNER_ROLES = frozenset({"dashcam_maker", "insurer", "driver_education"})
_PARTNER_COLUMNS = ["created_at", "email", "role", "state", "utm_source"]
_TOP_STATES = 10


@dataclass(frozen=True, eq=False)
class LeadsSummary:
    total_leads: int
    teen_drivers: int
    sms_opt_in_rate: float
    by_week: pd.DataFrame
    by_role: pd.DataFrame
    by_source: pd.DataFrame
    by_state: pd.DataFrame
    partner_leads: pd.DataFrame


def summarize_leads(frame: pd.DataFrame) -> LeadsSummary:
    missing = [column for column in LEAD_COLUMNS if column not in frame.columns]
    if missing:
        raise DataError(f"leads export is missing columns: {', '.join(missing)}")
    leads = frame.copy()
    leads["utm_source"] = leads["utm_source"].fillna("direct")
    leads["created_at"] = pd.to_datetime(leads["created_at"], utc=True)
    created_utc = leads["created_at"].dt.tz_convert(None)
    week_start = created_utc.dt.normalize() - pd.to_timedelta(created_utc.dt.day_of_week, unit="D")
    is_parent = leads["role"] == PARENT_ROLE
    is_partner = leads["role"].isin(PARTNER_ROLES)

    by_week = leads.groupby(week_start.rename("week_start")).size().rename("leads").reset_index()
    by_role = leads["role"].value_counts().rename_axis("role").rename("leads").reset_index()
    by_source = (
        leads.assign(parents=is_parent.astype(int), partners=is_partner.astype(int))
        .groupby("utm_source", as_index=False)
        .agg(
            leads=("email", "size"),
            parents=("parents", "sum"),
            partners=("partners", "sum"),
        )
        .sort_values(["leads", "utm_source"], ascending=[False, True], kind="stable")
        .reset_index(drop=True)
    )
    by_state = leads["state"].value_counts().head(_TOP_STATES).rename_axis("state").rename("leads").reset_index()
    partner_leads = (
        leads.loc[is_partner, _PARTNER_COLUMNS]
        .sort_values("created_at", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    consent = leads["sms_marketing_consent"].eq(True)
    return LeadsSummary(
        total_leads=len(leads),
        teen_drivers=int(pd.to_numeric(leads["teen_drivers"], errors="coerce").fillna(0).sum()),
        sms_opt_in_rate=float(consent.mean()) if len(leads) else float("nan"),
        by_week=by_week,
        by_role=by_role,
        by_source=by_source,
        by_state=by_state,
        partner_leads=partner_leads,
    )
