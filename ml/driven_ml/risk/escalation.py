"""From fused risk to recommended actions: the escalation tiers the app acts on. Nothing here sends anything.

Every number comes from ``product.json``: the ``escalation`` block, plus ``notify.min_gap_s`` and
``notify.max_sms_per_driver_per_day``. The tiers:

* Tier 1: an in-car spoken alert on any condition at ``tier1_min_level`` (elevated) or above. The visual
  response check opens.
* Tier 2: ``notify_contacts`` with ``share_location = "precise"`` when the driver does not respond, when a second
  qualifying alert lands within ``notify.min_gap_s`` (600 s), or when ``possible_impairment`` reaches high.
  Contact consent and the daily text cap are respected; without a consented contact, nothing would be sent.
* Tier 3: ``voice_call_recommended`` when ``possible_unresponsiveness`` is high and stays high for
  ``tier3_hold_s`` (10 s) after Tier 2. This flag is the hook for the app's future voice call; no call is placed
  here. An artificial-voice call to a wireless number needs the called party's prior express consent
  (47 CFR 64.1200(a)(1)(iii)), so the flag is true only when a contact's consent record covers calls.
* Emergency services are never contacted automatically: automated calls to 911 or any emergency line are
  prohibited by 47 CFR 64.1200(a)(1)(i), and ``emergency_services`` is always ``"never_automated"``. The app can
  prompt the parent to call 911 themselves.

An episode starts when a condition reaches Tier 1 and ends after ``episode_end_s`` without one. Within an episode
each tier is entered at most once and each action flag is true on exactly one decision, so escalation is
idempotent per episode. Every tier change is a typed ``TierEvent``.
"""

from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..errors import ComplianceError, ConfigurationError, DataError
from ..product import Product
from .fusion import LEVEL_ORDER, Condition, FusionResult, RiskLevel

EMERGENCY_SERVICES = "never_automated"
SECONDS_PER_DAY = 86400.0
SAME_ALERT_SECONDS = 5.0  # a model alert and the Tier 1 alert it causes land a few frames apart: one alert
_PHONE = re.compile(r"^\+[1-9]\d{7,14}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---- contacts (moved from studio/contacts.py) --------------------------------------------------------------
class ConsentState(StrEnum):
    MISSING = "missing"
    RECORDED = "recorded"
    WITHDRAWN = "withdrawn"


@dataclass(frozen=True)
class Contact:
    """A family contact. ``calls_consented`` is the separate prior express consent an automated voice call needs."""

    contact_id: str
    display_name: str
    phone_e164: str
    consent: ConsentState
    consent_date: str
    sms_enabled: bool = True
    calls_consented: bool = False

    def consented_for_sms(self) -> bool:
        return self.consent is ConsentState.RECORDED and self.sms_enabled and bool(_DATE.match(self.consent_date))

    def consented_for_calls(self) -> bool:
        return self.consented_for_sms() and self.calls_consented

    def to_json(self) -> dict[str, Any]:
        return {
            "contact_id": self.contact_id,
            "display_name": self.display_name,
            "phone_masked": _mask_phone(self.phone_e164),
            "consent": self.consent.value,
            "consent_date": self.consent_date,
            "sms_enabled": self.sms_enabled,
            "calls_consented": self.calls_consented,
        }


@dataclass
class ContactBook:
    """In-memory contacts. Phone numbers stay on the object and are masked in JSON."""

    contacts: dict[str, Contact] = field(default_factory=dict)

    def add(
        self,
        contact_id: str,
        display_name: str,
        phone_e164: str,
        *,
        consent: ConsentState = ConsentState.MISSING,
        consent_date: str = "",
        sms_enabled: bool = True,
        calls_consented: bool = False,
    ) -> Contact:
        if not contact_id.strip() or not display_name.strip():
            raise DataError("contact id and display name are required")
        if contact_id in self.contacts:
            raise DataError(f"contact {contact_id} already exists")
        if not _PHONE.match(phone_e164):
            raise DataError("phone must be E.164, such as +12145550100")
        if consent is ConsentState.RECORDED and not _DATE.match(consent_date):
            raise DataError("recorded consent needs a YYYY-MM-DD date")
        contact = Contact(
            contact_id=contact_id.strip(),
            display_name=display_name.strip(),
            phone_e164=phone_e164,
            consent=consent,
            consent_date=consent_date,
            sms_enabled=sms_enabled,
            calls_consented=calls_consented,
        )
        self.contacts[contact.contact_id] = contact
        return contact

    def set_consent(
        self, contact_id: str, consent: ConsentState, consent_date: str, *, calls_consented: bool | None = None
    ) -> Contact:
        current = self._get(contact_id)
        if consent is ConsentState.RECORDED and not _DATE.match(consent_date):
            raise DataError("recorded consent needs a YYYY-MM-DD date")
        recorded = consent is ConsentState.RECORDED
        updated = Contact(
            contact_id=current.contact_id,
            display_name=current.display_name,
            phone_e164=current.phone_e164,
            consent=consent,
            consent_date=consent_date if recorded else "",
            sms_enabled=current.sms_enabled,
            calls_consented=recorded and (current.calls_consented if calls_consented is None else calls_consented),
        )
        self.contacts[contact_id] = updated
        return updated

    def remove(self, contact_id: str) -> None:
        self._get(contact_id)
        del self.contacts[contact_id]

    def sms_recipients(self, *, require_consent: bool) -> list[Contact]:
        if require_consent:
            return [contact for contact in self.contacts.values() if contact.consented_for_sms()]
        return [
            contact
            for contact in self.contacts.values()
            if contact.sms_enabled and contact.consent is not ConsentState.WITHDRAWN
        ]

    def call_recipients(self) -> list[Contact]:
        """Always consent-filtered: the call rule has no product switch."""
        return [contact for contact in self.contacts.values() if contact.consented_for_calls()]

    def _get(self, contact_id: str) -> Contact:
        try:
            return self.contacts[contact_id]
        except KeyError as exc:
            raise DataError(f"unknown contact {contact_id}") from exc

    def to_json(self) -> list[dict[str, Any]]:
        return [contact.to_json() for contact in self.contacts.values()]


def _mask_phone(phone: str) -> str:
    return "hidden" if len(phone) < 4 else f"{phone[:2]}…{phone[-2:]}"


# ---- configuration -----------------------------------------------------------------------------------------
@dataclass(frozen=True)
class EscalationConfig:
    tier1_min_level: RiskLevel
    tier2_impairment_level: RiskLevel
    tier3_condition: Condition
    tier3_hold_seconds: float
    episode_end_seconds: float
    second_alert_window_seconds: float
    max_sms_per_day: int
    drowsy_sms: bool
    eyes_off_sms: bool
    require_contact_consent: bool

    @classmethod
    def from_product(cls, product: Product) -> EscalationConfig:
        rules, notify = product.escalation, product.notify
        try:
            config = cls(
                tier1_min_level=RiskLevel(rules.tier1_min_level),
                tier2_impairment_level=RiskLevel(rules.tier2_impairment_level),
                tier3_condition=Condition(rules.tier3_condition),
                tier3_hold_seconds=rules.tier3_hold_seconds,
                episode_end_seconds=rules.episode_end_seconds,
                second_alert_window_seconds=float(notify.min_gap_seconds),
                max_sms_per_day=notify.max_sms_per_driver_per_day,
                drowsy_sms=notify.drowsy_sms,
                eyes_off_sms=notify.eyes_off_sms,
                require_contact_consent=product.flags.get("require_contact_consent") is not False,
            )
        except ValueError as exc:
            raise ConfigurationError(f"product.json escalation block has a bad value: {exc}") from exc
        if product.flags.get("auto_call_911") is not False:
            raise ComplianceError(
                "automated emergency dispatch is not allowed",
                ["product.json flags.auto_call_911 must be false", "47 CFR 64.1200(a)(1)(i)"],
            )
        if config.tier3_hold_seconds <= 0 or config.episode_end_seconds <= 0 or config.max_sms_per_day < 0:
            raise ConfigurationError("escalation holds must be positive and the daily SMS cap non-negative")
        return config


# ---- events and decisions ----------------------------------------------------------------------------------
class TierEventKind(StrEnum):
    EPISODE_STARTED = "episode_started"
    TIER_CHANGED = "tier_changed"
    CONTACT_SKIPPED = "contact_skipped"
    EPISODE_ENDED = "episode_ended"
    ACKNOWLEDGED = "acknowledged"


@dataclass(frozen=True)
class TierEvent:
    kind: TierEventKind
    t_seconds: float
    episode: int
    tier: int
    previous_tier: int
    reason: str
    contact_ids: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "t": self.t_seconds,
            "episode": self.episode,
            "tier": self.tier,
            "previous_tier": self.previous_tier,
            "reason": self.reason,
            "contact_ids": list(self.contact_ids),
            "dispatched": False,
        }


@dataclass(frozen=True)
class EscalationDecision:
    tier: int
    in_car_alert: bool
    notify_contacts: bool
    share_location: str
    voice_call_recommended: bool
    reason: str
    open_response_window: bool
    events: tuple[TierEvent, ...]
    emergency_services: str = EMERGENCY_SERVICES


@dataclass
class _Episode:
    number: int
    tier: int = 0
    tier2_at: float | None = None
    high_since: float | None = None
    quiet_since: float | None = None
    call_skipped: bool = False


class EscalationEngine:
    """One per driving session. ``step`` once per fusion result, in time order."""

    def __init__(self, config: EscalationConfig, contacts: ContactBook, *, allow_911: bool = False) -> None:
        if allow_911:
            raise ComplianceError(
                "automated emergency dispatch is not allowed",
                ["product.json flags.auto_call_911 is false", "47 CFR 64.1200(a)(1)(i)"],
            )
        self.config = config
        self.contacts = contacts
        self.history: list[TierEvent] = []
        self._episode: _Episode | None = None
        self._episodes = 0
        self._last_qualifying_alert = -math.inf
        self._second_alert_due = False
        self._sms_times: deque[float] = deque()
        self.tier = 0
        self.reason = "No active episode."

    def request_emergency_dispatch(self) -> None:
        raise ComplianceError(
            "automated emergency dispatch is not allowed",
            ["Driven never contacts 911 or emergency services; the app can prompt a parent to call"],
        )

    def acknowledge(self, t_seconds: float, contact_id: str) -> TierEvent:
        """A contact acknowledged the text. It is recorded; it does not lower the tier or stop a voice call."""
        episode = self._episode
        if episode is None or episode.tier < 2:
            raise DataError("no notified episode to acknowledge")
        return self._event(TierEventKind.ACKNOWLEDGED, t_seconds, "contact acknowledged the text", (contact_id,))

    def step(self, result: FusionResult, responsiveness_state: str) -> EscalationDecision:
        t = result.t_seconds
        config = self.config
        events: list[TierEvent] = []
        active = (
            result.condition is not Condition.NONE and LEVEL_ORDER[result.level] >= LEVEL_ORDER[config.tier1_min_level]
        )
        episode = self._episode
        if episode is not None and not active:
            episode.quiet_since = t if episode.quiet_since is None else episode.quiet_since
            if t - episode.quiet_since >= config.episode_end_seconds:
                events.append(self._event(TierEventKind.EPISODE_ENDED, t, "no alert-level condition for a while"))
                self._episode, episode = None, None
                self.tier, self.reason = 0, "No active episode."
        elif episode is not None:
            episode.quiet_since = None
        if episode is None and active:
            self._episodes += 1
            episode = self._episode = _Episode(self._episodes)
            events.append(
                self._event(TierEventKind.EPISODE_STARTED, t, f"{result.condition.value} {result.level.value}")
            )
        if episode is None:
            return self._decision(events)
        in_car = open_window = notify = voice = False
        if episode.tier < 1 and active:
            in_car = open_window = True
            self._raise(episode, 1, t, f"In-car alert: {_label(result)}.", events)
            self.note_alert(t, result.condition)
        if episode.tier == 1:
            if self._second_alert_due:
                notify = self._tier2(episode, t, "A second alert within the notify gap.", events)
            elif responsiveness_state == "no_response":
                notify = self._tier2(episode, t, "No visible response to the in-car alert.", events)
            elif (
                result.condition is Condition.POSSIBLE_IMPAIRMENT
                and LEVEL_ORDER[result.level] >= LEVEL_ORDER[config.tier2_impairment_level]
            ):
                notify = self._tier2(episode, t, "Possible impairment reached high.", events)
        self._second_alert_due = False
        if episode.tier == 2:
            voice = self._tier3(episode, t, result, events)
        return self._decision(
            events,
            in_car=in_car,
            open_window=open_window,
            notify=notify,
            voice=voice,
        )

    # ---- tiers -----------------------------------------------------------------------------------------
    def note_alert(self, t_seconds: float, condition: Condition) -> None:
        """Record a qualifying alert. Tier 1 entries call it; so does Studio for each 60 s model alert it raises.

        A second one within ``notify.min_gap_s`` of the last makes the next step enter Tier 2. Distraction counts
        only when ``notify.eyes_off_sms`` allows it.
        """
        config = self.config
        qualifies = config.eyes_off_sms if condition is Condition.DISTRACTED else config.drowsy_sms
        gap = t_seconds - self._last_qualifying_alert
        if not qualifies or gap < SAME_ALERT_SECONDS:
            return
        if gap <= config.second_alert_window_seconds:
            self._second_alert_due = True
        self._last_qualifying_alert = t_seconds

    def _tier2(self, episode: _Episode, t: float, why: str, events: list[TierEvent]) -> bool:
        """Enter Tier 2; True when a consented contact would actually be texted."""
        config = self.config
        while self._sms_times and self._sms_times[0] <= t - SECONDS_PER_DAY:
            self._sms_times.popleft()
        recipients = self.contacts.sms_recipients(require_consent=config.require_contact_consent)
        skipped: tuple[str, str] | None = None
        if not recipients:
            skipped = ("no contact with recorded consent for texts", "No consented contact, so nothing would be sent.")
        elif len(self._sms_times) >= config.max_sms_per_day:
            skipped = ("daily text cap reached", "The daily text cap is reached, so nothing would be sent.")
        contact_ids = () if skipped else tuple(contact.contact_id for contact in recipients)
        self._raise(episode, 2, t, why, events, contact_ids=contact_ids)
        episode.tier2_at = t
        if skipped is not None:
            events.append(self._event(TierEventKind.CONTACT_SKIPPED, t, skipped[0]))
            self.reason = f"{why} {skipped[1]}"
            return False
        self._sms_times.append(t)
        self.reason = f"{why} Contacts would be texted with the phone's precise location."
        return True

    def _tier3(self, episode: _Episode, t: float, result: FusionResult, events: list[TierEvent]) -> bool:
        config = self.config
        high = result.condition is config.tier3_condition and result.level is RiskLevel.HIGH
        if not high:
            episode.high_since = None
            return False
        if episode.high_since is None:
            episode.high_since = t
        start = max(episode.high_since, episode.tier2_at if episode.tier2_at is not None else t)
        if t - start < config.tier3_hold_seconds:
            return False
        recipients = self.contacts.call_recipients()
        if not recipients:
            if not episode.call_skipped:
                episode.call_skipped = True
                events.append(self._event(TierEventKind.CONTACT_SKIPPED, t, "no contact consented to automated calls"))
                self.reason = "Still possibly unresponsive, but no contact consented to calls; texts only."
            return False
        self._raise(
            episode,
            3,
            t,
            "Possible unresponsiveness stayed high after the texts. A voice call would be recommended.",
            events,
            contact_ids=tuple(contact.contact_id for contact in recipients),
        )
        return True

    def _raise(
        self,
        episode: _Episode,
        tier: int,
        t: float,
        why: str,
        events: list[TierEvent],
        *,
        contact_ids: tuple[str, ...] = (),
    ) -> None:
        previous = episode.tier
        episode.tier = tier
        self.tier, self.reason = tier, why
        events.append(self._event(TierEventKind.TIER_CHANGED, t, why, contact_ids, previous_tier=previous))

    # ---- helpers -----------------------------------------------------------------------------------------
    def _event(
        self,
        kind: TierEventKind,
        t: float,
        reason: str,
        contact_ids: tuple[str, ...] = (),
        *,
        previous_tier: int | None = None,
    ) -> TierEvent:
        episode = self._episode
        number = episode.number if episode is not None else self._episodes
        tier = episode.tier if episode is not None else 0
        event = TierEvent(
            kind=kind,
            t_seconds=t,
            episode=number,
            tier=tier,
            previous_tier=tier if previous_tier is None else previous_tier,
            reason=reason,
            contact_ids=contact_ids,
        )
        self.history.append(event)
        return event

    def _decision(
        self,
        events: list[TierEvent],
        *,
        in_car: bool = False,
        open_window: bool = False,
        notify: bool = False,
        voice: bool = False,
    ) -> EscalationDecision:
        return EscalationDecision(
            tier=self.tier,
            in_car_alert=in_car,
            notify_contacts=notify,
            share_location="precise" if notify else "none",
            voice_call_recommended=voice,
            reason=self.reason,
            open_response_window=open_window,
            events=tuple(events),
        )


def _label(result: FusionResult) -> str:
    return f"{result.condition.value.replace('_', ' ')} ({result.level.value})"
