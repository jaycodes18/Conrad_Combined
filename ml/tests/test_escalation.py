"""Escalation tiers: transitions, consent, the daily text cap, the Tier 3 hold, idempotence, and no automated 911."""

from __future__ import annotations

from dataclasses import replace

import pytest
from conftest import consented_book

from driven_ml.errors import ComplianceError, DataError
from driven_ml.product import load_product, parse_product
from driven_ml.risk import (
    EMERGENCY_SERVICES,
    Condition,
    ConsentState,
    ContactBook,
    EscalationConfig,
    EscalationEngine,
    FusionResult,
    RiskLevel,
    TierEventKind,
)


def result(t: float, condition: Condition = Condition.NONE, level: RiskLevel = RiskLevel.NONE) -> FusionResult:
    return FusionResult(
        t_seconds=t,
        condition=condition,
        level=level,
        score=0.0,
        confidence=0.8,
        reasons=(),
        scores={},
        monitoring_degraded=False,
        degraded_situations=(),
        readings={},
    )


def steps(
    engine: EscalationEngine,
    start: float,
    seconds: float,
    condition: Condition,
    level: RiskLevel,
    *,
    response: str = "idle",
):
    return [engine.step(result(start + index / 10, condition, level), response) for index in range(round(seconds * 10))]


def test_tier_one_is_an_in_car_alert_once_per_episode(escalation_config: EscalationConfig) -> None:
    engine = EscalationEngine(escalation_config, consented_book())
    decisions = steps(engine, 0.0, 3.0, Condition.DROWSY, RiskLevel.ELEVATED)
    assert decisions[0].tier == 1
    assert decisions[0].in_car_alert
    assert decisions[0].open_response_window
    assert not any(decision.in_car_alert for decision in decisions[1:])
    assert all(decision.emergency_services == EMERGENCY_SERVICES for decision in decisions)
    kinds = [event.kind for event in engine.history]
    assert kinds == [TierEventKind.EPISODE_STARTED, TierEventKind.TIER_CHANGED]


def test_no_response_texts_the_consented_contact_once_with_precise_location(
    escalation_config: EscalationConfig,
) -> None:
    engine = EscalationEngine(escalation_config, consented_book())
    steps(engine, 0.0, 1.0, Condition.DROWSY, RiskLevel.ELEVATED, response="waiting")
    after = steps(engine, 1.0, 2.0, Condition.DROWSY, RiskLevel.ELEVATED, response="no_response")
    assert after[0].tier == 2
    assert after[0].notify_contacts
    assert after[0].share_location == "precise"
    assert not any(decision.notify_contacts for decision in after[1:])
    tier2 = [event for event in engine.history if event.kind is TierEventKind.TIER_CHANGED and event.tier == 2]
    assert len(tier2) == 1
    assert tier2[0].contact_ids == ("parent",)
    assert tier2[0].to_json()["dispatched"] is False


def test_second_qualifying_alert_within_the_gap_escalates(escalation_config: EscalationConfig) -> None:
    engine = EscalationEngine(escalation_config, consented_book())
    steps(engine, 0.0, 1.0, Condition.DROWSY, RiskLevel.ELEVATED)
    steps(engine, 1.0, 31.0, Condition.NONE, RiskLevel.NONE)  # the episode ends after 30 s quiet
    assert engine.tier == 0
    second = engine.step(result(200.0, Condition.DROWSY, RiskLevel.ELEVATED), "idle")
    assert second.tier == 2
    assert second.in_car_alert
    assert second.notify_contacts
    late = EscalationEngine(escalation_config, consented_book())
    steps(late, 0.0, 1.0, Condition.DROWSY, RiskLevel.ELEVATED)
    steps(late, 1.0, 31.0, Condition.NONE, RiskLevel.NONE)
    assert late.step(result(700.0, Condition.DROWSY, RiskLevel.ELEVATED), "idle").tier == 1


def test_eyes_off_episodes_do_not_count_toward_texts(escalation_config: EscalationConfig) -> None:
    assert escalation_config.eyes_off_sms is False
    engine = EscalationEngine(escalation_config, consented_book())
    steps(engine, 0.0, 1.0, Condition.DISTRACTED, RiskLevel.ELEVATED)
    steps(engine, 1.0, 31.0, Condition.NONE, RiskLevel.NONE)
    assert engine.step(result(100.0, Condition.DISTRACTED, RiskLevel.ELEVATED), "idle").tier == 1


def test_possible_impairment_high_escalates_to_tier_two(escalation_config: EscalationConfig) -> None:
    engine = EscalationEngine(escalation_config, consented_book())
    steps(engine, 0.0, 1.0, Condition.POSSIBLE_IMPAIRMENT, RiskLevel.ELEVATED)
    assert engine.tier == 1
    decision = engine.step(result(1.0, Condition.POSSIBLE_IMPAIRMENT, RiskLevel.HIGH), "responded")
    assert decision.tier == 2
    assert decision.notify_contacts


def test_without_consent_nothing_would_be_sent(escalation_config: EscalationConfig) -> None:
    book = ContactBook()
    book.add("parent", "Parent", "+12145550100")  # consent missing
    engine = EscalationEngine(escalation_config, book)
    steps(engine, 0.0, 1.0, Condition.DROWSY, RiskLevel.ELEVATED)
    decision = engine.step(result(1.0, Condition.DROWSY, RiskLevel.ELEVATED), "no_response")
    assert decision.tier == 2
    assert not decision.notify_contacts
    assert decision.share_location == "none"
    assert "No consented contact" in decision.reason
    assert engine.history[-1].kind is TierEventKind.CONTACT_SKIPPED


def test_the_daily_text_cap_is_respected(escalation_config: EscalationConfig) -> None:
    capped = replace(escalation_config, max_sms_per_day=1)
    engine = EscalationEngine(capped, consented_book())
    notified = []
    for episode in range(3):
        start = episode * 100.0
        steps(engine, start, 1.0, Condition.DROWSY, RiskLevel.ELEVATED)
        notified += [
            d.notify_contacts
            for d in steps(engine, start + 1, 1.0, Condition.DROWSY, RiskLevel.ELEVATED, response="no_response")
        ]
        steps(engine, start + 2, 31.0, Condition.NONE, RiskLevel.NONE)
    assert sum(notified) == 1
    assert any(event.reason == "daily text cap reached" for event in engine.history)
    assert load_product().notify.max_sms_per_driver_per_day == escalation_config.max_sms_per_day


def unresponsive_after_tier_two(engine: EscalationEngine, hold: float):
    steps(engine, 0.0, 1.0, Condition.POSSIBLE_UNRESPONSIVENESS, RiskLevel.ELEVATED, response="waiting")
    steps(engine, 1.0, 0.1, Condition.POSSIBLE_UNRESPONSIVENESS, RiskLevel.HIGH, response="no_response")
    return steps(engine, 1.1, hold, Condition.POSSIBLE_UNRESPONSIVENESS, RiskLevel.HIGH, response="no_response")


def test_tier_three_needs_ten_seconds_of_high_after_tier_two(escalation_config: EscalationConfig) -> None:
    assert escalation_config.tier3_hold_seconds == 10.0
    engine = EscalationEngine(escalation_config, consented_book(calls=True))
    held = unresponsive_after_tier_two(engine, 11.0)
    recommended = [index for index, decision in enumerate(held) if decision.voice_call_recommended]
    assert len(recommended) == 1
    first = held[recommended[0]]
    assert first.tier == 3
    assert 1.1 + recommended[0] / 10 - 1.0 == pytest.approx(10.0, abs=0.11)
    assert all(decision.emergency_services == "never_automated" for decision in held)
    dropped = EscalationEngine(escalation_config, consented_book(calls=True))
    unresponsive_after_tier_two(dropped, 5.0)
    steps(dropped, 6.1, 1.0, Condition.POSSIBLE_UNRESPONSIVENESS, RiskLevel.ELEVATED, response="no_response")
    restarted = steps(dropped, 7.1, 9.0, Condition.POSSIBLE_UNRESPONSIVENESS, RiskLevel.HIGH, response="no_response")
    assert not any(decision.voice_call_recommended for decision in restarted)  # the hold restarted at 7.1 s


def test_tier_three_needs_call_consent(escalation_config: EscalationConfig) -> None:
    engine = EscalationEngine(escalation_config, consented_book(calls=False))
    held = unresponsive_after_tier_two(engine, 15.0)
    assert not any(decision.voice_call_recommended for decision in held)
    assert engine.tier == 2
    skipped = [event for event in engine.history if event.reason == "no contact consented to automated calls"]
    assert len(skipped) == 1


def test_automated_emergency_dispatch_is_refused(escalation_config: EscalationConfig) -> None:
    with pytest.raises(ComplianceError):
        EscalationEngine(escalation_config, ContactBook(), allow_911=True)
    engine = EscalationEngine(escalation_config, ContactBook())
    with pytest.raises(ComplianceError):
        engine.request_emergency_dispatch()
    raw = dict(load_product().raw)
    raw["flags"] = {**raw["flags"], "auto_call_911": True}
    with pytest.raises(ComplianceError):
        EscalationConfig.from_product(parse_product(raw))


def test_acknowledgement_is_recorded_only_after_a_text(escalation_config: EscalationConfig) -> None:
    engine = EscalationEngine(escalation_config, consented_book())
    with pytest.raises(DataError):
        engine.acknowledge(0.0, "parent")
    steps(engine, 0.0, 1.0, Condition.DROWSY, RiskLevel.ELEVATED)
    engine.step(result(1.0, Condition.DROWSY, RiskLevel.ELEVATED), "no_response")
    event = engine.acknowledge(2.0, "parent")
    assert event.kind is TierEventKind.ACKNOWLEDGED
    assert engine.tier == 2


def test_contact_book_validates_and_masks() -> None:
    book = ContactBook()
    with pytest.raises(DataError, match=r"E\.164"):
        book.add("a", "A", "2145550100")
    with pytest.raises(DataError, match="YYYY-MM-DD"):
        book.add("a", "A", "+12145550100", consent=ConsentState.RECORDED, consent_date="pending")
    book.add("a", "A", "+12145550100", consent=ConsentState.RECORDED, consent_date="2026-11-01", calls_consented=True)
    assert book.to_json()[0]["phone_masked"] == "+1…00"
    assert [contact.contact_id for contact in book.call_recipients()] == ["a"]
    withdrawn = book.set_consent("a", ConsentState.WITHDRAWN, "")
    assert not withdrawn.calls_consented
    assert book.sms_recipients(require_consent=True) == []
