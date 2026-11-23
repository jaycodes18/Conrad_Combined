"""Fusion rules 1-4, positive and negative, plus the ported Wave B hysteresis, staleness and fault handling."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from driven_ml.errors import DataError
from driven_ml.indicators import BEHAVIOUR_INDICATORS, Coverage, IndicatorName, IndicatorReading, Situation
from driven_ml.indicators import unavailable_reading as missing
from driven_ml.risk import Condition, FusionInputs, RiskFusion, RiskLevel, fit_fusion
from driven_ml.risk.fusion import FUSION_FEATURES, Hysteresis
from driven_ml.settings import DataSource, Track
from driven_ml.spec import FusionConfig
from driven_ml.splits import is_locked_test_subject
from driven_ml.staging.gate import fusion_weights_check

T = 100.0
ALL_AVAILABLE = dict.fromkeys(BEHAVIOUR_INDICATORS, Coverage.AVAILABLE)


def reading(name: IndicatorName, state: str, value: float | None = None, *, t: float = T) -> IndicatorReading:
    return IndicatorReading(
        name=name,
        t_seconds=t,
        available=True,
        state=state,
        value=value,
        severity=0.5,
        confidence=0.9,
        evidence=(f"{name.value} is {state}.",),
        unavailable_reason=None,
    )


def assess(
    *readings: IndicatorReading,
    driving: bool = True,
    fusion: RiskFusion | None = None,
    t: float = T,
    coverage: dict[IndicatorName, Coverage] | None = None,
    situations: tuple[Situation, ...] = (),
    probability: float | None = None,
    probability_t: float | None = None,
    eyes_off_seconds: float = 0.0,
):
    engine = fusion or RiskFusion()
    return engine.assess(
        FusionInputs(
            t_seconds=t,
            readings={item.name: item for item in readings},
            driving=driving,
            coverage=coverage or ALL_AVAILABLE,
            situations=situations,
            drowsiness_probability=probability,
            probability_t_seconds=t if probability_t is None and probability is not None else probability_t,
            probability_threshold=0.6,
            eyes_off_seconds=eyes_off_seconds,
        )
    )


EYES_CRITICAL = reading(IndicatorName.EYES, "closure_critical", 5.2)
HEAD_DROP = reading(IndicatorName.HEAD, "head_drop", 6.0)
NO_RESPONSE = reading(IndicatorName.RESPONSIVENESS, "no_response")
AGONAL = reading(IndicatorName.BREATHING, "agonal_pattern_suspected")
QUIET = reading(IndicatorName.BREATHING, "not_detected_quiet_cabin")
REGULAR = reading(IndicatorName.BREATHING, "regular", 14.0)
NODDING = reading(IndicatorName.HEAD, "nodding", 4.0)
ERRATIC_5 = reading(IndicatorName.OCULAR, "erratic_5s", 4.0)
ERRATIC_10 = reading(IndicatorName.OCULAR, "erratic_10s", 4.0)


# ---- rule 1: possible unresponsiveness --------------------------------------------------------------------
@pytest.mark.parametrize("confirming", [HEAD_DROP, NO_RESPONSE, AGONAL, QUIET], ids=lambda item: item.state)
def test_closure_critical_with_a_confirming_sign_is_high(confirming: IndicatorReading) -> None:
    result = assess(EYES_CRITICAL, confirming, driving=False)
    assert result.condition is Condition.POSSIBLE_UNRESPONSIVENESS
    assert result.level is RiskLevel.HIGH
    assert "eyes is closure_critical." in result.reasons
    assert f"{confirming.name.value} is {confirming.state}." in result.reasons


def test_breathing_alone_never_triggers_unresponsiveness() -> None:
    for breathing in (AGONAL, QUIET):
        result = assess(breathing, reading(IndicatorName.EYES, "open", 0.0), driving=False)
        assert result.condition is Condition.NONE
        assert result.scores[Condition.POSSIBLE_UNRESPONSIVENESS] == 0.0


def test_regular_breathing_does_not_veto() -> None:
    result = assess(EYES_CRITICAL, HEAD_DROP, REGULAR, driving=False)
    assert result.condition is Condition.POSSIBLE_UNRESPONSIVENESS
    assert result.level is RiskLevel.HIGH


def test_one_primary_sign_alone_is_elevated_not_high() -> None:
    parked = assess(EYES_CRITICAL, driving=False)
    assert parked.condition is Condition.POSSIBLE_UNRESPONSIVENESS
    assert parked.level is RiskLevel.ELEVATED
    moving = assess(EYES_CRITICAL)
    assert moving.condition is Condition.DROWSY  # closure_critical alone is drowsy high while moving
    assert moving.level is RiskLevel.HIGH


def test_sunglasses_fall_back_to_a_five_second_head_drop_with_no_response() -> None:
    hidden_eyes = missing(IndicatorName.EYES, T, "eyes_not_visible")
    result = assess(hidden_eyes, HEAD_DROP, NO_RESPONSE, driving=False)
    assert result.condition is Condition.POSSIBLE_UNRESPONSIVENESS
    assert result.level is RiskLevel.HIGH
    short = assess(hidden_eyes, reading(IndicatorName.HEAD, "head_drop", 3.0), NO_RESPONSE, driving=False)
    assert short.level is RiskLevel.ELEVATED  # held under 5 s
    responding = assess(hidden_eyes, HEAD_DROP, reading(IndicatorName.RESPONSIVENESS, "waiting", 2.0))
    assert responding.level is not RiskLevel.HIGH


# ---- rule 2: possible impairment ----------------------------------------------------------------------------
def test_two_families_are_impairment_and_ocular_10s_makes_it_high() -> None:
    elevated = assess(ERRATIC_5, NODDING)
    assert elevated.condition is Condition.POSSIBLE_IMPAIRMENT
    assert elevated.level is RiskLevel.ELEVATED
    high = assess(ERRATIC_10, NODDING)
    assert high.condition is Condition.POSSIBLE_IMPAIRMENT
    assert high.level is RiskLevel.HIGH
    with_model = assess(NODDING, probability=0.8)
    assert with_model.condition is Condition.POSSIBLE_IMPAIRMENT  # rule 2 outranks rule 3 at the same level
    assert with_model.scores[Condition.DROWSY] > with_model.score


def test_one_family_alone_is_never_impairment() -> None:
    assert assess(ERRATIC_10).condition is Condition.DISTRACTED
    assert assess(ERRATIC_10).level is RiskLevel.LOW
    assert assess(NODDING).condition is Condition.DROWSY
    assert assess(probability=0.9).condition is Condition.DROWSY
    assert assess(ERRATIC_10, NODDING, driving=False).condition is Condition.NONE  # parked: no impairment


def test_repeated_closure_warnings_count_as_a_family() -> None:
    fusion = RiskFusion()
    for offset, state in ((0.0, "closure_warning"), (5.0, "open"), (20.0, "closure_warning"), (25.0, "open")):
        assess(reading(IndicatorName.EYES, state, 2.1, t=T + offset), fusion=fusion, t=T + offset)
    ocular = reading(IndicatorName.OCULAR, "erratic_5s", 4.0, t=T + 30)
    result = assess(ocular, reading(IndicatorName.EYES, "open", 0.0, t=T + 30), fusion=fusion, t=T + 30)
    assert fusion.closure_warnings_last_minute == 2
    assert result.condition is Condition.POSSIBLE_IMPAIRMENT


# ---- rules 3 and 4 --------------------------------------------------------------------------------------------
def test_drowsy_from_the_model_or_closure_while_moving() -> None:
    assert assess(probability=0.7).condition is Condition.DROWSY
    assert assess(probability=0.5).condition is Condition.NONE
    warning = assess(reading(IndicatorName.EYES, "closure_warning", 2.0))
    assert warning.condition is Condition.DROWSY
    assert warning.level is RiskLevel.ELEVATED
    assert assess(reading(IndicatorName.EYES, "closure_warning", 2.0), driving=False).condition is Condition.NONE


def test_distracted_after_two_seconds_off_the_road_while_moving() -> None:
    result = assess(eyes_off_seconds=2.5)
    assert result.condition is Condition.DISTRACTED
    assert result.level is RiskLevel.ELEVATED
    assert assess(eyes_off_seconds=1.9).condition is Condition.NONE
    assert assess(eyes_off_seconds=5.0, driving=False).condition is Condition.NONE


# ---- ported Wave B behaviour: staleness, faults, coverage, hysteresis -------------------------------------
def test_the_model_and_eye_closure_are_not_added_together() -> None:
    model_only = assess(probability=1.0)
    closure_only = assess(reading(IndicatorName.EYES, "closure_warning", 2.5))
    both = assess(reading(IndicatorName.EYES, "closure_warning", 2.5), probability=1.0)
    assert both.condition is Condition.DROWSY
    assert both.score == max(model_only.score, closure_only.score)


def test_stale_readings_and_probabilities_drop_out() -> None:
    old = reading(IndicatorName.EYES, "closure_critical", 6.0, t=T - 3.0)
    assert assess(old, HEAD_DROP, driving=False).level is not RiskLevel.HIGH
    assert assess(probability=0.9, probability_t=T - 16.0).condition is Condition.NONE
    assert assess(probability=0.9, probability_t=T - 14.0).condition is Condition.DROWSY


def test_unavailable_inputs_lower_confidence_without_reweighting_the_rest() -> None:
    full = assess(EYES_CRITICAL, HEAD_DROP, driving=False)
    degraded_coverage = {
        **ALL_AVAILABLE,
        IndicatorName.OCULAR: Coverage.UNAVAILABLE,
        IndicatorName.BREATHING: Coverage.UNAVAILABLE,
    }
    degraded = assess(EYES_CRITICAL, HEAD_DROP, driving=False, coverage=degraded_coverage)
    assert degraded.score == full.score
    assert degraded.confidence == pytest.approx(full.confidence * 3 / 5)


def test_monitoring_degraded_names_situations_but_not_the_default_mic_state() -> None:
    sunglasses = assess(situations=(Situation.SUNGLASSES_OR_EYE_OCCLUSION, Situation.MIC_BLOCKED_BY_POLICY))
    assert sunglasses.monitoring_degraded
    assert sunglasses.degraded_situations == (Situation.SUNGLASSES_OR_EYE_OCCLUSION,)
    assert not assess(situations=(Situation.MIC_BLOCKED_BY_POLICY,)).monitoring_degraded


def test_camera_fault_with_every_reading_unavailable_is_no_condition() -> None:
    readings = [missing(name, T, "face_not_tracked") for name in BEHAVIOUR_INDICATORS]
    no_coverage = dict.fromkeys(BEHAVIOUR_INDICATORS, Coverage.UNAVAILABLE)
    result = assess(*readings, coverage=no_coverage, situations=(Situation.CAMERA_FAULT,))
    assert result.condition is Condition.NONE
    assert result.score == 0.0
    assert result.confidence == 0.0
    assert result.degraded_situations == (Situation.CAMERA_FAULT,)


def test_hysteresis_holds_a_level_until_it_falls_below_the_exit() -> None:
    gate = Hysteresis()
    assert [gate.apply(score) for score in (0.44, 0.46, 0.40, 0.34)] == [
        RiskLevel.LOW,
        RiskLevel.ELEVATED,
        RiskLevel.ELEVATED,
        RiskLevel.LOW,
    ]
    assert [gate.apply(score) for score in (0.71, 0.60, 0.54, 0.30, 0.10)] == [
        RiskLevel.HIGH,
        RiskLevel.HIGH,
        RiskLevel.ELEVATED,
        RiskLevel.LOW,
        RiskLevel.NONE,
    ]
    with pytest.raises(ValueError, match="thresholds"):
        Hysteresis(FusionConfig(elevated_enter=0.8))


def test_one_conditions_high_level_is_not_inherited_by_another() -> None:
    fusion = RiskFusion()
    assert assess(eyes_off_seconds=5.0, fusion=fusion).level is RiskLevel.HIGH
    switched = assess(ERRATIC_5, NODDING, fusion=fusion, t=T + 0.1)
    assert switched.condition is Condition.POSSIBLE_IMPAIRMENT
    assert switched.level is RiskLevel.ELEVATED


# ---- learning the weights later --------------------------------------------------------------------------
def fusion_windows(people: int, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    for person in range(people):
        for _ in range(20):
            label = int(rng.random() < 0.4)
            row = {name: float(rng.random() * 0.3) for name in FUSION_FEATURES}
            row["eyes_severity"] = float(0.7 + 0.3 * rng.random()) if label else float(0.3 * rng.random())
            row["subject"] = f"P{person:03d}"
            rows.append(row)
            labels.append(label)
    return pd.DataFrame(rows), np.asarray(labels)


def test_fit_fusion_is_person_grouped_and_research_weights_cannot_ship() -> None:
    windows, labels = fusion_windows(12)
    weights = fit_fusion(windows, labels)
    locked = {subject for subject in windows["subject"] if is_locked_test_subject(subject)}
    assert weights.people == 12 - len(locked)
    assert weights.out_of_fold_auc > 0.9
    assert weights.coefficients[FUSION_FEATURES.index("eyes_severity")] > 0
    assert weights.promotion_blockers()
    probabilities = weights.probability(windows[list(FUSION_FEATURES)].to_numpy())
    assert probabilities.shape == (len(windows),)
    assert not fusion_weights_check(weights).passed
    commercial = fit_fusion(windows, labels, track=Track.COMMERCIAL, data_source=DataSource.REAL)
    assert commercial.promotion_blockers() == []
    assert fusion_weights_check(commercial).passed


def test_fit_fusion_refuses_unusable_data() -> None:
    windows, labels = fusion_windows(3)
    with pytest.raises(DataError, match="at least 5 people"):
        fit_fusion(windows, labels)
    windows, labels = fusion_windows(12)
    with pytest.raises(DataError, match="both"):
        fit_fusion(windows, np.zeros(len(labels), dtype=int))
    with pytest.raises(DataError, match="missing columns"):
        fit_fusion(windows.drop(columns=["head_severity"]), labels)
