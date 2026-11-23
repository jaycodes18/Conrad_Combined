"""Risk fusion: the latest indicator readings, the 60 s drowsiness probability and motion state, to one condition.

The conditions are observable ones, worded so they claim no more than the data supports: ``distracted``,
``drowsy``, ``possible_impairment`` and ``possible_unresponsiveness``. Nothing here measures alcohol or identifies
a medical condition (FTC Act section 5; AGENTS.md invariant 4).

Rules, in this order (each condition gets a score and its own level; the highest level wins, and at equal level the
earlier rule wins):

1. ``possible_unresponsiveness`` is high when the eyes are ``closure_critical`` (5 s or more, face tracked) and at
   least one of: the head is ``head_drop`` or ``slumped``; responsiveness is ``no_response``; breathing is
   ``agonal_pattern_suspected`` or ``not_detected_quiet_cabin``. When the eyes are unavailable (sunglasses), it is
   instead a head drop or slump held for 5 s with ``no_response``. Breathing alone never triggers it, and regular
   breathing never vetoes it: a person asleep at the wheel still breathes. One primary sign without its confirming
   sign is ``elevated``, so the in-car alert opens the response check that can confirm it.
2. ``possible_impairment`` needs the vehicle moving and at least two independent families abnormal at once,
   relative to the driver's baseline: ocular (``erratic_5s`` elevated, ``erratic_10s`` high), head nodding,
   drowsiness probability at the model's threshold, and repeated ``closure_warning`` episodes within 60 s. Three
   families at once is also high. One family alone gives at most ``drowsy`` or ``distracted``.
3. ``drowsy``: the drowsiness model and closure warnings, while moving.
4. ``distracted``: eyes off the road for more than 2 s while moving (NHTSA visual-manual guidelines,
   78 FR 24818, 2013).

Unavailable or stale readings drop out. The others are not re-weighted upward to compensate; ``confidence``
falls in proportion to the lost coverage, and ``monitoring_degraded`` names the situations. Levels use the Wave B
hysteresis (elevated at 0.45, back below 0.35; high at 0.70, back below 0.55), kept per condition so one
condition's high level is never inherited by another.

The scores below are hand-set constants because no labelled Driven sessions exist to learn them from.
``fit_fusion`` fits a person-grouped logistic regression so real sessions can replace them; fitted weights keep
the track rule, and research-track weights never pass the staging gate.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..errors import DataError
from ..indicators import BEHAVIOUR_INDICATORS, Coverage, IndicatorName, IndicatorReading, Situation
from ..indicators.breathing import SUSPECTED_STATES
from ..indicators.eyes import CLOSED_STATES, CLOSURE_CRITICAL, CLOSURE_WARNING
from ..indicators.head import LOST_CONTROL_STATES, NODDING
from ..indicators.ocular import ERRATIC_10S, ERRATIC_STATES
from ..indicators.responsiveness import NO_RESPONSE
from ..settings import DataSource, Track
from ..spec import FEATURE_SPEC, INDICATOR_SPEC, FusionConfig
from ..splits import is_locked_test_subject

logger = logging.getLogger(__name__)

# ---- fusion weights the team cannot learn yet ------------------------------------------------------------
# Every number is a score on the 0..1 scale the hysteresis reads: 0.45 enters elevated, 0.70 enters high.
UNRESPONSIVE_HIGH_SCORE = 0.90  # rule 1 met: a primary sign plus an independent confirming sign
UNRESPONSIVE_PRECURSOR_SCORE = 0.50  # one primary sign only: elevated, enough for Tier 1 to open the response check
IMPAIRMENT_HIGH_SCORE = 0.85  # two families with ocular erratic for 10 s, or three families
IMPAIRMENT_ELEVATED_SCORE = 0.55  # two independent families abnormal together
DROWSY_CLOSURE_WARNING_SCORE = 0.55  # eyes shut 2 s or more: an in-car alert
DROWSY_CLOSURE_CRITICAL_SCORE = 0.75  # eyes shut 5 s or more: high on its own
DROWSY_REPEATED_WARNINGS_SCORE = 0.65  # two 2 s closures in 60 s: elevated, not high
DROWSY_NODDING_SCORE = 0.50  # nodding alone: an alert, never impairment
DROWSY_PROBABILITY_FLOOR = 0.45  # the model at its own threshold maps to elevated ...
DROWSY_PROBABILITY_SPAN = 0.40  # ... and rises linearly to 0.85 at probability 1
DISTRACTED_SCORE = 0.60  # eyes off the road past 2 s
DISTRACTED_PER_SECOND = 0.05  # each further second away adds this ...
DISTRACTED_MAX_SCORE = 0.80  # ... up to this
OCULAR_ALONE_SCORE = 0.30  # erratic gaze with nothing else: low, shown but never alerted
NONE_BELOW = 0.15  # under this, no condition is reported
FUSION_FEATURES: tuple[str, ...] = (
    "eyes_severity",
    "ocular_severity",
    "head_severity",
    "breathing_severity",
    "responsiveness_severity",
    "drowsiness_probability",
    "closure_warnings_60s",
)
FIT_FOLDS = 5
EYES_OFF_SECONDS = FEATURE_SPEC.live_alerts.eyes_off_seconds  # the NHTSA 2 s rule, from one place


class Condition(StrEnum):
    NONE = "none"
    DISTRACTED = "distracted"
    DROWSY = "drowsy"
    POSSIBLE_IMPAIRMENT = "possible_impairment"
    POSSIBLE_UNRESPONSIVENESS = "possible_unresponsiveness"


RULE_ORDER: tuple[Condition, ...] = (
    Condition.POSSIBLE_UNRESPONSIVENESS,
    Condition.POSSIBLE_IMPAIRMENT,
    Condition.DROWSY,
    Condition.DISTRACTED,
)


class RiskLevel(StrEnum):
    NONE = "none"
    LOW = "low"
    ELEVATED = "elevated"
    HIGH = "high"


LEVEL_ORDER: dict[RiskLevel, int] = {RiskLevel.NONE: 0, RiskLevel.LOW: 1, RiskLevel.ELEVATED: 2, RiskLevel.HIGH: 3}


@dataclass(frozen=True)
class FusionInputs:
    t_seconds: float
    readings: Mapping[IndicatorName, IndicatorReading]
    driving: bool
    coverage: Mapping[IndicatorName, Coverage]
    situations: Sequence[Situation] = ()
    drowsiness_probability: float | None = None
    probability_t_seconds: float | None = None
    probability_threshold: float | None = None
    eyes_off_seconds: float = 0.0


@dataclass(frozen=True)
class ConditionScore:
    condition: Condition
    score: float
    reasons: tuple[str, ...]
    sources: tuple[IndicatorName, ...]


@dataclass(frozen=True)
class FusionResult:
    t_seconds: float
    condition: Condition
    level: RiskLevel
    score: float
    confidence: float
    reasons: tuple[str, ...]
    scores: Mapping[Condition, float]
    monitoring_degraded: bool
    degraded_situations: tuple[Situation, ...]
    readings: Mapping[IndicatorName, IndicatorReading]


@dataclass
class Hysteresis:
    """Ported from the Wave B ``RiskFusionEngine``: levels do not flicker when a score sits near a threshold."""

    config: FusionConfig = INDICATOR_SPEC.fusion
    level: RiskLevel = RiskLevel.NONE

    def __post_init__(self) -> None:
        config = self.config
        if not 0 < config.elevated_exit < config.elevated_enter < config.high_enter <= 1:
            raise ValueError("thresholds must satisfy 0 < elevated_exit < elevated_enter < high_enter <= 1")
        if not config.elevated_exit < config.high_exit < config.high_enter:
            raise ValueError("high_exit must be between elevated_exit and high_enter")

    def apply(self, score: float) -> RiskLevel:
        config, current = self.config, self.level
        if current is RiskLevel.HIGH:
            if score < config.high_exit:
                current = RiskLevel.ELEVATED if score >= config.elevated_exit else RiskLevel.NONE
        elif current is RiskLevel.ELEVATED:
            if score >= config.high_enter:
                current = RiskLevel.HIGH
            elif score < config.elevated_exit:
                current = RiskLevel.NONE
        elif score >= config.high_enter:
            current = RiskLevel.HIGH
        elif score >= config.elevated_enter:
            current = RiskLevel.ELEVATED
        else:
            current = RiskLevel.NONE
        self.level = current
        if current is RiskLevel.NONE and score >= NONE_BELOW:
            return RiskLevel.LOW
        return current


@dataclass
class RiskFusion:
    """One per driving session. ``assess`` once per frame, in time order."""

    config: FusionConfig = INDICATOR_SPEC.fusion
    hysteresis: dict[Condition, Hysteresis] = field(
        default_factory=lambda: {condition: Hysteresis() for condition in RULE_ORDER}
    )
    _warning_onsets: deque[float] = field(default_factory=deque)
    _eyes_closed_last: bool = False
    _head_lost_since: float | None = None
    last: FusionResult | None = None

    def reset(self) -> None:
        self.hysteresis = {condition: Hysteresis(self.config) for condition in RULE_ORDER}
        self._warning_onsets.clear()
        self._eyes_closed_last = False
        self._head_lost_since = None
        self.last = None

    def assess(self, inputs: FusionInputs) -> FusionResult:
        t = inputs.t_seconds
        readings = self._current(inputs)
        self._track_history(t, readings)
        candidates = (
            self._unresponsiveness(t, readings),
            self._impairment(inputs, readings),
            self._drowsy(inputs, readings),
            self._distracted(inputs, readings),
        )
        levels = {item.condition: self.hysteresis[item.condition].apply(item.score) for item in candidates}
        best = max(
            candidates,
            key=lambda item: (LEVEL_ORDER[levels[item.condition]], -RULE_ORDER.index(item.condition), item.score),
        )
        level = levels[best.condition]
        condition = best.condition if best.score >= NONE_BELOW or level is not RiskLevel.NONE else Condition.NONE
        degraded = tuple(
            situation for situation in inputs.situations if situation is not Situation.MIC_BLOCKED_BY_POLICY
        )
        result = FusionResult(
            t_seconds=t,
            condition=condition,
            level=level if condition is not Condition.NONE else RiskLevel.NONE,
            score=best.score if condition is not Condition.NONE else 0.0,
            confidence=self._confidence(best, readings, inputs.coverage),
            reasons=best.reasons if condition is not Condition.NONE else (),
            scores={item.condition: item.score for item in candidates},
            monitoring_degraded=bool(degraded),
            degraded_situations=degraded,
            readings=readings,
        )
        self.last = result
        return result

    # ---- inputs ----------------------------------------------------------------------------------------
    def _current(self, inputs: FusionInputs) -> dict[IndicatorName, IndicatorReading]:
        """Readings that are available and fresh; stale ones drop out like unavailable ones."""
        limit = self.config.max_reading_age_seconds
        return {
            name: reading
            for name, reading in inputs.readings.items()
            if reading.available and -0.25 <= inputs.t_seconds - reading.t_seconds <= limit
        }

    def _probability(self, inputs: FusionInputs) -> float | None:
        if inputs.drowsiness_probability is None or inputs.probability_t_seconds is None:
            return None
        age = inputs.t_seconds - inputs.probability_t_seconds
        if not -0.25 <= age <= self.config.probability_max_age_seconds:
            return None
        return inputs.drowsiness_probability

    def _track_history(self, t: float, readings: Mapping[IndicatorName, IndicatorReading]) -> None:
        eyes = readings.get(IndicatorName.EYES)
        closed = eyes is not None and eyes.state in CLOSED_STATES
        if closed and not self._eyes_closed_last:
            self._warning_onsets.append(t)
        self._eyes_closed_last = closed
        while self._warning_onsets and self._warning_onsets[0] < t - self.config.repeated_closure_window_seconds:
            self._warning_onsets.popleft()
        head = readings.get(IndicatorName.HEAD)
        if head is not None and head.state in LOST_CONTROL_STATES:
            if self._head_lost_since is None:
                self._head_lost_since = t - (head.value or 0.0)
        else:
            self._head_lost_since = None

    @property
    def closure_warnings_last_minute(self) -> int:
        return len(self._warning_onsets)

    # ---- rules -----------------------------------------------------------------------------------------
    def _unresponsiveness(self, t: float, readings: Mapping[IndicatorName, IndicatorReading]) -> ConditionScore:
        eyes = readings.get(IndicatorName.EYES)
        head = readings.get(IndicatorName.HEAD)
        response = readings.get(IndicatorName.RESPONSIVENESS)
        breathing = readings.get(IndicatorName.BREATHING)
        eyes_critical = eyes is not None and eyes.state == CLOSURE_CRITICAL
        head_lost = head is not None and head.state in LOST_CONTROL_STATES
        head_held = 0.0 if self._head_lost_since is None else t - self._head_lost_since
        no_response = response is not None and response.state == NO_RESPONSE
        breathing_flag = breathing is not None and breathing.state in SUSPECTED_STATES
        confirming = [
            reading
            for reading, flagged in ((head, head_lost), (response, no_response), (breathing, breathing_flag))
            if flagged and reading is not None
        ]
        if eyes_critical and eyes is not None and confirming:
            return _score(Condition.POSSIBLE_UNRESPONSIVENESS, UNRESPONSIVE_HIGH_SCORE, [eyes, *confirming])
        held_long = head_held >= self.config.unresponsive_head_hold_seconds
        if eyes is None and head_lost and no_response and held_long and head is not None and response is not None:
            return _score(Condition.POSSIBLE_UNRESPONSIVENESS, UNRESPONSIVE_HIGH_SCORE, [head, response])
        primary = [reading for reading, flagged in ((eyes, eyes_critical), (head, head_lost)) if flagged and reading]
        if primary:
            return _score(Condition.POSSIBLE_UNRESPONSIVENESS, UNRESPONSIVE_PRECURSOR_SCORE, primary)
        return _score(Condition.POSSIBLE_UNRESPONSIVENESS, 0.0, [])

    def _impairment(self, inputs: FusionInputs, readings: Mapping[IndicatorName, IndicatorReading]) -> ConditionScore:
        if not inputs.driving:
            return _score(Condition.POSSIBLE_IMPAIRMENT, 0.0, [])
        ocular = readings.get(IndicatorName.OCULAR)
        head = readings.get(IndicatorName.HEAD)
        eyes = readings.get(IndicatorName.EYES)
        probability = self._probability(inputs)
        families: list[IndicatorReading] = []
        extra: list[str] = []
        ocular_long = False
        if ocular is not None and ocular.state in ERRATIC_STATES:
            families.append(ocular)
            ocular_long = ocular.state == ERRATIC_10S
        if head is not None and head.state == NODDING:
            families.append(head)
        model_abnormal = (
            probability is not None
            and inputs.probability_threshold is not None
            and probability >= inputs.probability_threshold
        )
        family_count = len(families) + int(model_abnormal)
        if model_abnormal and probability is not None:
            extra.append(f"The 60 s drowsiness model reads {probability:.0%}, at or above its alert threshold.")
        if self.closure_warnings_last_minute >= self.config.repeated_closure_warnings:
            family_count += 1
            extra.append(f"{self.closure_warnings_last_minute} eye closures of 2 s or more within 60 s.")
            if eyes is not None:
                families.append(eyes)
        if family_count < 2:
            return _score(Condition.POSSIBLE_IMPAIRMENT, 0.0, [])
        high = ocular_long or family_count >= 3
        reasons = [f"{family_count} independent indicator families are abnormal at once, against baseline."]
        return _score(
            Condition.POSSIBLE_IMPAIRMENT,
            IMPAIRMENT_HIGH_SCORE if high else IMPAIRMENT_ELEVATED_SCORE,
            families,
            [*reasons, *extra],
        )

    def _drowsy(self, inputs: FusionInputs, readings: Mapping[IndicatorName, IndicatorReading]) -> ConditionScore:
        if not inputs.driving:
            return _score(Condition.DROWSY, 0.0, [])
        best = _score(Condition.DROWSY, 0.0, [])
        probability = self._probability(inputs)
        threshold = inputs.probability_threshold
        if probability is not None and threshold is not None and probability >= threshold:
            span = (probability - threshold) / max(1.0 - threshold, 1e-6)
            best = _score(
                Condition.DROWSY,
                DROWSY_PROBABILITY_FLOOR + DROWSY_PROBABILITY_SPAN * min(1.0, span),
                [],
                [f"The 60 s drowsiness model reads {probability:.0%}, at or above its alert threshold."],
            )
        eyes = readings.get(IndicatorName.EYES)
        head = readings.get(IndicatorName.HEAD)
        options = [best]
        if eyes is not None and eyes.state == CLOSURE_CRITICAL:
            options.append(_score(Condition.DROWSY, DROWSY_CLOSURE_CRITICAL_SCORE, [eyes]))
        elif eyes is not None and eyes.state == CLOSURE_WARNING:
            options.append(_score(Condition.DROWSY, DROWSY_CLOSURE_WARNING_SCORE, [eyes]))
        if self.closure_warnings_last_minute >= self.config.repeated_closure_warnings:
            options.append(
                _score(
                    Condition.DROWSY,
                    DROWSY_REPEATED_WARNINGS_SCORE,
                    [],
                    [f"{self.closure_warnings_last_minute} eye closures of 2 s or more within 60 s."],
                )
            )
        if head is not None and head.state == NODDING:
            options.append(_score(Condition.DROWSY, DROWSY_NODDING_SCORE, [head]))
        return max(options, key=lambda item: item.score)

    def _distracted(self, inputs: FusionInputs, readings: Mapping[IndicatorName, IndicatorReading]) -> ConditionScore:
        if not inputs.driving:
            return _score(Condition.DISTRACTED, 0.0, [])
        rule = EYES_OFF_SECONDS
        if inputs.eyes_off_seconds > rule:
            score = min(
                DISTRACTED_MAX_SCORE, DISTRACTED_SCORE + DISTRACTED_PER_SECOND * (inputs.eyes_off_seconds - rule)
            )
            return _score(
                Condition.DISTRACTED,
                score,
                [],
                [f"Eyes off the road for {inputs.eyes_off_seconds:.1f} s while moving (2 s guideline)."],
            )
        ocular = readings.get(IndicatorName.OCULAR)
        if ocular is not None and ocular.state in ERRATIC_STATES:
            return _score(Condition.DISTRACTED, OCULAR_ALONE_SCORE, [ocular])
        return _score(Condition.DISTRACTED, 0.0, [])

    # ---- confidence ------------------------------------------------------------------------------------
    @staticmethod
    def _confidence(
        best: ConditionScore,
        readings: Mapping[IndicatorName, IndicatorReading],
        coverage: Mapping[IndicatorName, Coverage],
    ) -> float:
        share = sum(
            1.0 if coverage.get(name) is Coverage.AVAILABLE else 0.5 if coverage.get(name) is Coverage.DEGRADED else 0.0
            for name in BEHAVIOUR_INDICATORS
        ) / len(BEHAVIOUR_INDICATORS)
        used = [readings[name].confidence for name in best.sources if name in readings]
        base = float(np.mean(used)) if used else 0.8
        return min(1.0, max(0.0, base * share))


def _score(
    condition: Condition,
    score: float,
    sources: Sequence[IndicatorReading],
    reasons: Sequence[str] = (),
) -> ConditionScore:
    evidence = [*reasons, *(line for reading in sources for line in reading.evidence)]
    return ConditionScore(
        condition=condition,
        score=min(1.0, max(0.0, score)),
        reasons=tuple(dict.fromkeys(evidence)),
        sources=tuple(dict.fromkeys(reading.name for reading in sources)),
    )


# ---- learning the weights later --------------------------------------------------------------------------
@dataclass(frozen=True)
class FusionWeights:
    """A fitted logistic regression over ``FUSION_FEATURES``, standardized like the drowsiness head."""

    features: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    out_of_fold_auc: float
    people: int
    track: Track
    data_source: DataSource

    def probability(self, rows: npt.ArrayLike) -> np.ndarray:
        x = (np.atleast_2d(np.asarray(rows, dtype=float)) - np.asarray(self.scaler_mean)) / np.asarray(
            self.scaler_scale
        )
        return 1.0 / (1.0 + np.exp(-(x @ np.asarray(self.coefficients) + self.intercept)))

    def promotion_blockers(self) -> list[str]:
        """Why these weights cannot ship; empty only for commercial-track weights fitted on real data."""
        reasons = []
        if self.track is not Track.COMMERCIAL:
            reasons.append("research-track fusion weights never pass the staging gate (L9)")
        if self.data_source is not DataSource.REAL:
            reasons.append("fusion weights fitted on synthetic data never ship")
        return reasons

    def to_json(self) -> dict[str, Any]:
        return {
            "features": list(self.features),
            "coefficients": list(self.coefficients),
            "intercept": self.intercept,
            "scaler_mean": list(self.scaler_mean),
            "scaler_scale": list(self.scaler_scale),
            "out_of_fold_auc": self.out_of_fold_auc,
            "people": self.people,
            "track": self.track.value,
            "data_source": self.data_source.value,
        }


def fit_fusion(
    windows: pd.DataFrame,
    labels: npt.ArrayLike,
    *,
    track: Track = Track.RESEARCH,
    data_source: DataSource = DataSource.SYNTHETIC,
    seed: int = 7,
) -> FusionWeights:
    """Fit fusion weights on labelled windows, grouped by person, never touching the locked test people.

    ``windows`` needs a ``subject`` column and every name in ``FUSION_FEATURES``; ``labels`` is 1 where the window
    held a labelled risk event. Out-of-fold AUC uses ``GroupKFold`` by person, the same grouping as training.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    missing = [name for name in ("subject", *FUSION_FEATURES) if name not in windows.columns]
    if missing:
        raise DataError(f"fusion windows are missing columns: {', '.join(missing)}")
    y_all = np.asarray(labels, dtype=int)
    if len(y_all) != len(windows):
        raise DataError(f"got {len(y_all)} labels for {len(windows)} windows")
    keep = ~windows["subject"].astype(str).map(is_locked_test_subject).to_numpy(dtype=bool)
    frame = windows.loc[keep]
    y = y_all[keep]
    groups = frame["subject"].astype(str).to_numpy()
    people = len(set(groups))
    if set(np.unique(y)) != {0, 1}:
        raise DataError("fusion fitting needs both event and no-event windows outside the locked test")
    if people < FIT_FOLDS:
        raise DataError(f"person-grouped fusion fitting needs at least {FIT_FOLDS} people; found {people}")
    x = frame[list(FUSION_FEATURES)].to_numpy(dtype=float)
    x = np.where(np.isnan(x), np.nanmedian(x, axis=0), x)
    x = np.nan_to_num(x)
    logits = np.empty(len(y), dtype=float)
    for train_rows, held_out in GroupKFold(n_splits=FIT_FOLDS).split(x, y, groups):
        scaler = StandardScaler().fit(x[train_rows])
        model = LogisticRegression(max_iter=1000, random_state=seed).fit(scaler.transform(x[train_rows]), y[train_rows])
        logits[held_out] = model.decision_function(scaler.transform(x[held_out]))
    scaler = StandardScaler().fit(x)
    model = LogisticRegression(max_iter=1000, random_state=seed).fit(scaler.transform(x), y)
    auc = float(roc_auc_score(y, logits))
    logger.info("fit_fusion: %d people, out-of-fold AUC %.3f, track %s", people, auc, track.value)
    return FusionWeights(
        features=FUSION_FEATURES,
        coefficients=tuple(float(value) for value in model.coef_.ravel()),
        intercept=float(model.intercept_[0]),
        scaler_mean=tuple(float(value) for value in scaler.mean_),
        scaler_scale=tuple(float(value) for value in scaler.scale_),
        out_of_fold_auc=auc,
        people=people,
        track=track,
        data_source=data_source,
    )
