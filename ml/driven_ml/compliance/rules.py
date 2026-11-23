"""The team's legal findings, written as data the compliance gate can check.

These are engineering guardrails derived from the team's legal review, not legal advice. A lawyer should
review them before launch.

Each ``LegalRule`` with a ``flag`` is enforced against product.json. Rules without a flag (L8-L11) are
enforced elsewhere: L8 by the claim scanner, L9 and L10 by the dataset registry and promotion checks, and L11
by the recording protocol and consent file.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LegalRule:
    rule_id: str
    description: str
    legal_basis: str
    flag: str | None = None
    required_value: bool | None = None


@dataclass(frozen=True)
class GatedSetting:
    """A setting whose risky values need a written legal opinion, recorded as ``opinion_flag``."""

    rule_id: str
    setting: str
    allowed_without_opinion: frozenset[str | bool]
    opinion_flag: str
    description: str
    legal_basis: str


@dataclass(frozen=True)
class ForbiddenCodePattern:
    rule_id: str
    pattern: str
    meaning: str


LEGAL_RULES: tuple[LegalRule, ...] = (
    LegalRule(
        "L1",
        "No automated, artificial or prerecorded voice calls to 911 or any emergency line",
        "47 CFR 64.1200(a)(1)(i)",
        "auto_call_911",
        False,
    ),
    LegalRule(
        "L2",
        "Automated texts only to contacts with recorded consent",
        "47 CFR 64.1200 (prior express consent)",
        "require_contact_consent",
        True,
    ),
    LegalRule(
        "L3",
        "Nothing for the driver to tap, read or say while moving; alerts are audio and clear by themselves",
        "Tex. Transp. Code 545.4251; 545.424",
        "driver_input_while_moving",
        False,
    ),
    LegalRule(
        "L4",
        "No cabin audio recording or voice recognition (voiceprints are biometric identifiers)",
        "Tex. Bus. & Com. Code 503.001",
        "record_audio",
        False,
    ),
    LegalRule(
        "L5",
        "Notice and consent from the driver, and a guardian for minors, before measuring a face",
        "Tex. Bus. & Com. Code 503.001",
        "require_biometric_consent",
        True,
    ),
    LegalRule(
        "L6",
        "Face data and personal baselines never leave the device",
        "Tex. Bus. & Com. Code 503.001; TDPSA",
        "upload_face_data",
        False,
    ),
    LegalRule(
        "L7",
        "Never sell or share personal or biometric data; partners see opt-in aggregates only",
        "TDPSA (Tex. Bus. & Com. Code ch. 541); 503.001",
        "sell_personal_data",
        False,
    ),
    LegalRule(
        "L8",
        "No unsubstantiated claims ('detects drunk driving', 'measures BAC', 'prevents crashes')",
        "FTC Act section 5",
    ),
    LegalRule(
        "L9",
        "Production models train only on commercially licensed data or the team's consented recordings",
        "Dataset licenses; copyright",
    ),
    LegalRule(
        "L10",
        "No scraping YouTube, social media or stock sites for training data",
        "Platform terms of service",
    ),
    LegalRule(
        "L11",
        "Nobody drives drowsy or impaired for data; acted cues only, in a parked car",
        "Tex. Penal Code 49.04; Tex. Alco. Bev. Code ch. 106",
    ),
)

GATED_SETTINGS: tuple[GatedSetting, ...] = (
    GatedSetting(
        rule_id="G1",
        setting="tx_under18_device",
        allowed_without_opinion=frozenset({"no_sim_device", "partner_dashcam"}),
        opinion_flag="legal_opinion_tx_under18",
        description="Texas drivers under 18: no wireless device in use while driving, hands-free included",
        legal_basis="Tex. Transp. Code 545.424",
    ),
    GatedSetting(
        rule_id="G2",
        setting="studio_research_audio",
        allowed_without_opinion=frozenset({False}),
        opinion_flag="legal_opinion_studio_audio",
        description=(
            "Studio-only ephemeral microphone analysis for research breathing estimates; "
            "never persists raw audio (record_audio stays false)"
        ),
        legal_basis="Tex. Bus. & Com. Code 503.001; voiceprints are biometric identifiers",
    ),
)

BANNED_CLAIM_PATTERNS: tuple[str, ...] = (
    r"detect\w*\s+drunk",
    r"\bBAC\b",
    r"blood[-\s]alcohol\s+(level|concentration)",
    r"prevent\w*\s+(crash|accident)",
    r"guarantee",
    r"diagnos\w*",
    r"breathalyz",
)

FORBIDDEN_CODE_PATTERNS: tuple[ForbiddenCodePattern, ...] = (
    ForbiddenCodePattern(
        "L4",
        r"SpeechRecognition|MediaRecorder|audio\s*:\s*true",
        "microphone or recording API",
    ),
    ForbiddenCodePattern(
        "L6",
        r"toDataURL|toBlob\s*\(|captureStream|getImageData",
        "camera pixels copied out of the video element",
    ),
    ForbiddenCodePattern(
        "L1",
        r"""/Calls\.json|tel:\s*911|['"]\+?1?911['"]""",
        "automated call or call to 911",
    ),
    ForbiddenCodePattern(
        "L7",
        r"\b(segment\.io|mixpanel|facebook\.net|doubleclick|googletagmanager)\b",
        "third-party tracking or advertising SDK",
    ),
)

# The app and backend live beside this pipeline; files that do not exist are skipped.
USER_FACING_TEXT_FILES: tuple[str, ...] = (
    "apps/web/index.html",
    "apps/web/app.html",
    "apps/web/parent.html",
    "apps/web/privacy.html",
    "apps/web/src/content/strings.json",
    "supabase/functions/_shared/domain/messages.ts",
)
SOURCE_GLOBS: tuple[str, ...] = ("apps/web/src/**/*.ts", "supabase/functions/**/*.ts")
BACKEND_PRODUCT_COPY = "supabase/functions/_shared/product.json"
