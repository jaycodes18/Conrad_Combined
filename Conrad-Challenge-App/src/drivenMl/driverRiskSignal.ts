// DriverRiskSignal v1: the message the Driven ML side emits once per camera frame.
// Mirrors ml/contracts/driver_risk_signal.v1.schema.json and driven_ml/risk/signal.py; ml/tests/test_contract.py
// checks that this file names every field and enum value the schema does. Field names are snake_case because the
// app's Supabase tables already are (guardian_test_alerts: session_id, cue, level, evidence_ms).
// The signal never carries pixels, landmarks, audio, transcripts or GPS coordinates.

export type DriverRiskCondition =
  | 'none'
  | 'distracted'
  | 'drowsy'
  | 'possible_impairment'
  | 'possible_unresponsiveness';

export type DriverRiskLevel = 'none' | 'low' | 'elevated' | 'high';

export type IndicatorName = 'eyes' | 'ocular' | 'head' | 'breathing' | 'responsiveness' | 'sensor_quality';

export type BehaviourIndicatorName = Exclude<IndicatorName, 'sensor_quality'>;

export type IndicatorStates = {
  eyes: 'open' | 'blinking' | 'closure_warning' | 'closure_critical' | 'unavailable';
  ocular: 'steady' | 'scanning' | 'erratic_5s' | 'erratic_10s' | 'unavailable';
  head: 'stable' | 'nodding' | 'head_drop' | 'slumped' | 'unavailable';
  breathing: 'regular' | 'irregular' | 'agonal_pattern_suspected' | 'not_detected_quiet_cabin' | 'unavailable';
  responsiveness: 'idle' | 'waiting' | 'responded' | 'no_response' | 'unavailable';
  sensor_quality: 'ok' | 'degraded' | 'fault';
};

export type IndicatorSnapshot<S extends string> = {
  available: boolean;
  state: S;
  /** Units per indicator: eyes and head seconds held, ocular instability (z), breathing breaths/min,
   *  responsiveness latency or elapsed seconds, sensor_quality the share of indicators available. */
  value: number | null;
  severity: number; // 0..1; always 0 when unavailable
  confidence: number; // 0..1
  evidence: string[]; // plain sentences a parent could read
  unavailable_reason: string | null;
};

export type CoverageState = 'available' | 'degraded' | 'unavailable';

export type Situation =
  | 'sunglasses_or_eye_occlusion'
  | 'clear_glasses_glare'
  | 'low_light'
  | 'ir_mode'
  | 'backlit'
  | 'face_out_of_frame'
  | 'camera_fault'
  | 'mic_fault'
  | 'cabin_too_noisy'
  | 'image_blur'
  | 'fisheye_distortion'
  | 'lower_face_occluded';

export type ResponsivenessCheck = 'idle' | 'waiting' | 'responded' | 'no_response' | 'unavailable';

export type Escalation = {
  tier: 0 | 1 | 2 | 3;
  /** true on exactly one signal per episode: speak the in-car alert */
  in_car_alert: boolean;
  responsiveness_check: ResponsivenessCheck;
  /** true on exactly one signal per episode: text the consented contacts */
  notify_contacts: boolean;
  /** 'precise': attach the phone's current location, read under the app's own permission flow */
  share_location: 'none' | 'precise';
  /** true on exactly one signal per episode; only when a contact's consent covers automated calls */
  voice_call_recommended: boolean;
  emergency_services: 'never_automated';
  reason: string;
};

export type ModelInfo = {
  feature_spec_version: number;
  artifact_hash: string | null;
  track: 'research' | 'commercial';
  experimental: IndicatorName[];
};

/** The app's own alert fields, unchanged: src/risk.ts RiskAlert (cue, level, evidenceMs) and guardian_test_alerts. */
export type AppAlert = {
  cue: 'eye-closure' | 'head-away' | null;
  level: 1 | 2 | 3 | null;
  evidence_ms: number | null;
};

export type DriverRiskSignal = {
  schema_version: 'driver_risk_signal.v1';
  session_id: string;
  driver_ref: string; // pseudonymous, never a name or an email
  t_seconds: number;
  emitted_at: string; // ISO 8601 UTC, ending in Z
  condition: DriverRiskCondition;
  level: DriverRiskLevel;
  score: number;
  confidence: number;
  indicators: { [K in IndicatorName]: IndicatorSnapshot<IndicatorStates[K]> };
  coverage: Record<BehaviourIndicatorName, CoverageState>;
  monitoring_degraded: { active: boolean; situations: Situation[] };
  escalation: Escalation;
  model: ModelInfo;
  app_alert: AppAlert;
  disclaimer: 'Driven reports observable driving-risk indicators. It does not measure alcohol or identify any medical condition.';
};
