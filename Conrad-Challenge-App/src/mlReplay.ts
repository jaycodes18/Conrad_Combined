import type { DriverRiskSignal } from './drivenMl/driverRiskSignal';
import type { RiskAlert } from './risk';

export function mlReplayWarning(signal: DriverRiskSignal): RiskAlert | null {
  const { cue, level, evidence_ms } = signal.app_alert;
  if (signal.schema_version !== 'driver_risk_signal.v1' || !signal.escalation.in_car_alert ||
    level !== 1 || (cue !== 'eye-closure' && cue !== 'head-away') ||
    evidence_ms === null || !Number.isFinite(evidence_ms) || evidence_ms < 0) {
    return null;
  }
  return { cue, level, evidenceMs: evidence_ms };
}
