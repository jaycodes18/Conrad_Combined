export type Cue = 'eye-closure' | 'head-away';
export type Level = 1 | 2 | 3;
export type Observation = {
  at: number;
  face: { eyesClosed: number; yaw: number; pitch: number } | null;
};

export type RiskAlert = { cue: Cue; level: Level; evidenceMs: number };
export type Coverage = 'calibrating' | 'no-face' | 'tracking';

/** Same vocabulary as the Driven ML contract (ml/contracts/driver_risk_signal.v1.ts). */
export type Responsiveness = 'idle' | 'waiting' | 'responded' | 'no_response' | 'unavailable';
export type Condition = 'none' | 'distracted' | 'drowsy' | 'possible_unresponsiveness';
export type Tier = 0 | 1 | 2 | 3;

export type RiskSnapshot = {
  eyesClosedMs: number;
  headAwayMs: number;
  responsiveness: Responsiveness;
  responseLatencyMs: number | null;
  responseRemainingMs: number | null;
  condition: Condition;
  tier: Tier;
  /** Latest face relative to calibration, for the schematic live view. Null when the face is not tracked. */
  face: { eyesClosed: boolean; yaw: number; pitch: number } | null;
};

export type RiskUpdate = { coverage: Coverage; alerts: RiskAlert[]; snapshot: RiskSnapshot };

export const MAX_OBSERVATION_GAP_MS = 900;
const CALIBRATION_MS = 2000;
const ALERT_REARM_MS = 1500;
const REPEAT_WINDOW_MS = 120_000;
const WARNING_AFTER: Record<Cue, number> = { 'eye-closure': 2100, 'head-away': 2600 };
/** Driven ML closure_critical_seconds = 5.0: five seconds of closed eyes escalates to the guardian at once. */
export const CRITICAL_CLOSURE_MS = 5000;
const ESCALATE_AFTER: Record<Cue, number> = { 'eye-closure': CRITICAL_CLOSURE_MS, 'head-away': 8000 };
/** Driven ML response_window_seconds = 5.0, response hold 0.5 s. */
export const RESPONSE_WINDOW_MS = 5000;
const RESPONSE_HOLD_MS = 500;

type Neutral = { yaw: number; pitch: number };
type Check = { state: Responsiveness; startedAt?: number; cue?: Cue; latencyMs: number | null };

export class RiskEngine {
  private lastAt?: number;
  private calibration?: { start: number; anchor: Neutral; sum: Neutral; count: number };
  private neutral?: Neutral;
  private onset: Partial<Record<Cue, number>> = {};
  private clearSince?: number;
  private warned = false;
  private lastWarning?: number;
  private sentContact = false;
  private sentDrill = false;
  private tier: Tier = 0;
  private check: Check = { state: 'idle', latencyMs: null };
  private face: RiskSnapshot['face'] = null;
  private readonly emergencyDrill: boolean;

  constructor(emergencyDrill = false) {
    this.emergencyDrill = emergencyDrill;
  }

  interruptTracking() {
    this.calibration = undefined;
    this.onset = {};
    this.clearSince = undefined;
    this.face = null;
    if (this.check.state === 'waiting') this.check = { state: 'unavailable', latencyMs: null };
  }

  snapshot(at = this.lastAt ?? 0): RiskSnapshot {
    const held = (cue: Cue) => this.onset[cue] === undefined ? 0 : Math.max(0, at - (this.onset[cue] as number));
    const eyesClosedMs = held('eye-closure');
    const headAwayMs = held('head-away');
    let condition: Condition = 'none';
    if (eyesClosedMs >= CRITICAL_CLOSURE_MS) condition = 'possible_unresponsiveness';
    else if (eyesClosedMs >= WARNING_AFTER['eye-closure']) condition = 'drowsy';
    else if (headAwayMs >= WARNING_AFTER['head-away']) condition = 'distracted';
    const remaining = this.check.state === 'waiting' && this.check.startedAt !== undefined
      ? Math.max(0, RESPONSE_WINDOW_MS - (at - this.check.startedAt)) : null;
    return {
      eyesClosedMs,
      headAwayMs,
      responsiveness: this.check.state,
      responseLatencyMs: this.check.latencyMs,
      responseRemainingMs: remaining,
      condition,
      tier: this.tier,
      face: this.face,
    };
  }

  update({ at, face }: Observation): RiskUpdate {
    const alerts: RiskAlert[] = [];
    if (!Number.isFinite(at) || (this.lastAt !== undefined && at <= this.lastAt)) {
      return { coverage: this.neutral ? 'tracking' : 'calibrating', alerts, snapshot: this.snapshot() };
    }

    if (this.lastAt !== undefined && at - this.lastAt > MAX_OBSERVATION_GAP_MS) this.interruptTracking();
    this.lastAt = at;
    if (!face || ![face.eyesClosed, face.yaw, face.pitch].every(Number.isFinite)) {
      this.interruptTracking();
      return { coverage: 'no-face', alerts, snapshot: this.snapshot(at) };
    }

    if (!this.neutral) {
      this.calibrate(at, face);
      return { coverage: this.neutral ? 'tracking' : 'calibrating', alerts, snapshot: this.snapshot(at) };
    }

    this.face = {
      eyesClosed: face.eyesClosed >= 0.57,
      yaw: face.yaw - this.neutral.yaw,
      pitch: face.pitch - this.neutral.pitch,
    };
    const cues: [Cue, boolean][] = [
      ['eye-closure', face.eyesClosed >= 0.57],
      ['head-away', Math.abs(face.yaw - this.neutral.yaw) >= 25 ||
        Math.abs(face.pitch - this.neutral.pitch) >= 20],
    ];

    if (cues.every(([, active]) => !active)) {
      this.clearSince ??= at;
      if (at - this.clearSince >= ALERT_REARM_MS) {
        this.warned = false;
        this.tier = 0;
      }
    } else {
      this.clearSince = undefined;
    }

    for (const [cue, active] of cues) {
      if (!active) {
        delete this.onset[cue];
        continue;
      }

      this.onset[cue] ??= at;
      const evidenceMs = at - this.onset[cue];
      const newWarning = !this.warned && evidenceMs >= WARNING_AFTER[cue];
      const repeated = this.lastWarning !== undefined && at - this.lastWarning <= REPEAT_WINDOW_MS;
      if (newWarning) {
        this.warned = true;
        this.lastWarning = at;
        this.raise(1);
        this.check = { state: 'waiting', startedAt: at, cue, latencyMs: null };
        alerts.push({ cue, level: 1, evidenceMs });
      }
      if (!this.sentContact && (evidenceMs >= ESCALATE_AFTER[cue] || (newWarning && repeated))) {
        this.sentContact = true;
        this.raise(2);
        alerts.push({ cue, level: 2, evidenceMs });
      }
      if (this.emergencyDrill && cue === 'eye-closure' && !this.sentDrill && evidenceMs >= 18_000) {
        this.sentDrill = true;
        this.raise(3);
        alerts.push({ cue, level: 3, evidenceMs });
      }
    }

    this.updateCheck(at, alerts);
    return { coverage: 'tracking', alerts, snapshot: this.snapshot(at) };
  }

  /** A response is visual: eyes open and head back to centre, held for half a second within the window. */
  private updateCheck(at: number, alerts: RiskAlert[]) {
    const check = this.check;
    if (check.state !== 'waiting' || check.startedAt === undefined || check.cue === undefined) return;
    if (this.clearSince !== undefined && at - this.clearSince >= RESPONSE_HOLD_MS) {
      this.check = { state: 'responded', latencyMs: Math.max(0, this.clearSince - check.startedAt) };
      return;
    }
    if (at - check.startedAt >= RESPONSE_WINDOW_MS) {
      this.check = { state: 'no_response', latencyMs: null };
      if (!this.sentContact) {
        this.sentContact = true;
        this.raise(2);
        const onset = this.onset[check.cue];
        alerts.push({ cue: check.cue, level: 2, evidenceMs: onset === undefined ? at - check.startedAt : at - onset });
      }
    }
  }

  private raise(level: Level) {
    if (level > this.tier) this.tier = level;
  }

  private calibrate(at: number, face: NonNullable<Observation['face']>) {
    if (face.eyesClosed > 0.35 || Math.abs(face.yaw) > 22 || Math.abs(face.pitch) > 26) {
      this.calibration = undefined;
      return;
    }

    const prior = this.calibration;
    if (!prior || Math.abs(face.yaw - prior.anchor.yaw) > 8 ||
        Math.abs(face.pitch - prior.anchor.pitch) > 8) {
      const anchor = { yaw: face.yaw, pitch: face.pitch };
      this.calibration = { start: at, anchor, sum: { ...anchor }, count: 1 };
      return;
    }
    prior.sum.yaw += face.yaw;
    prior.sum.pitch += face.pitch;
    prior.count += 1;
    if (at - prior.start >= CALIBRATION_MS && prior.count >= 4) {
      this.neutral = { yaw: prior.sum.yaw / prior.count, pitch: prior.sum.pitch / prior.count };
      this.calibration = undefined;
    }
  }
}

export const cueLabel: Record<Cue, string> = {
  'eye-closure': 'Sustained eye closure',
  'head-away': 'Sustained head turn or tilt',
};

export const conditionLabel: Record<Condition | 'possible_impairment', string> = {
  none: 'All clear',
  distracted: 'Eyes off the road',
  drowsy: 'Drowsiness signs',
  possible_impairment: 'Possible impairment',
  possible_unresponsiveness: 'Possible unresponsiveness',
};
