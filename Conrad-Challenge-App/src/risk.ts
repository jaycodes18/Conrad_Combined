export type Cue = 'eye-closure' | 'head-away';
export type Level = 1 | 2 | 3;
export type Observation = {
  at: number;
  face: { eyesClosed: number; yaw: number; pitch: number } | null;
};

export type RiskAlert = { cue: Cue; level: Level; evidenceMs: number };
export type Coverage = 'calibrating' | 'no-face' | 'tracking';

export const MAX_OBSERVATION_GAP_MS = 900;
const CALIBRATION_MS = 2000;
const ALERT_REARM_MS = 1500;
const REPEAT_WINDOW_MS = 120_000;
const WARNING_AFTER: Record<Cue, number> = { 'eye-closure': 2100, 'head-away': 2600 };

type Neutral = { yaw: number; pitch: number };

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
  private readonly emergencyDrill: boolean;

  constructor(emergencyDrill = false) {
    this.emergencyDrill = emergencyDrill;
  }

  interruptTracking() {
    this.calibration = undefined;
    this.onset = {};
    this.clearSince = undefined;
  }

  update({ at, face }: Observation): { coverage: Coverage; alerts: RiskAlert[] } {
    const alerts: RiskAlert[] = [];
    if (!Number.isFinite(at) || (this.lastAt !== undefined && at <= this.lastAt)) {
      return { coverage: this.neutral ? 'tracking' : 'calibrating', alerts };
    }

    if (this.lastAt !== undefined && at - this.lastAt > MAX_OBSERVATION_GAP_MS) this.interruptTracking();
    this.lastAt = at;
    if (!face || ![face.eyesClosed, face.yaw, face.pitch].every(Number.isFinite)) {
      this.interruptTracking();
      return { coverage: 'no-face', alerts };
    }

    if (!this.neutral) {
      this.calibrate(at, face);
      return { coverage: this.neutral ? 'tracking' : 'calibrating', alerts };
    }

    const cues: [Cue, boolean][] = [
      ['eye-closure', face.eyesClosed >= 0.57],
      ['head-away', Math.abs(face.yaw - this.neutral.yaw) >= 25 ||
        Math.abs(face.pitch - this.neutral.pitch) >= 20],
    ];

    if (cues.every(([, active]) => !active)) {
      this.clearSince ??= at;
      if (at - this.clearSince >= ALERT_REARM_MS) this.warned = false;
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
        alerts.push({ cue, level: 1, evidenceMs });
      }
      if (!this.sentContact && (evidenceMs >= 8000 || (newWarning && repeated))) {
        this.sentContact = true;
        alerts.push({ cue, level: 2, evidenceMs });
      }
      if (this.emergencyDrill && cue === 'eye-closure' && !this.sentDrill && evidenceMs >= 18_000) {
        this.sentDrill = true;
        alerts.push({ cue, level: 3, evidenceMs });
      }
    }
    return { coverage: 'tracking', alerts };
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
