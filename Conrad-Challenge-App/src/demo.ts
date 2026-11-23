import type { Cue, Observation } from './risk';

export const DEMO_DURATION_MS = 25_000;

export function drillObservation(at: number, cue: Cue): Observation {
  const signal = at >= 4000 && at < 23_000;
  return {
    at,
    face: {
      eyesClosed: signal && cue === 'eye-closure' ? 0.95 : 0.08,
      yaw: signal && cue === 'head-away' ? 34 : 0,
      pitch: 0,
    },
  };
}
