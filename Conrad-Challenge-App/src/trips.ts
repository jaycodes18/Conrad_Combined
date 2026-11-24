import type { Cue, RiskAlert } from './risk';

export type SessionSource = { mode: 'camera'; scenario?: never } |
  { mode: 'drill' | 'ml-replay'; scenario: Cue };

export type Contact = { name: string; phone: string };
export type Role = 'driver' | 'guardian';
export type Preferences = {
  role: Role | null;
  driver: string;
  contact: Contact | null;
  gps: boolean;
  voice: boolean;
};
export const defaultPreferences: Preferences = { role: null, driver: '', contact: null, gps: false, voice: true };

export type Delivery = {
  state: 'pending' | 'delivered' | 'failed' | 'unknown' | 'local';
  elapsedMs?: number;
};

export type TripEvent = RiskAlert & {
  at: string;
  elapsedMs: number;
  processingMs: number;
};

type CameraTiming = {
  firstUsableFrameMs: number | null;
  longestFrameMs: number | null;
  droppedSlowFrames: number;
};

export type Trip = {
  id: string;
  startedAt: string;
  durationMs: number;
  mode: SessionSource['mode'];
  scenario?: Cue;
  end: 'finished' | 'background' | 'fault';
  samples: number;
  trackingSamples: number;
  noFaceSamples: number;
  maximumGapMs: number;
  cameraTiming?: CameraTiming;
  maxSpeedKmh: number | null;
  events: TripEvent[];
  delivery?: Delivery;
};

export function sessionLabel(mode: Trip['mode']) {
  switch (mode) {
    case 'camera': return 'Camera session';
    case 'drill': return 'Parked drill';
    case 'ml-replay': return 'Driven ML replay';
  }
}

export function formatDuration(ms: number) {
  const total = Math.floor(ms / 1000);
  return Math.floor(total / 60) + 'm ' + (total % 60) + 's';
}
