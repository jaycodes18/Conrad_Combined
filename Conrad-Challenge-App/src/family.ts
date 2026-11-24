// The parent's view of each connected teen: where the car is, what the driver-facing model sees right now,
// and recent drives. Live rows come from `guardian_list_driver_status()`; the demo family is synthetic and
// always labeled as such in the UI. Camera pixels never leave the teen's phone, so the "live view" is the
// model's reading of the face (eyes open or closed, head direction), not video.
import type { LiveCondition, LiveFace, TileTone } from './liveView';
import type { Tier } from './risk';

export type IndicatorReading = { title: string; value: string; note: string; tone: TileTone };

export type DriverStatus = {
  sessionId: string;
  active: boolean;
  condition: LiveCondition;
  tier: Tier;
  score: number;
  indicators: IndicatorReading[];
  face: LiveFace | null;
  speedKmh: number | null;
  location: { latitude: number; longitude: number; heading: number | null } | null;
  updatedAt: string;
};

export type TripSummary = {
  sessionId: string;
  startedAt: string;
  durationS: number;
  driverAlerts: number;
  guardianAlerts: number;
  worstCondition: LiveCondition;
  trackedPct: number | null;
};

export type FamilyDriver = {
  id: string;
  name: string;
  initials: string;
  color: string;
  locationSharing: boolean;
  videoSharing: boolean;
  status: DriverStatus | null;
  trips: TripSummary[];
};

export const DRIVER_COLORS = ['#5AA9FF', '#C08BFF', '#FF9F5A', '#3DE0C2'];
/** A status older than this is shown as "last seen", not live. */
export const LIVE_STALE_MS = 30_000;

export function initialsFor(name: string) {
  const words = name.replace(/@.*/, '').split(/[\s._-]+/).filter(Boolean);
  const letters = (words.length > 1 ? words[0][0] + words[1][0] : (words[0] ?? '?').slice(0, 2));
  return letters.toUpperCase();
}

export function displayName(email: string) {
  const local = email.replace(/@.*/, '').split(/[._-]+/)[0] ?? email;
  return local ? local[0].toUpperCase() + local.slice(1) : 'Driver';
}

export function isLive(status: DriverStatus | null, now = Date.now()) {
  return Boolean(status?.active && now - Date.parse(status.updatedAt) <= LIVE_STALE_MS);
}

// ---- demo family ---------------------------------------------------------------------------------------

type Route = [number, number][];

// Loops on public roads around Coppell, TX. Coordinates are approximate waypoints, not a recorded drive.
const ROUTES: Route[] = [
  [[32.9545, -97.0150], [32.9545, -96.9950], [32.9545, -96.9800], [32.9660, -96.9800], [32.9760, -96.9800],
    [32.9760, -96.9950], [32.9760, -97.0150], [32.9660, -97.0150]],
  [[32.9420, -96.9700], [32.9420, -96.9500], [32.9300, -96.9500], [32.9180, -96.9500], [32.9180, -96.9700],
    [32.9300, -96.9700]],
];

function along(route: Route, fraction: number) {
  const lengths = route.map((point, i) => {
    const next = route[(i + 1) % route.length];
    return Math.hypot(next[0] - point[0], next[1] - point[1]);
  });
  const total = lengths.reduce((a, b) => a + b, 0);
  let distance = ((fraction % 1) + 1) % 1 * total;
  for (let i = 0; i < route.length; i += 1) {
    if (distance <= lengths[i]) {
      const a = route[i];
      const b = route[(i + 1) % route.length];
      const t = lengths[i] ? distance / lengths[i] : 0;
      const heading = (Math.atan2(b[1] - a[1], b[0] - a[0]) * 180) / Math.PI;
      return { latitude: a[0] + (b[0] - a[0]) * t, longitude: a[1] + (b[1] - a[1]) * t, heading };
    }
    distance -= lengths[i];
  }
  return { latitude: route[0][0], longitude: route[0][1], heading: 0 };
}

function tile(title: string, value: string, note: string, tone: TileTone): IndicatorReading {
  return { title, value, note, tone };
}

function demoStatus(now: number, script: 'steady' | 'drowsy-episode', route: Route, lapMs: number): DriverStatus {
  const position = along(route, now / lapMs);
  // A 70 s cycle: 46 s clear, then eyes closing, a warning, a 5 s closure and recovery.
  const phase = (now / 1000) % 70;
  let condition: LiveCondition = 'none';
  let tier: Tier = 0;
  let score = 0.06;
  let eyes: LiveFace['eyes'] = 'open';
  let response = tile('Response check', 'Standby', 'Starts after a warning', 'safe');
  let eyeTile = tile('Eyes', 'Open', 'Blinks are ignored', 'safe');
  if (script === 'drowsy-episode' && phase >= 46 && phase < 60) {
    const closed = phase - 46;
    eyes = 'closed';
    if (closed < 2.1) {
      score = 0.25 + closed / 10;
      eyeTile = tile('Eyes', 'Closing', `${closed.toFixed(1)} s so far`, 'watch');
    } else if (closed < 5) {
      condition = 'drowsy'; tier = 1; score = 0.55 + closed / 20;
      eyeTile = tile('Eyes', `Closed ${closed.toFixed(1)} s`, 'Warning issued', 'alert');
      response = tile('Response check', `Waiting ${(5 - (closed - 2.1)).toFixed(1)} s`, 'Eyes open and head forward', 'critical');
    } else {
      condition = 'possible_unresponsiveness'; tier = 2; score = 0.94;
      eyeTile = tile('Eyes', `Closed ${closed.toFixed(1)} s`, '5 s critical line passed', 'critical');
      response = tile('Response check', 'No response', 'Guardian step triggered', 'critical');
    }
  } else if (script === 'drowsy-episode' && phase >= 60) {
    condition = 'none'; tier = 0; score = 0.18;
    response = tile('Response check', 'Responded', 'in 1.4 s', 'safe');
  }
  const speed = condition === 'possible_unresponsiveness' ? 38 : 54 + 8 * Math.sin(now / 9000);
  return {
    sessionId: 'demo-' + script,
    active: true,
    condition,
    tier,
    score,
    face: { eyes, yaw: script === 'steady' ? 6 * Math.sin(now / 2500) : 0, pitch: eyes === 'closed' && phase > 51 ? -12 : 0 },
    indicators: [
      eyeTile,
      tile('Eye movement', 'Off', 'Gaze tracking runs in the Driven ML model', 'off'),
      tile('Head position', 'Centered', 'Matches calibration', 'safe'),
      tile('Breathing', 'Off', 'Microphone off by design', 'off'),
      response,
      tile('Camera', 'Tracking', 'Face found', 'safe'),
    ],
    speedKmh: speed,
    location: position,
    updatedAt: new Date(now).toISOString(),
  };
}

function demoTrips(now: number, seed: number): TripSummary[] {
  const day = 24 * 3600_000;
  return [0, 1, 2, 4, 6].map((daysAgo, i) => ({
    sessionId: `demo-trip-${seed}-${i}`,
    startedAt: new Date(now - daysAgo * day - (seed + i) * 3_100_000).toISOString(),
    durationS: 900 + ((seed * 7 + i * 13) % 9) * 240,
    driverAlerts: seed === 1 && i === 1 ? 1 : (seed + i) % 3 === 0 ? 2 : (seed + i) % 4 === 1 ? 1 : 0,
    guardianAlerts: seed === 1 && i === 1 ? 1 : 0,
    worstCondition: seed === 1 && i === 1 ? 'possible_unresponsiveness'
      : (seed + i) % 3 === 0 ? 'drowsy' : (seed + i) % 4 === 1 ? 'distracted' : 'none',
    trackedPct: 91 + ((seed + i) % 8),
  }));
}

export function demoFamily(now = Date.now()): FamilyDriver[] {
  return [
    { id: 'demo-maya', name: 'Maya', initials: 'MA', color: DRIVER_COLORS[0], locationSharing: true, videoSharing: false,
      status: demoStatus(now, 'steady', ROUTES[0], 240_000), trips: demoTrips(now, 0) },
    { id: 'demo-arjun', name: 'Arjun', initials: 'AR', color: DRIVER_COLORS[1], locationSharing: true, videoSharing: false,
      status: demoStatus(now, 'drowsy-episode', ROUTES[1], 200_000), trips: demoTrips(now, 1) },
  ];
}
