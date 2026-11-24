// One view model for the live screen, whether the numbers come from the phone's own risk engine or from a
// Driven ML DriverRiskSignal replay. The six tiles use the ML contract's indicator names, so the screen shows
// the same structure the Python model emits. Indicators this phone build cannot measure are shown as off,
// with the reason, never as healthy.
import type { DriverRiskSignal, IndicatorName } from './drivenMl/driverRiskSignal';
import type { Condition, Coverage, RiskSnapshot, Tier } from './risk';

export type TileTone = 'safe' | 'watch' | 'alert' | 'critical' | 'off';
export type LiveCondition = Condition | 'possible_impairment';
export type SessionCoverage = Coverage | 'starting' | 'delayed';

export type Tile = { key: IndicatorName; title: string; value: string; note: string; tone: TileTone };

export type LiveFace = { eyes: 'open' | 'closed'; yaw: number; pitch: number };

export type LiveView = {
  condition: LiveCondition;
  /** 0..1 risk, for the meter. */
  score: number;
  face: LiveFace | null;
  tone: TileTone;
  headline: string;
  detail: string;
  tier: Tier;
  prompt: string | null;
  tiles: Tile[];
};

const TITLES: Record<IndicatorName, string> = {
  eyes: 'Eyes',
  ocular: 'Eye movement',
  head: 'Head position',
  breathing: 'Breathing',
  responsiveness: 'Response check',
  sensor_quality: 'Camera',
};

const HEADLINE: Record<LiveCondition, string> = {
  none: 'All clear',
  distracted: 'Eyes off the road',
  drowsy: 'Drowsiness signs',
  possible_impairment: 'Possible impairment',
  possible_unresponsiveness: 'Possible unresponsiveness',
};

function seconds(ms: number) {
  return (ms / 1000).toFixed(1) + ' s';
}

function conditionTone(condition: LiveCondition): TileTone {
  if (condition === 'possible_unresponsiveness') return 'critical';
  if (condition === 'possible_impairment' || condition === 'drowsy') return 'alert';
  if (condition === 'distracted') return 'watch';
  return 'safe';
}

const RESPONSE_PROMPT = 'Open your eyes and look at the road';

export function viewFromEngine(snapshot: RiskSnapshot, coverage: SessionCoverage): LiveView {
  const tracking = coverage === 'tracking';
  const offReason = coverage === 'calibrating' ? 'Calibrating' : coverage === 'starting' ? 'Starting'
    : coverage === 'delayed' ? 'Frames delayed' : 'Face not visible';
  const { eyesClosedMs, headAwayMs, responsiveness } = snapshot;

  const eyes: Tile = !tracking ? { key: 'eyes', title: TITLES.eyes, value: 'Off', note: offReason, tone: 'off' }
    : eyesClosedMs >= 5000 ? { key: 'eyes', title: TITLES.eyes, value: 'Closed ' + seconds(eyesClosedMs), note: '5 s critical line passed', tone: 'critical' }
      : eyesClosedMs >= 2100 ? { key: 'eyes', title: TITLES.eyes, value: 'Closed ' + seconds(eyesClosedMs), note: 'Warning issued', tone: 'alert' }
        : eyesClosedMs > 0 ? { key: 'eyes', title: TITLES.eyes, value: 'Closing', note: seconds(eyesClosedMs) + ' so far', tone: 'watch' }
          : { key: 'eyes', title: TITLES.eyes, value: 'Open', note: 'Blinks are ignored', tone: 'safe' };

  const head: Tile = !tracking ? { key: 'head', title: TITLES.head, value: 'Off', note: offReason, tone: 'off' }
    : headAwayMs >= 2600 ? { key: 'head', title: TITLES.head, value: 'Away ' + seconds(headAwayMs), note: 'Turned or tilted past calibration', tone: 'alert' }
      : headAwayMs > 0 ? { key: 'head', title: TITLES.head, value: 'Turning', note: seconds(headAwayMs) + ' so far', tone: 'watch' }
        : { key: 'head', title: TITLES.head, value: 'Centered', note: 'Matches calibration', tone: 'safe' };

  const response: Tile = responsiveness === 'waiting'
    ? { key: 'responsiveness', title: TITLES.responsiveness, value: 'Waiting ' + seconds(snapshot.responseRemainingMs ?? 0), note: 'Eyes open and head forward', tone: 'critical' }
    : responsiveness === 'responded'
      ? { key: 'responsiveness', title: TITLES.responsiveness, value: 'Responded', note: 'in ' + seconds(snapshot.responseLatencyMs ?? 0), tone: 'safe' }
      : responsiveness === 'no_response'
        ? { key: 'responsiveness', title: TITLES.responsiveness, value: 'No response', note: 'Guardian step triggered', tone: 'critical' }
        : responsiveness === 'unavailable'
          ? { key: 'responsiveness', title: TITLES.responsiveness, value: 'Off', note: 'Face lost during the check', tone: 'off' }
          : { key: 'responsiveness', title: TITLES.responsiveness, value: 'Standby', note: 'Starts after a warning', tone: 'safe' };

  const camera: Tile = tracking
    ? { key: 'sensor_quality', title: TITLES.sensor_quality, value: 'Tracking', note: 'Face found', tone: 'safe' }
    : { key: 'sensor_quality', title: TITLES.sensor_quality, value: offReason, note: coverage === 'calibrating' ? 'Look ahead, eyes open' : 'Adjust the mount while parked', tone: coverage === 'calibrating' || coverage === 'starting' ? 'watch' : 'alert' };

  const condition = tracking ? snapshot.condition : 'none';
  const tone = tracking ? conditionTone(condition) : 'off';
  const score = !tracking ? 0 : Math.min(1, Math.max(
    eyesClosedMs / 5000,
    0.8 * headAwayMs / 8000,
    responsiveness === 'no_response' ? 0.9 : 0,
    snapshot.tier / 3,
  ));
  const face = tracking && snapshot.face
    ? { eyes: snapshot.face.eyesClosed ? 'closed' as const : 'open' as const, yaw: snapshot.face.yaw, pitch: snapshot.face.pitch }
    : null;
  return {
    condition,
    score,
    face,
    tone,
    headline: tracking ? HEADLINE[condition] : offReason,
    detail: !tracking ? (coverage === 'calibrating' ? 'Face forward with your eyes open for a few seconds.' : 'Monitoring resumes when the face is back in view.')
      : condition === 'possible_unresponsiveness' ? 'Eyes closed past five seconds. The guardian step has started.'
        : condition === 'drowsy' ? 'If needed, pull over safely.'
          : condition === 'distracted' ? 'Look back at the road.'
            : 'Watching eyes and head position on this phone.',
    tier: snapshot.tier,
    prompt: responsiveness === 'waiting' ? RESPONSE_PROMPT : null,
    tiles: [
      eyes,
      { key: 'ocular', title: TITLES.ocular, value: 'Off', note: 'Gaze tracking runs in the Driven ML model', tone: 'off' },
      head,
      { key: 'breathing', title: TITLES.breathing, value: 'Off', note: 'Microphone off by design', tone: 'off' },
      response,
      camera,
    ],
  };
}

function signalTone(available: boolean, severity: number): TileTone {
  if (!available) return 'off';
  if (severity >= 0.8) return 'critical';
  if (severity >= 0.45) return 'alert';
  if (severity > 0.1) return 'watch';
  return 'safe';
}

const STATE_TEXT: Record<string, string> = {
  open: 'Open', blinking: 'Blinking', closure_warning: 'Closing', closure_critical: 'Closed 5 s+',
  steady: 'Steady', scanning: 'Scanning', erratic_5s: 'Erratic 5 s', erratic_10s: 'Erratic 10 s',
  stable: 'Stable', nodding: 'Nodding', head_drop: 'Head drop', slumped: 'Slumped',
  regular: 'Regular', irregular: 'Irregular', agonal_pattern_suspected: 'Gasping pattern', not_detected_quiet_cabin: 'Not detected',
  idle: 'Standby', waiting: 'Waiting', responded: 'Responded', no_response: 'No response',
  ok: 'Good', degraded: 'Degraded', fault: 'Fault', unavailable: 'Off',
};

export function viewFromSignal(signal: DriverRiskSignal): LiveView {
  const order: IndicatorName[] = ['eyes', 'ocular', 'head', 'breathing', 'responsiveness', 'sensor_quality'];
  const tiles = order.map((key): Tile => {
    const reading = signal.indicators[key];
    const note = reading.available ? (reading.evidence[0] ?? 'Within baseline') : (reading.unavailable_reason ?? 'Unavailable');
    const tone = key === 'sensor_quality'
      ? (reading.state === 'ok' ? 'safe' : reading.state === 'degraded' ? 'watch' : 'alert')
      : signalTone(reading.available, reading.severity);
    return { key, title: TITLES[key], value: STATE_TEXT[reading.state] ?? reading.state, note, tone };
  });
  const condition = signal.condition;
  const eyes = signal.indicators.eyes;
  const head = signal.indicators.head;
  const face: LiveFace | null = !eyes.available ? null : {
    eyes: eyes.state === 'closure_warning' || eyes.state === 'closure_critical' ? 'closed' : 'open',
    yaw: head.available && head.state !== 'stable' ? 30 : 0,
    pitch: head.state === 'head_drop' ? -30 : 0,
  };
  return {
    condition,
    score: signal.score,
    face,
    tone: conditionTone(condition),
    headline: HEADLINE[condition],
    detail: signal.escalation.reason || 'Driven ML replay',
    tier: signal.escalation.tier,
    prompt: signal.escalation.responsiveness_check === 'waiting' ? RESPONSE_PROMPT : null,
    tiles,
  };
}

/** What a parent reads for a guardian alert row. The database row keeps cue and evidence only. */
export function guardianAlertTitle(cue: 'eye-closure' | 'head-away', evidenceMs: number) {
  if (cue === 'eye-closure') return evidenceMs >= 5000 ? HEADLINE.possible_unresponsiveness : HEADLINE.drowsy;
  return HEADLINE.distracted;
}

export function guardianAlertTone(cue: 'eye-closure' | 'head-away', evidenceMs: number): TileTone {
  if (cue === 'eye-closure') return evidenceMs >= 5000 ? 'critical' : 'alert';
  return 'watch';
}
