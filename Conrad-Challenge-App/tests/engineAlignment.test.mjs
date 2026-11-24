import test from 'node:test';
import assert from 'node:assert/strict';
import { CRITICAL_CLOSURE_MS, RESPONSE_WINDOW_MS, RiskEngine } from '../src/risk.ts';
import { guardianAlertTitle, viewFromEngine, viewFromSignal } from '../src/liveView.ts';
import { drivenMlReplays } from '../src/drivenMl/replays.ts';

const neutral = { eyesClosed: 0.08, yaw: 0, pitch: 0 };
const closed = { eyesClosed: 0.95, yaw: 0, pitch: 0 };
const turned = { eyesClosed: 0.08, yaw: 34, pitch: 0 };

function calibrated(engine) {
  for (const at of [0, 700, 1400, 2000]) engine.update({ at, face: neutral });
  return 2000;
}

function feed(engine, from, to, face = neutral, step = 100) {
  const out = [];
  let last;
  for (let at = from; at <= to; at += step) {
    last = engine.update({ at, face });
    out.push(...last.alerts.map((alert) => ({ ...alert, at })));
  }
  return { alerts: out, last };
}

test('five seconds of closed eyes escalates to the guardian and reads as possible unresponsiveness', () => {
  const engine = new RiskEngine();
  const base = calibrated(engine);
  const { alerts, last } = feed(engine, base + 100, base + 100 + CRITICAL_CLOSURE_MS, closed);
  const l2 = alerts.filter((alert) => alert.level === 2);
  assert.equal(l2.length, 1);
  assert.equal(l2[0].evidenceMs, CRITICAL_CLOSURE_MS);
  assert.equal(last.snapshot.condition, 'possible_unresponsiveness');
  assert.equal(last.snapshot.tier, 2);
  assert.equal(viewFromEngine(last.snapshot, last.coverage).tone, 'critical');
});

test('a held head turn with no response escalates when the response window closes, before 8 s', () => {
  const engine = new RiskEngine();
  const base = calibrated(engine);
  const { alerts, last } = feed(engine, base + 100, base + 100 + 2600 + RESPONSE_WINDOW_MS, turned);
  const l1 = alerts.find((alert) => alert.level === 1);
  const l2 = alerts.find((alert) => alert.level === 2);
  assert.ok(l1 && l2);
  assert.equal(l2.at - l1.at, RESPONSE_WINDOW_MS);
  assert.ok(l2.evidenceMs < 8000);
  assert.equal(last.snapshot.responsiveness, 'no_response');
});

test('looking back within the window records a response and no guardian alert', () => {
  const engine = new RiskEngine();
  const base = calibrated(engine);
  const first = feed(engine, base + 100, base + 2700, turned);
  assert.equal(first.alerts.filter((alert) => alert.level === 1).length, 1);
  assert.equal(first.last.snapshot.responsiveness, 'waiting');
  assert.equal(viewFromEngine(first.last.snapshot, 'tracking').prompt, 'Open your eyes and look at the road');
  const after = feed(engine, base + 2800, base + 4000, neutral);
  assert.equal(after.alerts.length, 0);
  assert.equal(after.last.snapshot.responsiveness, 'responded');
  assert.ok(after.last.snapshot.responseLatencyMs < 1000);
});

test('losing the face during a response check marks it unavailable, not unresponsive', () => {
  const engine = new RiskEngine();
  const base = calibrated(engine);
  feed(engine, base + 100, base + 2700, turned);
  const lost = engine.update({ at: base + 2800, face: null });
  assert.equal(lost.coverage, 'no-face');
  assert.equal(lost.snapshot.responsiveness, 'unavailable');
  assert.equal(lost.alerts.length, 0);
});

test('the phone view shows all six ML indicators and marks unmeasured ones off', () => {
  const engine = new RiskEngine();
  const base = calibrated(engine);
  const { last } = feed(engine, base + 100, base + 500, neutral);
  const view = viewFromEngine(last.snapshot, last.coverage);
  assert.deepEqual(view.tiles.map((tile) => tile.key), ['eyes', 'ocular', 'head', 'breathing', 'responsiveness', 'sensor_quality']);
  assert.equal(view.tiles.find((tile) => tile.key === 'ocular').tone, 'off');
  assert.equal(view.tiles.find((tile) => tile.key === 'breathing').tone, 'off');
  assert.equal(view.headline, 'All clear');
});

test('a Driven ML replay signal maps onto the same view', () => {
  const signal = drivenMlReplays['eye-closure'].at(-1);
  const view = viewFromSignal(signal);
  assert.equal(view.condition, signal.condition);
  assert.equal(view.tier, signal.escalation.tier);
  assert.equal(view.tiles.length, 6);
});

test('parent alert titles never use banned wording', () => {
  for (const [cue, ms] of [['eye-closure', 2500], ['eye-closure', 6000], ['head-away', 3000]]) {
    const title = guardianAlertTitle(cue, ms);
    assert.doesNotMatch(title, /drunk|intoxicat|unconscious|BAC|diagnos/i);
  }
  assert.equal(guardianAlertTitle('eye-closure', 6000), 'Possible unresponsiveness');
});
