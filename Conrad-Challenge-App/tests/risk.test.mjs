import test from 'node:test';
import assert from 'node:assert/strict';
import { RiskEngine, cueLabel } from '../src/risk.ts';
import { DEMO_DURATION_MS, drillObservation } from '../src/demo.ts';

const neutral = { eyesClosed: 0.08, yaw: 0, pitch: 0 };
const closed = { eyesClosed: 0.95, yaw: 0, pitch: 0 };
const turned = { eyesClosed: 0.08, yaw: 34, pitch: 0 };
const both = { eyesClosed: 0.95, yaw: 34, pitch: 0 };

function calibrated(engine, start = 0) {
  for (const dt of [0, 700, 1400, 2000]) engine.update({ at: start + dt, face: neutral });
  return start + 2000;
}

function run(engine, samples) {
  return samples.flatMap(([at, face]) => engine.update({ at, face }).alerts);
}

function span(from, to, face, step = 500) {
  const samples = [];
  for (let at = from; at < to; at += step) samples.push([at, face]);
  samples.push([to, face]);
  return samples;
}

test('calibrates on repeated neutral samples and labels cues', () => {
  const engine = new RiskEngine();
  assert.equal(engine.update({ at: 0, face: neutral }).coverage, 'calibrating');
  engine.update({ at: 700, face: neutral });
  engine.update({ at: 1400, face: neutral });
  assert.equal(engine.update({ at: 2000, face: neutral }).coverage, 'tracking');
  assert.equal(cueLabel['eye-closure'], 'Sustained eye closure');
  assert.equal(cueLabel['head-away'], 'Sustained head turn or tilt');
});

test('one simultaneous-cue episode emits one L1, and rearms after both cues clear for 1.5s', () => {
  const engine = new RiskEngine();
  const base = calibrated(engine);
  const alerts = run(engine, [
    ...span(base + 100, base + 2200, both),
    ...span(base + 2700, base + 4200, neutral),
    ...span(base + 4700, base + 6800, both),
  ]);
  assert.equal(alerts.filter((a) => a.level === 1).length, 2);
  assert.equal(alerts[0].cue, 'eye-closure');
  assert.equal(alerts[1].cue, 'eye-closure');
});

test('a separated warning within two minutes can produce at most one L2 per session', () => {
  const engine = new RiskEngine();
  const base = calibrated(engine);
  const alerts = run(engine, [
    ...span(base + 100, base + 2200, closed),
    ...span(base + 2700, base + 4200, neutral),
    ...span(base + 4700, base + 7300, turned),
    ...span(base + 7800, base + 9300, neutral),
    ...span(base + 9800, base + 11900, closed),
  ]);
  assert.equal(alerts.filter((a) => a.level === 1).length, 3);
  assert.equal(alerts.filter((a) => a.level === 2).length, 1);
});

test('eight seconds of continuous cue evidence escalates once', () => {
  const engine = new RiskEngine();
  const base = calibrated(engine);
  const alerts = run(engine, span(base + 1, base + 8001, closed));
  assert.equal(alerts.filter((a) => a.level === 1).length, 1);
  assert.equal(alerts.filter((a) => a.level === 2).length, 1);
});

test('L3 requires drill mode and 18 seconds of eye-closure evidence, once', () => {
  const normal = new RiskEngine();
  const drill = new RiskEngine(true);
  const normalBase = calibrated(normal);
  const drillBase = calibrated(drill);
  const samples = (base) => span(base + 1, base + 18001, closed);
  assert.equal(run(normal, samples(normalBase)).filter((a) => a.level === 3).length, 0);
  assert.equal(run(drill, samples(drillBase)).filter((a) => a.level === 3).length, 1);
});

test('missing face clears active evidence and reports no-face coverage', () => {
  const engine = new RiskEngine();
  const base = calibrated(engine);
  engine.update({ at: base + 1, face: closed });
  assert.equal(engine.update({ at: base + 1002, face: null }).coverage, 'no-face');
  const alerts = run(engine, span(base + 1003, base + 3203, closed));
  assert.equal(alerts.filter((a) => a.level === 1).length, 1);
  assert.ok(alerts[0].evidenceMs < 3000);
});

test('out-of-order observations do not add alerts or advance evidence', () => {
  const engine = new RiskEngine();
  const base = calibrated(engine);
  engine.update({ at: base + 1, face: closed });
  assert.deepEqual(engine.update({ at: base, face: closed }).alerts, []);
  const alerts = run(engine, span(base + 501, base + 2501, closed)).filter((a) => a.level === 1);
  assert.equal(alerts.filter((a) => a.level === 1).length, 1);
});

test('an interrupted camera frame sequence must establish fresh evidence before an alert', () => {
  const engine = new RiskEngine();
  const base = calibrated(engine);
  assert.deepEqual(run(engine, span(base + 100, base + 1900, closed, 300)), []);
  engine.interruptTracking();
  assert.deepEqual(run(engine, span(base + 2100, base + 4100, closed, 400)), []);
  assert.deepEqual(engine.update({ at: base + 4200, face: closed }).alerts,
    [{ cue: 'eye-closure', level: 1, evidenceMs: 2100 }]);
});

test('demo drill is a 25 second simulated cue and starts/ends with neutral data', () => {
  assert.equal(DEMO_DURATION_MS, 25_000);
  assert.deepEqual(drillObservation(0, 'eye-closure').face, neutral);
  assert.deepEqual(drillObservation(4000, 'eye-closure').face, closed);
  assert.deepEqual(drillObservation(23_000, 'eye-closure').face, neutral);
});
