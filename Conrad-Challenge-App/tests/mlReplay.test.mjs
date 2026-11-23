import test from 'node:test';
import assert from 'node:assert/strict';
import { drivenMlReplays } from '../src/drivenMl/replays.ts';
import { mlReplayWarning } from '../src/mlReplay.ts';

for (const [cue, evidenceMs] of [['eye-closure', 2000], ['head-away', 2100]]) {
  test(`Driven ML ${cue} replay yields one warning using the Python evidence duration`, () => {
    const signals = drivenMlReplays[cue];
    assert.ok(signals.length > 1);
    for (let index = 1; index < signals.length; index += 1) {
      assert.ok(signals[index].t_seconds > signals[index - 1].t_seconds);
    }
    assert.ok(signals.slice(0, -1).every((signal) => mlReplayWarning(signal) === null));
    assert.deepEqual(mlReplayWarning(signals.at(-1)), { cue, level: 1, evidenceMs });
  });
}

test('a replay never turns contact or call recommendations into warnings', () => {
  const warning = drivenMlReplays['eye-closure'].at(-1);
  for (const level of [2, 3]) {
    assert.equal(mlReplayWarning({
      ...warning,
      app_alert: { ...warning.app_alert, level },
      escalation: { ...warning.escalation, notify_contacts: true, voice_call_recommended: true },
    }), null);
  }
  assert.equal(mlReplayWarning({ ...warning, escalation: { ...warning.escalation, in_car_alert: false } }), null);
});
