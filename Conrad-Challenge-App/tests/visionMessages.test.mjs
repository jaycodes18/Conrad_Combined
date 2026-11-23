import test from 'node:test';
import assert from 'node:assert/strict';
import { parseVisionMessage } from '../src/visionHtml.ts';

test('a face and a genuine tracking miss are valid model observations', () => {
  for (const face of [null, { eyesClosed: 0.2, yaw: 10, pitch: -20 }]) {
    const message = { type: 'observation', at: 100.5, face };
    assert.deepEqual(parseVisionMessage(JSON.stringify(message)), message);
  }
});

test('an incomplete or nonfinite result cannot be counted as a fresh frame', () => {
  assert.equal(parseVisionMessage('{"type":"observation","at":1,"face":{"eyesClosed":0.5,"yaw":1e400,"pitch":0}}'), null);
  assert.equal(parseVisionMessage('{"type":"observation","at":1}'), null);
  assert.equal(parseVisionMessage('{"type":"observation","at":1,"face":{"eyesClosed":2,"yaw":0,"pitch":0}}'), null);
});

test('asset acknowledgements identify both the asset and the chunk', () => {
  const message = { type: 'assetChunk', asset: 'model', index: 4 };
  assert.deepEqual(parseVisionMessage(JSON.stringify(message)), message);
  assert.equal(parseVisionMessage('{"type":"assetChunk","asset":"model"}'), null);
});
