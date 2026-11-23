const { copyFileSync, mkdirSync } = require('node:fs');
const path = require('node:path');

const root = path.join(__dirname, '..');
const mediaPipe = path.join(root, 'node_modules', '@mediapipe', 'tasks-vision');
const leaflet = path.join(root, 'node_modules', 'leaflet');
const destination = path.join(root, 'assets', 'runtime');

mkdirSync(destination, { recursive: true });
for (const [source, target] of [
  ['vision_bundle.cjs', 'face-bundle.txt'],
  ['wasm/vision_wasm_internal.js', 'face-loader.txt'],
  ['wasm/vision_wasm_internal.wasm', 'face-runtime.wasm'],
]) {
  copyFileSync(path.join(mediaPipe, source), path.join(destination, target));
}
for (const [source, target] of [
  ['dist/leaflet.js', 'leaflet-script.txt'],
  ['dist/leaflet.css', 'leaflet-style.txt'],
  ['LICENSE', 'leaflet-license.txt'],
]) {
  copyFileSync(path.join(leaflet, source), path.join(destination, target));
}
