import type { Observation } from './risk';

export const VISION_CHUNK_SIZE = 128 * 1024;
type VisionAsset = 'wasm' | 'model';

type VisionMessage = { type: 'boot' | 'ready' } |
  { type: 'assetChunk'; asset: VisionAsset; index: number } |
  ({ type: 'observation' } & Observation) |
  { type: 'fault'; message: string };

export function makeVisionHtml(bundle: string, loader: string) {
  // Runtime bytes and model data are transferred locally in acknowledged chunks. This keeps
  // the initial WebView HTML small enough for Android's loadDataWithBaseURL bridge.
  return `<!doctype html>
<html><head>
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1" />
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline' 'wasm-unsafe-eval'; img-src data: blob:; style-src 'unsafe-inline'; connect-src 'none';" />
</head><body style="margin:0">
<canvas id="frame" width="640" height="480" style="display:none"></canvas>
<script>
self.Module = {};
${loader}
var exports = {};
var module = { exports };
${bundle}
self.Vision = module.exports;
self.ModuleFactory = ModuleFactory;
</script>
<script>
(() => {
  const send = (message) => window.ReactNativeWebView.postMessage(JSON.stringify(message));
  let transfer = null;
  let task = null;
  let busy = false;

  const appendChunk = (message) => {
    const { type, index, bytes, data, last } = message;
    if (index === 0 && !transfer && Number.isSafeInteger(bytes) && bytes > 0 && bytes <= 32 * 1024 * 1024) {
      transfer = { type, index: 0, offset: 0, buffer: new Uint8Array(bytes) };
    }
    if (!transfer || transfer.type !== type || transfer.index !== index || transfer.buffer.length !== bytes) {
      throw Error('Local ' + type + ' transfer received an unexpected chunk');
    }
    const binary = atob(data);
    const end = transfer.offset + binary.length;
    if (end > bytes || last !== (end === bytes)) throw Error('Local ' + type + ' transfer has the wrong size');
    for (let i = 0; i < binary.length; i += 1) transfer.buffer[transfer.offset + i] = binary.charCodeAt(i);
    transfer.offset = end;
    transfer.index += 1;
    if (!last) return null;
    const buffer = transfer.buffer;
    transfer = null;
    return buffer;
  };

  const headAngles = (matrix) => {
    const values = matrix.data || matrix;
    if (!values || values.length < 16) throw Error('Invalid facial transformation matrix');
    // MediaPipe returns a row-major 4x4 face transformation matrix.
    const r00 = values[0];
    const r10 = values[4];
    const r20 = values[8];
    const r21 = values[9];
    const r22 = values[10];
    return {
      yaw: Math.atan2(-r20, Math.hypot(r00, r10)) * 180 / Math.PI,
      pitch: Math.atan2(r21, r22) * 180 / Math.PI,
    };
  };

  const initializeModel = async (modelAssetBuffer) => {
    const fileset = { wasmLoaderPath: '', wasmBinaryPath: '', assetLoaderPath: '', assetBinaryPath: '' };
    task = await self.Vision.FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetBuffer },
      runningMode: 'IMAGE',
      numFaces: 1,
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    });
    send({ type: 'ready' });
  };

  const inferFrame = (data, at) => {
    if (!task || busy) return;
    busy = true;
    const image = new Image();
    image.onload = () => {
      try {
        const canvas = document.getElementById('frame');
        const context = canvas.getContext('2d');
        const scale = Math.min(640 / image.width, 480 / image.height, 1);
        canvas.width = Math.max(1, Math.round(image.width * scale));
        canvas.height = Math.max(1, Math.round(image.height * scale));
        context.drawImage(image, 0, 0, canvas.width, canvas.height);

        const result = task.detect(canvas);
        let face = null;
        const blendshapes = result.faceBlendshapes?.[0]?.categories;
        const matrix = result.facialTransformationMatrixes?.[0];
        if (blendshapes && matrix) {
          const left = blendshapes.find((item) => item.categoryName === 'eyeBlinkLeft')?.score;
          const right = blendshapes.find((item) => item.categoryName === 'eyeBlinkRight')?.score;
          if (Number.isFinite(left) && Number.isFinite(right)) {
            const angles = headAngles(matrix);
            const features = { eyesClosed: Math.min(left, right), yaw: angles.yaw, pitch: angles.pitch };
            if (Object.values(features).every(Number.isFinite)) face = features;
          }
        }
        send({ type: 'observation', at, face });
      } catch (error) {
        send({ type: 'fault', message: String(error) });
      } finally {
        busy = false;
      }
    };
    image.onerror = () => {
      busy = false;
      send({ type: 'fault', message: 'Could not decode captured camera frame' });
    };
    image.src = 'data:image/jpeg;base64,' + data;
  };

  window.receiveNative = async (raw) => {
    try {
      const message = JSON.parse(raw);
      if (message.type === 'wasm' || message.type === 'model') {
        const buffer = appendChunk(message);
        if (buffer && message.type === 'wasm') self.Module.wasmBinary = buffer;
        send({ type: 'assetChunk', asset: message.type, index: message.index });
        if (buffer && message.type === 'model') await initializeModel(buffer);
      } else if (message.type === 'frame') {
        inferFrame(message.data, message.at);
      }
    } catch (error) {
      send({ type: 'fault', message: String(error) });
    }
  };

  send({ type: 'boot' });
})();
</script>
</body></html>`;
}

export function parseVisionMessage(raw: string): VisionMessage | null {
  try {
    const value = JSON.parse(raw);
    switch (value?.type) {
      case 'boot':
      case 'ready':
        return { type: value.type };
      case 'assetChunk':
        return (value.asset === 'wasm' || value.asset === 'model') && Number.isSafeInteger(value.index) && value.index >= 0
          ? { type: 'assetChunk', asset: value.asset, index: value.index } : null;
      case 'observation': {
        if (!Number.isFinite(value.at) || value.at < 0) return null;
        const face = value.face;
        if (face !== null && (!face || ![face.eyesClosed, face.yaw, face.pitch].every(Number.isFinite) ||
          face.eyesClosed < 0 || face.eyesClosed > 1)) return null;
        return { type: 'observation', at: value.at, face };
      }
      case 'fault':
        return typeof value.message === 'string' ? { type: 'fault', message: value.message } : null;
      default:
        return null;
    }
  } catch { return null; }
}
