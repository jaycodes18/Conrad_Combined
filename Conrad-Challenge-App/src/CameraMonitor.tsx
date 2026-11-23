import { CameraView, type CameraCapturedPicture } from 'expo-camera';
import { File } from 'expo-file-system';
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { StyleSheet, View } from 'react-native';
import WebView, { type WebViewMessageEvent, type WebViewProps } from 'react-native-webview';
import { readLocalAsset } from './localAsset';
import { MAX_OBSERVATION_GAP_MS } from './risk';
import type { Observation } from './risk';
import { colors } from './ui';
import { makeVisionHtml, parseVisionMessage, VISION_CHUNK_SIZE } from './visionHtml';

export type CameraMonitorProps = {
  onObservation: (observation: Observation) => void;
  onTrackingDelay: (slowFrameMs?: number) => void;
  onFault: (message: string) => void;
};

const MODEL = require('../assets/face-landmarker.task');
const BUNDLE = require('../assets/runtime/face-bundle.txt');
const LOADER = require('../assets/runtime/face-loader.txt');
const WASM = require('../assets/runtime/face-runtime.wasm');
// CameraX falls back to a supported size. Set the target before mounting to avoid rebinding during capture.
const CAPTURE_SIZE = '640x480';
const CAMERA_INTERVAL_MS = 300;
const STALL_MS = 6000;
const STARTUP_WATCHDOG_MS = 45_000;
type MonitorWebViewProps = WebViewProps & { ref?: React.Ref<WebView> };
const NativeWebView = WebView as unknown as React.ComponentType<MonitorWebViewProps>;

export function CameraMonitor({ onObservation, onTrackingDelay, onFault }: CameraMonitorProps) {
  const camera = useRef<CameraView>(null);
  const webview = useRef<WebView>(null);
  const cameraReady = useRef(false);
  const visionReady = useRef(false);
  const running = useRef(false);
  const disposed = useRef(false);
  const faulted = useRef(false);
  const pendingFrameAt = useRef<number | null>(null);
  const lastFreshAt = useRef<number | null>(null);
  const delayed = useRef(false);
  const assetChunkAck = useRef<{ asset: string; index: number; resolve: () => void } | null>(null);
  const booted = useRef(false);
  const [html, setHtml] = useState<string>();
  const visionPage = useMemo(() => html ? { html, baseUrl: 'https://localhost/' } : null, [html]);
  const onObservationRef = useRef(onObservation);
  const onFaultRef = useRef(onFault);
  const onTrackingDelayRef = useRef(onTrackingDelay);
  onObservationRef.current = onObservation;
  onFaultRef.current = onFault;
  onTrackingDelayRef.current = onTrackingDelay;

  const fail = useCallback((message: string) => {
    if (disposed.current || faulted.current) return;
    faulted.current = true;
    running.current = false;
    assetChunkAck.current?.resolve();
    onFaultRef.current(message);
  }, []);

  const reportDelay = useCallback((slowFrameMs?: number) => {
    if (faulted.current || disposed.current || (delayed.current && slowFrameMs === undefined)) return;
    delayed.current = true;
    onTrackingDelayRef.current(slowFrameMs);
  }, []);

  const postToVision = useCallback((value: object) => {
    const serialized = JSON.stringify(JSON.stringify(value));
    webview.current?.injectJavaScript(`window.receiveNative&&window.receiveNative(${serialized});true;`);
  }, []);

  useEffect(() => {
    disposed.current = false;
    let cancelled = false;
    void (async () => {
      try {
        const [bundle, loader] = await Promise.all([
          readLocalAsset(BUNDLE, 'text'), readLocalAsset(LOADER, 'text'),
        ]);
        if (!cancelled) setHtml(makeVisionHtml(bundle, loader));
      } catch (error) {
        if (!cancelled) fail(`Could not load the local vision runtime: ${String(error)}`);
      }
    })();
    return () => {
      cancelled = true;
      disposed.current = true;
      running.current = false;
      assetChunkAck.current?.resolve();
    };
  }, [fail]);

  const sendAssets = useCallback(async () => {
    try {
      for (const [assetType, assetModule] of [['wasm', WASM], ['model', MODEL]] as const) {
        if (disposed.current || faulted.current) return;
        const content = await readLocalAsset(assetModule, 'base64');
        const total = Math.ceil(content.length / VISION_CHUNK_SIZE);
        const padding = content.endsWith('==') ? 2 : content.endsWith('=') ? 1 : 0;
        const bytes = content.length / 4 * 3 - padding;
        for (let index = 0; index < total; index += 1) {
          if (disposed.current || faulted.current) return;
          await new Promise<void>((resolve, reject) => {
            const timeout = setTimeout(() => {
              assetChunkAck.current = null;
              reject(new Error(`Local ${assetType} transfer stalled`));
            }, 15000);
            assetChunkAck.current = {
              asset: assetType, index, resolve: () => { clearTimeout(timeout); assetChunkAck.current = null; resolve(); },
            };
            const offset = index * VISION_CHUNK_SIZE;
            postToVision({ type: assetType, index, bytes, data: content.slice(offset, offset + VISION_CHUNK_SIZE), last: index === total - 1 });
          });
        }
      }
    } catch (error) {
      fail(`Could not load the local face model: ${String(error)}`);
    }
  }, [fail, postToVision]);

  const runCaptureLoop = useCallback(async () => {
    if (running.current || faulted.current || disposed.current || !cameraReady.current || !visionReady.current) return;
    running.current = true;
    let captureStartedAt = -CAMERA_INTERVAL_MS;
    while (!disposed.current && running.current) {
      if (pendingFrameAt.current === null) {
        const remaining = CAMERA_INTERVAL_MS - (performance.now() - captureStartedAt);
        if (remaining > 0) await new Promise((resolve) => setTimeout(resolve, remaining));
        if (disposed.current || !running.current) break;
        let photo: CameraCapturedPicture | undefined;
        try {
          if (!camera.current) throw new Error('Camera is not available');
          captureStartedAt = performance.now();
          photo = await camera.current.takePictureAsync({ base64: true, shutterSound: false, quality: 0.5 });
          if (!photo.base64) throw new Error('Camera returned no image data');
          if (disposed.current || !running.current) break;
          pendingFrameAt.current = captureStartedAt;
          postToVision({ type: 'frame', at: captureStartedAt, data: photo.base64 });
        } catch (error) {
          fail(`Camera capture failed: ${String(error)}`);
        } finally {
          if (photo?.uri) {
            try { new File(photo.uri).delete(); } catch (error) {
              fail(`Could not remove a temporary camera image: ${String(error)}`);
            }
          }
        }
      }
      if (!disposed.current && running.current) await new Promise((resolve) => setTimeout(resolve, 100));
    }
  }, [fail, postToVision]);

  const onMessage = useCallback((event: WebViewMessageEvent) => {
    if (disposed.current || faulted.current) return;
    const message = parseVisionMessage(event.nativeEvent.data);
    if (!message) return;
    const ack = assetChunkAck.current;
    if (message.type === 'assetChunk' && ack?.asset === message.asset && ack.index === message.index) {
      ack.resolve();
    } else if (message.type === 'boot') {
      if (!booted.current) { booted.current = true; void sendAssets(); }
    } else if (message.type === 'ready') {
      visionReady.current = true;
      void runCaptureLoop();
    } else if (message.type === 'observation' && pendingFrameAt.current === message.at) {
      pendingFrameAt.current = null;
      const now = performance.now();
      const frameMs = now - message.at;
      if (frameMs > MAX_OBSERVATION_GAP_MS) {
        reportDelay(frameMs);
        return;
      }
      lastFreshAt.current = now;
      delayed.current = false;
      onObservationRef.current({ at: message.at, face: message.face });
    } else if (message.type === 'fault') {
      fail(`Face model failed: ${message.message}`);
    }
  }, [fail, reportDelay, runCaptureLoop, sendAssets]);

  const onCameraReady = useCallback(() => {
    cameraReady.current = true;
    void runCaptureLoop();
  }, [runCaptureLoop]);

  useEffect(() => {
    const startupTimer = setTimeout(() => {
      if (lastFreshAt.current === null) fail('Camera and face model did not deliver a fresh frame within 45 seconds.');
    }, STARTUP_WATCHDOG_MS);
    const timer = setInterval(() => {
      if (lastFreshAt.current === null) return;
      const gap = performance.now() - lastFreshAt.current;
      if (gap > STALL_MS) fail('Camera or face model stopped delivering fresh frames.');
      else if (gap > MAX_OBSERVATION_GAP_MS) reportDelay();
    }, 250);
    return () => { clearTimeout(startupTimer); clearInterval(timer); };
  }, [fail, reportDelay]);

  return (
    <View style={styles.preview}>
      <CameraView ref={camera} style={StyleSheet.absoluteFill} facing="front" pictureSize={CAPTURE_SIZE}
        onCameraReady={onCameraReady} onMountError={({ message }) => fail(message)} />
      {visionPage ? <NativeWebView
        ref={webview}
        originWhitelist={['*']}
        source={visionPage}
        javaScriptEnabled
        domStorageEnabled={false}
        onShouldStartLoadWithRequest={(request) => {
          if (request.url === 'about:blank') return true;
          try {
            const url = new URL(request.url);
            return url.protocol === 'https:' && url.hostname === 'localhost' && url.pathname === '/';
          } catch { return false; }
        }}
        onMessage={onMessage}
        onError={({ nativeEvent }) => fail(`Local vision WebView failed: ${nativeEvent.description}`)}
        onHttpError={({ nativeEvent }) => fail(`Local vision WebView returned HTTP ${nativeEvent.statusCode}`)}
        onRenderProcessGone={() => fail('Android WebView stopped running the face model. Reopen the session to retry.')}
        onContentProcessDidTerminate={() => fail('The face model renderer stopped. Reopen the session to retry.')}
        style={styles.webview}
      /> : null}
    </View>
  );
}

const styles = StyleSheet.create({
  preview: { position: 'relative', width: '100%', aspectRatio: 4 / 3, maxHeight: 265, borderRadius: 18, overflow: 'hidden', backgroundColor: colors.surface },
  webview: { position: 'absolute', top: 0, left: 0, width: 1, height: 1, opacity: 0.001, backgroundColor: 'transparent' },
});

export default CameraMonitor;
