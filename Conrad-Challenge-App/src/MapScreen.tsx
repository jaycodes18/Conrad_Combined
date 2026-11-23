import { useEffect, useRef, useState } from 'react';
import { ActivityIndicator, Linking, StyleSheet, useWindowDimensions, View } from 'react-native';
import * as Location from 'expo-location';
import WebView from 'react-native-webview';
import { readLocalAsset } from './localAsset';
import { makeMapHtml, MAP_DOCUMENT, type MapPin } from './mapHtml';
import { Action, Body, colors, Heading } from './ui';

const MAP_ORIGIN = 'https://www.openstreetmap.org';
const DEMO_PIN = { latitude: 32.7767, longitude: -96.7970 };
const LEAFLET_SCRIPT = require('../assets/runtime/leaflet-script.txt');
const LEAFLET_STYLE = require('../assets/runtime/leaflet-style.txt');

function fullMapUrl({ latitude, longitude }: MapPin) {
  const lat = latitude.toFixed(4);
  const lon = longitude.toFixed(4);
  return `${MAP_ORIGIN}/?mlat=${lat}&mlon=${lon}#map=14/${lat}/${lon}`;
}

async function findLocation() {
  const previous = await Location.getLastKnownPositionAsync({ maxAge: 60_000, requiredAccuracy: 500 }).catch(() => null);
  if (previous) return previous;
  let timer: ReturnType<typeof setTimeout> | undefined;
  try {
    return await Promise.race([
      Location.getCurrentPositionAsync({ accuracy: Location.Accuracy.Balanced }),
      new Promise<null>((resolve) => { timer = setTimeout(() => resolve(null), 15_000); }),
    ]);
  } finally {
    clearTimeout(timer);
  }
}

function ParkedMap({ pin }: { pin: MapPin }) {
  const { height } = useWindowDimensions();
  const mapHeight = Math.min(400, Math.max(240, height * 0.48));
  const [attempt, setAttempt] = useState(0);
  const [source, setSource] = useState<{ html: string; baseUrl: string }>();
  const [status, setStatus] = useState<'loading' | 'ready' | 'unavailable'>('loading');
  const [browserError, setBrowserError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    void Promise.all([readLocalAsset(LEAFLET_SCRIPT, 'text'), readLocalAsset(LEAFLET_STYLE, 'text')])
      .then(([script, css]) => {
        if (!cancelled) setSource({ html: makeMapHtml(script, css, pin), baseUrl: MAP_DOCUMENT });
      })
      .catch(() => { if (!cancelled) setStatus('unavailable'); });
    return () => { cancelled = true; };
  }, [attempt, pin]);

  useEffect(() => {
    if (status !== 'loading') return;
    const timer = setTimeout(() => setStatus('unavailable'), 15_000);
    return () => clearTimeout(timer);
  }, [status, attempt]);

  return <>
    <View style={[mapStyles.frame, { height: mapHeight }]}>
      {status !== 'unavailable' && source && <WebView
        source={source}
        style={mapStyles.webview}
        originWhitelist={['*']}
        onShouldStartLoadWithRequest={({ url }) => {
          if (url === MAP_DOCUMENT || url === 'about:blank') return true;
          if (url === `${MAP_ORIGIN}/copyright` || url === 'https://leafletjs.com/') {
            void Linking.openURL(url).catch(() => setBrowserError(true));
          }
          return false;
        }}
        applicationNameForUserAgent="ConradDriverMonitor/0.1"
        nestedScrollEnabled
        setSupportMultipleWindows={false}
        thirdPartyCookiesEnabled={false}
        mixedContentMode="never"
        geolocationEnabled={false}
        onMessage={({ nativeEvent }) => {
          if (nativeEvent.data === 'tiles-ready') setStatus((current) => current === 'loading' ? 'ready' : current);
        }}
        onError={() => setStatus('unavailable')}
        onRenderProcessGone={() => setStatus('unavailable')}
        onContentProcessDidTerminate={() => setStatus('unavailable')}
      />}
      {status === 'loading' && <View style={mapStyles.overlay} pointerEvents="none">
        <ActivityIndicator color={colors.accent} />
        <Body>Loading map…</Body>
      </View>}
      {status === 'unavailable' && <View style={mapStyles.overlay}>
        <Body urgent>Map unavailable. Check your internet connection.</Body>
        <Action title="Retry map" secondary onPress={() => {
          setBrowserError(false);
          setSource(undefined);
          setStatus('loading');
          setAttempt((value) => value + 1);
        }} />
      </View>}
    </View>
    {status === 'unavailable' && <Action title="Open map in browser" secondary onPress={() => {
      setBrowserError(false);
      void Linking.openURL(fullMapUrl(pin)).catch(() => setBrowserError(true));
    }} />}
    {browserError && <Body urgent>The browser could not open the map.</Body>}
  </>;
}

export function MapScreen() {
  const [pin, setPin] = useState<MapPin>();
  const [locating, setLocating] = useState(false);
  const [message, setMessage] = useState('');
  const request = useRef(0);
  useEffect(() => () => { request.current += 1; }, []);

  async function locate() {
    const id = ++request.current;
    setLocating(true);
    setMessage('');
    try {
      const permission = await Location.requestForegroundPermissionsAsync();
      if (request.current !== id) return;
      if (!permission.granted) {
        setMessage(permission.canAskAgain ? 'Allow location to center the map on you.' : 'Enable location for the app in Android or iPhone settings.');
        return;
      }
      if (!await Location.hasServicesEnabledAsync()) {
        if (request.current === id) setMessage('Turn on location services and try again.');
        return;
      }
      if (request.current !== id) return;
      const fix = await findLocation();
      if (request.current !== id) return;
      if (!fix) setMessage('Location took too long. Try again with a clear view of the sky.');
      else if (!Number.isFinite(fix.coords.latitude) || !Number.isFinite(fix.coords.longitude) ||
        Math.abs(fix.coords.latitude) > 85.05 || Math.abs(fix.coords.longitude) > 180) {
        setMessage('The map does not cover this location.');
      } else setPin({ latitude: fix.coords.latitude, longitude: fix.coords.longitude });
    } catch {
      if (request.current === id) setMessage('Location is temporarily unavailable. Try again.');
    } finally {
      if (request.current === id) setLocating(false);
    }
  }

  const shown = pin ?? DEMO_PIN;
  return <>
    <Heading>Map</Heading>
    <Body>{pin ? 'Your location' : 'Demo location · Dallas, TX'}</Body>
    <ParkedMap key={`${shown.latitude},${shown.longitude}`} pin={shown} />
    {!pin && <Body>Centering on your location loads that area from OpenStreetMap.</Body>}
    <Action title={locating ? 'Finding location…' : pin ? 'Refresh my location' : 'Use my location'}
      disabled={locating} onPress={() => void locate()} />
    {pin && <Action title="Show demo location" secondary disabled={locating} onPress={() => {
      setPin(undefined);
      setMessage('');
    }} />}
    {message && <Body urgent>{message}</Body>}
  </>;
}

const mapStyles = StyleSheet.create({
  frame: { borderRadius: 20, overflow: 'hidden', borderWidth: 1, borderColor: colors.line, backgroundColor: colors.raised },
  webview: { backgroundColor: colors.raised },
  overlay: { ...StyleSheet.absoluteFill, padding: 16, justifyContent: 'center', gap: 12, backgroundColor: colors.bg },
});
