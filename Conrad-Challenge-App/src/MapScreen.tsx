import React, { useEffect, useMemo, useRef, useState } from 'react';
import { ActivityIndicator, Linking, StyleSheet, Text, View } from 'react-native';
import type { StyleProp, ViewStyle } from 'react-native';
import WebView, { type WebViewProps } from 'react-native-webview';
import { readLocalAsset } from './localAsset';
import { makeMapHtml, MAP_DOCUMENT, markersJson } from './mapHtml';
import type { MapMarker } from './mapHtml';
import { colors } from './ui';

export type { MapMarker };

const MAP_ORIGIN = 'https://www.openstreetmap.org';
const LEAFLET_SCRIPT = require('../assets/runtime/leaflet-script.txt');
const LEAFLET_STYLE = require('../assets/runtime/leaflet-style.txt');

type MapWebViewProps = WebViewProps & { ref?: React.Ref<WebView> };
const NativeWebView = WebView as unknown as React.ComponentType<MapWebViewProps>;

/** Interactive family map. Markers update in place; tapping a marker calls `onSelect`. */
export function LiveMap({ markers, focus = null, onSelect, style }: {
  markers: MapMarker[]; focus?: string | null; onSelect?: (id: string) => void; style?: StyleProp<ViewStyle>;
}) {
  const webview = useRef<WebView>(null);
  const [assets, setAssets] = useState<{ script: string; css: string }>();
  const [ready, setReady] = useState(false);
  const [tiles, setTiles] = useState<'loading' | 'ok' | 'offline'>('loading');
  const [failed, setFailed] = useState(false);
  const initial = useRef({ markers, focus });

  useEffect(() => {
    let cancelled = false;
    void Promise.all([readLocalAsset(LEAFLET_SCRIPT, 'text'), readLocalAsset(LEAFLET_STYLE, 'text')])
      .then(([script, css]) => { if (!cancelled) setAssets({ script, css }); })
      .catch(() => { if (!cancelled) setFailed(true); });
    return () => { cancelled = true; };
  }, []);

  const source = useMemo(() => assets
    ? { html: makeMapHtml(assets.script, assets.css, initial.current.markers, initial.current.focus), baseUrl: MAP_DOCUMENT }
    : null, [assets]);

  useEffect(() => {
    if (ready) webview.current?.injectJavaScript(`window.setMarkers(${markersJson(markers)}); true;`);
  }, [markers, ready]);

  useEffect(() => {
    if (ready) webview.current?.injectJavaScript(`window.focusMarker(${JSON.stringify(focus)}); true;`);
  }, [focus, ready]);

  useEffect(() => {
    const timer = setTimeout(() => setTiles((current) => current === 'loading' ? 'offline' : current), 12_000);
    return () => clearTimeout(timer);
  }, []);

  return <View style={[mapStyles.frame, style]}>
    {source && !failed && <NativeWebView
      ref={webview}
      source={source}
      style={mapStyles.webview}
      originWhitelist={['*']}
      onShouldStartLoadWithRequest={({ url }) => {
        if (url === MAP_DOCUMENT || url === 'about:blank') return true;
        if (url === `${MAP_ORIGIN}/copyright`) void Linking.openURL(url).catch(() => {});
        return false;
      }}
      applicationNameForUserAgent="ConradDriverMonitor/0.2"
      nestedScrollEnabled
      setSupportMultipleWindows={false}
      thirdPartyCookiesEnabled={false}
      mixedContentMode="never"
      geolocationEnabled={false}
      onMessage={({ nativeEvent }) => {
        const data = nativeEvent.data;
        if (data === 'map-ready') setReady(true);
        else if (data === 'tiles-ready') setTiles('ok');
        else {
          try {
            const message = JSON.parse(data) as { type?: string; id?: string };
            if (message.type === 'select' && typeof message.id === 'string') onSelect?.(message.id);
          } catch { /* ignore non-JSON messages */ }
        }
      }}
      onError={() => setFailed(true)}
      onRenderProcessGone={() => setFailed(true)}
      onContentProcessDidTerminate={() => setFailed(true)}
    />}
    {(!source || !ready) && !failed && <View style={mapStyles.overlay} pointerEvents="none">
      <ActivityIndicator color={colors.brand} />
    </View>}
    {failed && <View style={mapStyles.overlay}><Text style={mapStyles.note}>Map could not start on this phone.</Text></View>}
    {tiles === 'offline' && !failed && <View style={mapStyles.chip} pointerEvents="none">
      <Text style={mapStyles.chipText}>Street map offline · positions still live</Text>
    </View>}
  </View>;
}

const mapStyles = StyleSheet.create({
  frame: { overflow: 'hidden', backgroundColor: '#DDE5E2' },
  webview: { flex: 1, backgroundColor: '#DDE5E2' },
  overlay: { ...StyleSheet.absoluteFill, alignItems: 'center', justifyContent: 'center', backgroundColor: colors.surface },
  note: { color: colors.muted, fontSize: 14 },
  chip: { position: 'absolute', top: 12, alignSelf: 'center', paddingHorizontal: 12, paddingVertical: 6,
    borderRadius: 999, backgroundColor: 'rgba(15,18,24,0.9)', borderWidth: 1, borderColor: colors.line },
  chipText: { color: colors.muted, fontSize: 12, fontWeight: '700' },
});
