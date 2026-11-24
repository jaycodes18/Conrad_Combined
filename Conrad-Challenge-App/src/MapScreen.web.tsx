// Browser version of LiveMap for the web parent dashboard (`npm run web`): the same Leaflet page, in an iframe.
import React, { useEffect, useRef, useState } from 'react';
import { View } from 'react-native';
import type { StyleProp, ViewStyle } from 'react-native';
import { Asset } from 'expo-asset';
import { makeMapHtml, markersJson } from './mapHtml';
import type { MapMarker } from './mapHtml';
export type { MapMarker };
const LEAFLET_SCRIPT = require('../assets/runtime/leaflet-script.txt');
const LEAFLET_STYLE = require('../assets/runtime/leaflet-style.txt');
async function text(mod: number) { const a = Asset.fromModule(mod); return (await fetch(a.uri)).text(); }
export function LiveMap({ markers, focus = null, onSelect, style }: { markers: MapMarker[]; focus?: string | null; onSelect?: (id: string) => void; style?: StyleProp<ViewStyle> }) {
  const frame = useRef<HTMLIFrameElement | null>(null);
  const [html, setHtml] = useState<string>();
  const initial = useRef({ markers, focus });
  const [ready, setReady] = useState(false);
  useEffect(() => { void Promise.all([text(LEAFLET_SCRIPT), text(LEAFLET_STYLE)]).then(([s, c]) => setHtml(makeMapHtml(s, c, initial.current.markers, initial.current.focus))); }, []);
  useEffect(() => {
    const on = (e: MessageEvent) => { if (e.source !== frame.current?.contentWindow) return; if (e.data === 'map-ready') setReady(true);
      try { const m = JSON.parse(e.data); if (m.type === 'select') onSelect?.(m.id); } catch {} };
    window.addEventListener('message', on); return () => window.removeEventListener('message', on);
  }, [onSelect]);
  useEffect(() => { if (ready) (frame.current?.contentWindow as any)?.setMarkers(JSON.parse(markersJson(markers))); }, [markers, ready]);
  useEffect(() => { if (ready) (frame.current?.contentWindow as any)?.focusMarker(focus); }, [focus, ready]);
  return <View style={[{ overflow: 'hidden', backgroundColor: '#DDE5E2' }, style]}>
    {html ? React.createElement('iframe', { ref: frame, srcDoc: html, style: { border: 0, width: '100%', height: '100%' } }) : null}
  </View>;
}
