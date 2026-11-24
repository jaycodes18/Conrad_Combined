export const MAP_DOCUMENT = 'https://conrad-driver-monitor.local/';

export type MapMarker = {
  id: string;
  latitude: number;
  longitude: number;
  kind: 'car' | 'alert';
  label: string;
  initials?: string;
  /** Driver colour for cars, severity colour for alerts. */
  color: string;
  /** Status ring colour. */
  ring: string;
  heading?: number | null;
  pulse?: boolean;
};

export function markersJson(markers: MapMarker[]) {
  return JSON.stringify(markers).replace(/</g, '\\u003c');
}

/**
 * Leaflet page. The host updates it in place with `window.setMarkers(json)` and `window.focusMarker(id)`,
 * so moving cars never reload the map. A tap posts `{"type":"select","id":...}` back to the host.
 * Labels are set with textContent, never as HTML.
 */
export function makeMapHtml(script: string, css: string, markers: MapMarker[], focus: string | null = null) {
  return `<!doctype html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src https://tile.openstreetmap.org data:;">
<style>
${css}
html, body, #map { width: 100%; height: 100%; margin: 0; background: #DDE5E2; }
.leaflet-tile-pane { filter: saturate(0.75) brightness(0.96); }
.leaflet-container { font: 700 12px -apple-system, system-ui, sans-serif; background: #DDE5E2; }
.leaflet-control-attribution { background: rgba(255,255,255,0.8) !important; color: #34505D; font-size: 9px; }
.leaflet-control-attribution a { color: #12232D; }
.leaflet-div-icon { background: none; border: none; }
.car { position: relative; width: 56px; height: 56px; }
.car .pulse { position: absolute; inset: 0; border-radius: 50%; background: var(--r); opacity: 0.0; }
.car.pulse .pulse { animation: pulse 1.4s ease-out infinite; }
@keyframes pulse { 0% { transform: scale(0.55); opacity: 0.55; } 100% { transform: scale(1.35); opacity: 0; } }
.car .arrow { position: absolute; left: 50%; top: 50%; width: 0; height: 0; margin: -34px 0 0 -7px;
  border-left: 7px solid transparent; border-right: 7px solid transparent; border-bottom: 11px solid var(--r);
  transform-origin: 7px 34px; }
.car .badge { position: absolute; left: 8px; top: 8px; width: 40px; height: 40px; border-radius: 50%;
  background: var(--c); color: #09141C; display: flex; align-items: center; justify-content: center;
  font-size: 14px; font-weight: 900; letter-spacing: 0.5px; box-shadow: 0 0 0 3px #09141C, 0 0 0 6px var(--r), 0 8px 22px rgba(9,20,28,0.45); }
.car .name { position: absolute; top: 60px; left: 50%; transform: translateX(-50%); white-space: nowrap;
  background: rgba(18,35,45,0.95); color: #F5F9F9; border: 1px solid #34505D; border-radius: 999px;
  padding: 3px 9px; font-size: 11px; }
.car.focus .badge { box-shadow: 0 0 0 3px #09141C, 0 0 0 7px var(--r), 0 0 30px var(--r); }
.alert { width: 22px; height: 22px; border-radius: 50%; background: var(--c); box-shadow: 0 0 0 3px #09141C, 0 0 0 5px var(--c), 0 0 18px var(--c); }
.leaflet-marker-icon { transition: transform 0.9s linear; }
.leaflet-zoom-anim .leaflet-marker-icon, .leaflet-dragging .leaflet-marker-icon { transition: none; }
.alert.focus { transform: scale(1.35); }
</style>
</head><body>
<div id="map" aria-label="Family map"></div>
<script>${script}</script>
<script>
  const send = (data) => (window.ReactNativeWebView || { postMessage: (d) => window.parent.postMessage(d, '*') }).postMessage(data);
  const map = L.map('map', { zoomControl: false, attributionControl: true, minZoom: 3, maxZoom: 19 });
  map.setView([32.9483, -96.9895], 12);
  const layers = new Map();
  let focused = ${JSON.stringify(focus)};
  let fitted = false;

  function element(marker) {
    const root = document.createElement('div');
    root.className = marker.kind + (marker.pulse ? ' pulse' : '') + (marker.id === focused ? ' focus' : '');
    root.style.setProperty('--c', marker.color);
    root.style.setProperty('--r', marker.ring);
    if (marker.kind === 'car') {
      const pulse = document.createElement('div'); pulse.className = 'pulse'; root.appendChild(pulse);
      if (typeof marker.heading === 'number') {
        const arrow = document.createElement('div'); arrow.className = 'arrow';
        arrow.style.transform = 'rotate(' + marker.heading + 'deg)'; root.appendChild(arrow);
      }
      const badge = document.createElement('div'); badge.className = 'badge'; badge.textContent = marker.initials || '';
      root.appendChild(badge);
      const name = document.createElement('div'); name.className = 'name'; name.textContent = marker.label;
      root.appendChild(name);
    }
    return root;
  }

  function icon(marker) {
    const size = marker.kind === 'car' ? [56, 56] : [22, 22];
    return L.divIcon({ html: element(marker), className: '', iconSize: size, iconAnchor: [size[0] / 2, size[1] / 2] });
  }

  window.setMarkers = function (markers) {
    const seen = new Set();
    markers.forEach((marker) => {
      seen.add(marker.id);
      const key = JSON.stringify([marker.kind, marker.label, marker.initials, marker.color, marker.ring,
        Math.round((marker.heading || 0) / 10), marker.pulse, marker.id === focused]);
      const existing = layers.get(marker.id);
      if (existing) {
        existing.layer.setLatLng([marker.latitude, marker.longitude]);
        if (existing.key !== key) { existing.layer.setIcon(icon(marker)); existing.key = key; }
        existing.marker = marker;
      } else {
        const layer = L.marker([marker.latitude, marker.longitude], { icon: icon(marker), keyboard: false })
          .on('click', () => send(JSON.stringify({ type: 'select', id: marker.id })))
          .addTo(map);
        layers.set(marker.id, { layer, key, marker });
      }
    });
    for (const [id, entry] of layers) {
      if (!seen.has(id)) { map.removeLayer(entry.layer); layers.delete(id); }
    }
    if (!fitted && markers.length) { fitted = true; fitAll(markers); }
  };

  let userMoved = false;
  map.on('dragstart', () => { userMoved = true; });
  function fitAll(markers) {
    const list = markers || Array.from(layers.values()).map((e) => e.marker);
    if (!list.length) return;
    // Leave room for the title card above and the driver cards below, in proportion to the frame.
    const h = map.getSize().y;
    const top = h > 400 ? Math.round(h * 0.22) : 30;
    const bottom = h > 400 ? Math.round(h * 0.5) : 30;
    if (list.length === 1) map.setView([list[0].latitude, list[0].longitude], 15);
    else map.fitBounds(list.map((m) => [m.latitude, m.longitude]),
      { paddingTopLeft: [50, top], paddingBottomRight: [50, bottom], maxZoom: 15 });
  }
  // The page can start before its frame has a size; refit once it has one, and on rotation.
  setTimeout(() => { map.invalidateSize(); if (!userMoved) fitAll(); }, 350);
  window.addEventListener('resize', () => { map.invalidateSize(); if (!userMoved) fitAll(); });

  window.focusMarker = function (id) {
    focused = id;
    const entry = id ? layers.get(id) : null;
    if (entry) map.flyTo(entry.layer.getLatLng(), Math.max(map.getZoom(), 14), { duration: 0.6 });
    window.setMarkers(Array.from(layers.values()).map((e) => e.marker));
  };

  window.setMarkers(${markersJson(markers)});
  const baseMap = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    referrerPolicy: 'strict-origin-when-cross-origin',
  });
  baseMap.once('tileload', () => send('tiles-ready'));
  baseMap.addTo(map);
  send('map-ready');
</script>
</body></html>`;
}
