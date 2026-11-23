export const MAP_DOCUMENT = 'https://conrad-driver-monitor.local/';

export type MapPin = { latitude: number; longitude: number };

export function makeMapHtml(script: string, css: string, { latitude, longitude }: MapPin) {
  const pin = JSON.stringify([latitude, longitude]);
  return `<!doctype html>
<html lang="en"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src https://tile.openstreetmap.org data:;">
<style>
${css}
html, body, #map { width: 100%; height: 100%; margin: 0; }
#map { background: #dbe5dd; }
.leaflet-touch .leaflet-bar a { width: 44px; height: 44px; line-height: 44px; }
</style>
</head><body>
<div id="map" aria-label="Interactive map"></div>
<script>${script}</script>
<script>
  const pin = ${pin};
  const map = L.map('map', { zoomControl: false, minZoom: 2, maxZoom: 19 }).setView(pin, 14);
  L.control.zoom({ position: 'bottomright' }).addTo(map);
  L.circleMarker(pin, { radius: 9, color: '#fff', weight: 3, fillColor: '#126650', fillOpacity: 1 }).addTo(map);
  const baseMap = L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>',
    referrerPolicy: 'strict-origin-when-cross-origin',
  });
  baseMap.once('tileload', () => window.ReactNativeWebView.postMessage('tiles-ready'));
  baseMap.addTo(map);
</script>
</body></html>`;
}
