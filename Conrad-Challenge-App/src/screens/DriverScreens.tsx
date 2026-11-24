import { useEffect, useState } from 'react';
import { Alert, ScrollView, Share, StyleSheet, Text, useWindowDimensions, View } from 'react-native';
import { useKeepAwake } from 'expo-keep-awake';
import CameraMonitor from '../CameraMonitor';
import { CameraPreview } from '../CameraPreview';
import type { GuardianLink } from '../guardian';
import { viewFromEngine } from '../liveView';
import type { LiveView, Tile } from '../liveView';
import { cueLabel } from '../risk';
import { formatDuration, sessionLabel } from '../trips';
import type { Preferences, SessionSource, Trip } from '../trips';
import type { useTrip } from '../useTrip';
import { Action, Body, colors, Gauge, IconButton, ListGroup, ListRow, Metric, SectionHeader, StatusStrip, styles, tint, toneColor } from '../ui';
import type { IconName } from '../ui';
import { Banner, EscalationLadder, FaceGuide, HudChip, SensorFace } from './parts';

type TripSession = ReturnType<typeof useTrip>;

export const INDICATOR_ICONS: Record<string, IconName> = {
  eyes: 'eye-outline', ocular: 'eye-refresh-outline', head: 'head-outline', breathing: 'lungs',
  responsiveness: 'hand-back-right-outline', sensor_quality: 'camera-front-variant',
};

const TITLE_ICONS: Record<string, IconName> = {
  'Eyes': 'eye-outline', 'Eye movement': 'eye-refresh-outline', 'Head position': 'head-outline',
  'Breathing': 'lungs', 'Response check': 'hand-back-right-outline', 'Camera': 'camera-front-variant',
};

export function iconForTile(tile: { key?: string; title: string }): IconName {
  return (tile.key && INDICATOR_ICONS[tile.key]) || TITLE_ICONS[tile.title] || 'circle-outline';
}

export function IndicatorList({ tiles }: { tiles: { key?: string; title: string; value: string; note: string; tone: Tile['tone'] }[] }) {
  return <ListGroup>
    {tiles.map((tile, index) => <ListRow key={tile.key ?? tile.title} icon={iconForTile(tile)} title={tile.title}
      subtitle={tile.note} value={tile.value} valueTone={tile.tone === 'safe' ? undefined : tile.tone}
      tone={tile.tone === 'off' ? 'off' : tile.tone === 'safe' ? 'brand' : tile.tone} last={index === tiles.length - 1} />)}
  </ListGroup>;
}

export function DriveHome({ prefs, trips, starting, status, fault, links, linkLabel, selectedLink, onSelectLink,
  shareLocation, setShareLocation, shareVideo, setShareVideo, setVoice, start, openTrips, openFamily,
  cameraGranted, requestCamera }: {
  prefs: Preferences; trips: Trip[]; starting: boolean; status: string; fault: string;
  links: GuardianLink[]; linkLabel: (link: GuardianLink) => string; selectedLink?: GuardianLink;
  onSelectLink: (id: string | undefined) => void; shareLocation: boolean; setShareLocation: (next: boolean) => void;
  shareVideo: boolean; setShareVideo: (next: boolean) => void; setVoice: (next: boolean) => void;
  start: (source: SessionSource) => void; openTrips: () => void; openFamily: () => void;
  cameraGranted: boolean; requestCamera: () => void;
}) {
  const { width, height } = useWindowDimensions();
  const [note, setNote] = useState('');
  const [practice, setPractice] = useState(false);
  const heroHeight = Math.min((Math.min(width, 640) - 36) * 1.05, height * 0.44);
  const shareName = selectedLink ? linkLabel(selectedLink).replace(/^Guardian · /, '').replace(/@.*/, '') : 'Share';
  const lastTrip = trips[0];

  function cycleShare() {
    setNote('');
    if (!links.length) { openFamily(); return; }
    const order = [undefined, ...links.map((link) => link.id)];
    const next = order[(order.indexOf(selectedLink?.id) + 1) % order.length];
    onSelectLink(next);
  }
  function toggleCamera() {
    if (!selectedLink) { setNote('Choose who to share this drive with first.'); return; }
    if (!selectedLink.videoSharing) { setNote('Turn on camera sharing for this parent in Family first.'); return; }
    setNote('');
    setShareVideo(!shareVideo);
  }
  function toggleLocation() {
    if (!selectedLink) { setNote('Choose who to share this drive with first.'); return; }
    if (!selectedLink.locationSharing) { setNote('Turn on location for this parent in Family first.'); return; }
    setNote('');
    setShareLocation(!shareLocation);
  }

  return <>
    <View style={{ gap: 4, paddingTop: 6 }}>
      <Text style={home.name}>{prefs.driver || 'Driver'}</Text>
      <View style={[styles.row, { justifyContent: 'flex-start', gap: 8 }]}>
        <View style={[styles.dot, { backgroundColor: cameraGranted ? colors.safe : colors.faint }]} />
        <Text style={home.status}>Parked · {cameraGranted ? 'Camera ready' : 'Camera off'}
          {selectedLink ? ` · Sharing with ${shareName}` : ''}</Text>
      </View>
    </View>

    <CameraPreview granted={cameraGranted} style={{ aspectRatio: undefined, height: heroHeight }}>
      <FaceGuide color={cameraGranted ? colors.brand : colors.faint}>
        <View style={home.heroTop}>
          <HudChip color={cameraGranted ? colors.text : colors.muted}>{cameraGranted ? 'FRONT CAMERA' : 'CAMERA OFF'}</HudChip>
          {selectedLink && shareVideo ? <HudChip color={colors.critical}>● FAMILY CAN SEE</HudChip> : null}
        </View>
        <View style={home.heroBottom}>
          {cameraGranted ? <HudChip>Center your face in the oval</HudChip>
            : <Action compact title="Turn on camera" onPress={requestCamera} />}
        </View>
      </FaceGuide>
    </CameraPreview>

    <View style={home.controls}>
      <IconButton icon="account-group-outline" label={shareName} active={Boolean(selectedLink)} onPress={cycleShare} disabled={starting} />
      <IconButton icon="cctv" label="Camera" active={Boolean(selectedLink && shareVideo)} onPress={toggleCamera} disabled={starting} tone="critical" />
      <IconButton icon="crosshairs-gps" label="Location" active={Boolean(selectedLink && shareLocation)} onPress={toggleLocation} disabled={starting} />
      <IconButton icon={prefs.voice ? 'volume-high' : 'volume-off'} label="Voice" active={prefs.voice} onPress={() => setVoice(!prefs.voice)} disabled={starting} />
    </View>
    {note ? <Body small>{note}</Body> : null}
    {fault ? <Body urgent>{fault}</Body> : null}
    {status ? <Body urgent>{status}</Body> : null}

    <Action title={starting ? 'Starting…' : 'Start drive'} onPress={() => start({ mode: 'camera' })} disabled={starting} />

    <ListGroup>
      <ListRow icon="play-circle-outline" title="Practice drills" subtitle="Parked demos of a warning and the family alert"
        value={practice ? 'Hide' : undefined} onPress={() => setPractice(!practice)} tone="brand" />
      {practice && <>
        <ListRow icon="eye-off-outline" title="Eye-closure drill" subtitle="25 s · warning, response check, family alert" onPress={() => start({ mode: 'drill', scenario: 'eye-closure' })} tone="watch" />
        <ListRow icon="head-alert-outline" title="Head-turn drill" subtitle="25 s · warning and response check" onPress={() => start({ mode: 'drill', scenario: 'head-away' })} tone="watch" />
        <ListRow icon="chart-timeline-variant" title="ML replay · eyes" subtitle="Signals exported by the Driven Python model" onPress={() => start({ mode: 'ml-replay', scenario: 'eye-closure' })} tone="brand" />
        <ListRow icon="chart-timeline-variant" title="ML replay · head" subtitle="Signals exported by the Driven Python model" onPress={() => start({ mode: 'ml-replay', scenario: 'head-away' })} tone="brand" />
      </>}
      <ListRow icon="history" title="Trips" subtitle={lastTrip ? `Last: ${sessionLabel(lastTrip.mode).toLowerCase()}, ${formatDuration(lastTrip.durationMs)}` : 'No drives yet'}
        value={String(trips.length)} onPress={openTrips} tone="brand" />
      <ListRow icon="account-group-outline" title="Family" subtitle={links.length ? `${links.length} connected` : 'Invite a parent'} onPress={openFamily} tone="brand" last />
    </ListGroup>
  </>;
}

function useElapsed() {
  const [started] = useState(() => Date.now());
  const [now, setNow] = useState(started);
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);
  const total = Math.floor((now - started) / 1000);
  return `${String(Math.floor(total / 60)).padStart(2, '0')}:${String(total % 60).padStart(2, '0')}`;
}

const IDLE_SNAPSHOT = { eyesClosedMs: 0, headAwayMs: 0, responsiveness: 'idle' as const, responseLatencyMs: null,
  responseRemainingMs: null, condition: 'none' as const, tier: 0 as const, face: null };

export function ActiveSession({ session }: { session: TripSession }) {
  useKeepAwake();
  const elapsed = useElapsed();
  const { width, height } = useWindowDimensions();
  const { run, view, coverage, gpsStatus, familyShare, videoShare, sendFrame, observe, markTrackingDelayed, stopOnFault, end } = session;
  const camera = run?.mode === 'camera';
  const replay = run?.mode === 'ml-replay';
  const live: LiveView = view ?? viewFromEngine(IDLE_SNAPSHOT, camera ? coverage : 'starting');
  const color = live.tone === 'off' ? colors.muted : toneColor(live.tone);
  const mode = replay ? 'ML REPLAY' : run?.mode === 'drill' ? 'PARKED DRILL' : 'LIVE';
  const heroHeight = Math.min((Math.min(width, 640) - 36) * 1.1, height * 0.46);
  const tile = (key: string) => live.tiles.find((item) => item.key === key);
  const hud = <>
    <View style={home.heroTop}>
      <HudChip color={mode === 'LIVE' ? colors.critical : colors.brand}>● {mode}  {elapsed}</HudChip>
      {videoShare === 'live' ? <HudChip color={colors.critical}>● FAMILY CAN SEE</HudChip>
        : familyShare === 'live' ? <HudChip color={colors.brand}>FAMILY LIVE</HudChip>
          : videoShare === 'connecting' ? <HudChip color={colors.watch}>CONNECTING</HudChip> : null}
    </View>
    <View style={[home.heroBottom, { alignItems: 'flex-start' }]}>
      <HudChip color={color}>{live.headline.toUpperCase()}</HudChip>
    </View>
  </>;
  return (
    <View style={styles.page}>
      <ScrollView contentContainerStyle={[styles.content, { paddingTop: 12 }]} removeClippedSubviews={false}>
        {camera
          ? <CameraMonitor onObservation={observe} onTrackingDelay={markTrackingDelayed} onFault={stopOnFault} onFrame={sendFrame}
              style={{ aspectRatio: undefined, height: heroHeight, borderWidth: 2, borderColor: tint(color, 0.8) }}>
              <FaceGuide color={color}>{hud}</FaceGuide>
            </CameraMonitor>
          : <View>
              <SensorFace face={live.face ?? { eyes: 'open', yaw: 0, pitch: 0 }} tone={live.tone} height={heroHeight} />
              <View style={StyleSheet.absoluteFill} pointerEvents="none">{hud}</View>
            </View>}

        {live.prompt && <Banner text={live.prompt} sub="Respond now or your family is alerted" tone="critical" />}

        <View style={[styles.row, { alignItems: 'flex-end' }]}>
          <View style={{ flex: 1, gap: 4 }}>
            <Text style={[home.condition, { color }]} numberOfLines={2}>{live.headline}</Text>
            <Text style={home.detail} numberOfLines={2}>{live.detail}</Text>
          </View>
          <Metric value={String(Math.round(live.score * 100))} label="Risk" size={52} tone={live.score >= 0.6 ? 'critical' : undefined} />
        </View>
        <Gauge value={live.score} />

        <StatusStrip items={[
          { icon: 'eye-outline', value: (tile('eyes')?.value ?? '--').split(' ')[0], label: 'EYES', tone: tile('eyes')?.tone },
          { icon: 'head-outline', value: tile('head')?.value ?? '--', label: 'HEAD', tone: tile('head')?.tone },
          { icon: 'hand-back-right-outline', value: tile('responsiveness')?.value.split(' ')[0] ?? '--', label: 'RESPONSE', tone: tile('responsiveness')?.tone },
          { icon: 'cctv', value: tile('sensor_quality')?.value ?? '--', label: 'CAMERA', tone: tile('sensor_quality')?.tone },
        ]} />

        <SectionHeader title="Escalation" />
        <EscalationLadder tier={live.tier} />
        <SectionHeader title="Risk factors" />
        <IndicatorList tiles={live.tiles} />
        {gpsStatus ? <Body small>{gpsStatus}</Body> : null}
        {!camera && <Body small>Synthetic data. Nothing leaves the phone unless a family member was selected.</Body>}
      </ScrollView>
      <View style={styles.sessionFooter}>
        <Action tone="danger" title={replay ? 'End replay' : run?.mode === 'drill' ? 'End drill' : 'End drive'} onPress={() => end()} />
      </View>
    </View>
  );
}

export function Trips({ trips, onBack }: { trips: Trip[]; onBack?: () => void }) {
  const events = trips.flatMap((trip) => trip.events);
  const tracked = trips.reduce((sum, trip) => sum + trip.trackingSamples, 0);
  const samples = trips.reduce((sum, trip) => sum + trip.samples, 0);
  return <>
    <View style={[styles.row, { paddingTop: 6 }]}>
      <Text style={home.name}>Trips</Text>
      {trips.length ? <Action compact tone="ghost" title="Export" onPress={() => {
        void Share.share({ message: JSON.stringify({ schemaVersion: 1, trips }, null, 2) }).catch(() =>
          Alert.alert('Export unavailable', 'The system share sheet could not open.'));
      }} /> : null}
    </View>
    <StatusStrip items={[
      { icon: 'car-outline', value: String(trips.length), label: 'DRIVES' },
      { icon: 'alert-octagon-outline', value: String(events.filter((e) => e.level === 1).length), label: 'WARNINGS', tone: 'watch' },
      { icon: 'account-group-outline', value: String(events.filter((e) => e.level === 2).length), label: 'FAMILY', tone: 'alert' },
      { icon: 'eye-outline', value: samples ? `${Math.round(100 * tracked / samples)}%` : '--', label: 'TRACKED' },
    ]} />
    {!trips.length ? <Body>Finished drives, drills and replays appear here.</Body> : <ListGroup>
      {trips.map((trip, index) => {
        const escalated = trip.events.some((e) => e.level >= 2);
        const tone = escalated ? 'alert' : trip.events.length ? 'watch' : 'safe';
        const first = trip.events[0];
        return <ListRow key={trip.id} icon={trip.mode === 'camera' ? 'car-outline' : trip.mode === 'drill' ? 'play-circle-outline' : 'chart-timeline-variant'}
          title={`${new Date(trip.startedAt).toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' })} · ${formatDuration(trip.durationMs)}`}
          subtitle={`${sessionLabel(trip.mode)}${first ? ` · ${cueLabel[first.cue]}` : ''}${trip.samples ? ` · ${Math.round(100 * trip.trackingSamples / trip.samples)}% tracked` : ''}`}
          value={escalated ? 'Family alert' : trip.events.length ? `${trip.events.length} alert${trip.events.length > 1 ? 's' : ''}` : 'Clean'}
          valueTone={tone === 'safe' ? 'safe' : tone} tone={tone} last={index === trips.length - 1} />;
      })}
    </ListGroup>}
    {onBack ? <Action compact tone="ghost" title="Back" onPress={onBack} /> : null}
  </>;
}

const home = StyleSheet.create({
  name: { color: colors.text, fontSize: 34, fontWeight: '600', letterSpacing: -0.8 },
  status: { color: colors.muted, fontSize: 14, fontWeight: '500' },
  heroTop: { position: 'absolute', top: 14, left: 14, right: 14, flexDirection: 'row', justifyContent: 'space-between', gap: 8 },
  heroBottom: { position: 'absolute', bottom: 14, left: 14, right: 14, alignItems: 'center' },
  controls: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 4 },
  condition: { fontSize: 24, lineHeight: 28, fontWeight: '600', letterSpacing: -0.5 },
  detail: { color: colors.muted, fontSize: 13, lineHeight: 18 },
});

