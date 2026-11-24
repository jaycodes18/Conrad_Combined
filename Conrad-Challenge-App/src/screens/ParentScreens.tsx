import { useEffect, useMemo, useState } from 'react';
import { Image, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import { isLive } from '../family';
import type { FamilyDriver, TripSummary } from '../family';
import type { GuardianTestAlert, VideoFrame } from '../guardian';
import { guardianGateway } from '../guardian';
import type { GuardianState } from '../GuardianScreen';
import { guardianAlertTitle, guardianAlertTone } from '../liveView';
import type { LiveCondition, TileTone } from '../liveView';
import { LiveMap } from '../MapScreen';
import type { MapMarker } from '../MapScreen';
import { conditionLabel } from '../risk';
import { Action, Avatar, Body, colors, FadeIn, Gauge, Icon, ListGroup, ListRow, LiveDot, Metric, Pill, PressScale, SectionHeader, StatusStrip, styles, tint, toneColor } from '../ui';
import { IndicatorList } from './DriverScreens';
import { EscalationLadder, HudChip, SensorFace } from './parts';

const MINUTE = 60_000;
const VIDEO_STALE_MS = 4000;

export function ago(iso: string, now = Date.now()) {
  const seconds = Math.max(0, Math.round((now - Date.parse(iso)) / 1000));
  if (seconds < 10) return 'just now';
  if (seconds < 60) return `${seconds} s ago`;
  const minutes = Math.round(seconds / 60);
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  return `${Math.round(hours / 24)} d ago`;
}

export function conditionTone(condition: LiveCondition): TileTone {
  if (condition === 'possible_unresponsiveness') return 'critical';
  if (condition === 'possible_impairment' || condition === 'drowsy') return 'alert';
  if (condition === 'distracted') return 'watch';
  return 'safe';
}

function driverState(driver: FamilyDriver, now: number) {
  const status = driver.status;
  const live = isLive(status, now);
  if (!status) return { live, label: 'No drives shared yet', tone: 'off' as TileTone };
  if (!live) return { live, label: `Parked · last seen ${ago(status.updatedAt, now)}`, tone: 'off' as TileTone };
  return { live, label: conditionLabel[status.condition], tone: conditionTone(status.condition) };
}

export function driverMarkers(drivers: FamilyDriver[], now: number): MapMarker[] {
  return drivers.flatMap((driver) => {
    const location = driver.status?.location;
    if (!location) return [];
    const state = driverState(driver, now);
    return [{
      id: driver.id, kind: 'car' as const, latitude: location.latitude, longitude: location.longitude,
      label: state.live ? `${driver.name} · ${Math.round(driver.status?.speedKmh ?? 0)} km/h` : `${driver.name} · parked`,
      initials: driver.initials, color: driver.color,
      ring: state.tone === 'off' ? colors.faint : toneColor(state.tone),
      heading: state.live ? location.heading : null,
      pulse: state.live && (state.tone === 'alert' || state.tone === 'critical'),
    }];
  });
}

/** Live camera frames from a driver who shared video for this drive. */
export function useDriverVideo(driverId: string | null, enabled: boolean) {
  const [frame, setFrame] = useState<(VideoFrame & { receivedAt: number }) | null>(null);
  const [status, setStatus] = useState('idle');
  const [, tick] = useState(0);
  useEffect(() => {
    setFrame(null);
    if (!enabled || !driverId || !guardianGateway) { setStatus('idle'); return; }
    setStatus('connecting');
    const stop = guardianGateway.subscribeVideo(driverId, (next) => setFrame({ ...next, receivedAt: Date.now() }), setStatus);
    const timer = setInterval(() => tick((n) => n + 1), 1000);
    return () => { stop(); clearInterval(timer); };
  }, [driverId, enabled]);
  const fresh = Boolean(frame && Date.now() - frame.receivedAt <= VIDEO_STALE_MS);
  return { frame: fresh ? frame : null, status };
}

// ---- family map ---------------------------------------------------------------------------------------

function CardThumb({ driver, live, tone }: { driver: FamilyDriver; live: boolean; tone: TileTone }) {
  const { frame } = useDriverVideo(driver.id, live && driver.videoSharing && !driver.id.startsWith('demo-'));
  return <View style={parent.thumb}>
    {frame
      ? <Image source={{ uri: `data:image/jpeg;base64,${frame.jpg}` }} style={StyleSheet.absoluteFill} resizeMode="cover" fadeDuration={0} />
      : <SensorFace face={live ? driver.status?.face ?? null : null} tone={live ? tone : 'off'} height={112} compact />}
    <View style={parent.thumbChip} pointerEvents="none">
      <Text style={[parent.thumbChipText, { color: frame ? colors.critical : colors.text }]}>{frame ? '● CAMERA' : live ? 'MODEL VIEW' : 'OFFLINE'}</Text>
    </View>
  </View>;
}

function DriverCard({ driver, now, index, selected, onPress }: {
  driver: FamilyDriver; now: number; index: number; selected: boolean; onPress: () => void;
}) {
  const state = driverState(driver, now);
  const color = state.tone === 'off' ? colors.faint : toneColor(state.tone);
  const status = driver.status;
  return <FadeIn delay={120 + index * 90}>
    <PressScale onPress={onPress} accessibilityLabel={`Open ${driver.name}`}
      style={[parent.card, { borderColor: selected || (state.live && state.tone !== 'safe') ? color : colors.line }]}>
      <CardThumb driver={driver} live={state.live} tone={state.tone} />
      <View style={parent.cardBody}>
        <View style={[styles.row, { gap: 10 }]}>
          <Avatar initials={driver.initials} color={driver.color} size={30} ring={color} />
          <Text style={parent.cardName} numberOfLines={1}>{driver.name}</Text>
          {state.live ? <LiveDot color={state.tone === 'safe' ? colors.safe : color} label="" /> : null}
        </View>
        <Text style={[parent.cardState, { color: state.tone === 'off' || state.tone === 'safe' ? colors.muted : color }]} numberOfLines={1}>{state.label}</Text>
        <Gauge value={state.live ? status?.score ?? 0 : 0} height={5} />
        <View style={styles.row}>
          <View style={parent.cardStat}><Icon name="speedometer" size={15} color={colors.faint} />
            <Text style={parent.cardStatText}>{state.live && status?.speedKmh != null ? `${Math.round(status.speedKmh)} km/h` : '--'}</Text></View>
          <View style={parent.cardStat}><Icon name="gauge" size={15} color={colors.faint} />
            <Text style={parent.cardStatText}>{state.live && status ? `${Math.round(status.score * 100)} risk` : '--'}</Text></View>
          <Icon name="chevron-right" size={20} color={colors.faint} />
        </View>
      </View>
    </PressScale>
  </FadeIn>;
}

export function FamilyMap({ drivers, demo, error, onOpenDriver, onFamily }: {
  drivers: FamilyDriver[]; demo: boolean; error: string; onOpenDriver: (id: string) => void; onFamily: () => void;
}) {
  const now = Date.now();
  const markers = useMemo(() => driverMarkers(drivers, now), [drivers, now]);
  const driving = drivers.filter((driver) => isLive(driver.status, now)).length;
  const [focus, setFocus] = useState<string | null>(null);
  return <View style={{ flex: 1 }}>
    <LiveMap markers={markers} focus={focus} onSelect={onOpenDriver} style={StyleSheet.absoluteFill} />
    <FadeIn style={parent.topBar}>
      <View style={parent.titleCard}>
        <Text style={parent.mapTitle}>Family</Text>
        <Text style={parent.mapSub}>{driving ? `${driving} driving now` : 'Nobody driving'} · {drivers.length} connected</Text>
        {demo && <View style={parent.demo}><Text style={parent.demoText}>DEMO FAMILY · SIMULATED DRIVERS</Text></View>}
      </View>
    </FadeIn>
    <View style={parent.cards} pointerEvents="box-none">
      {error ? <Text style={parent.error}>{error}</Text> : null}
      {drivers.length ? <ScrollView horizontal showsHorizontalScrollIndicator={false} snapToInterval={CARD_WIDTH + 12} decelerationRate="fast"
        contentContainerStyle={{ gap: 12, paddingHorizontal: 14 }}
        onMomentumScrollEnd={({ nativeEvent }) => {
          const index = Math.round(nativeEvent.contentOffset.x / (CARD_WIDTH + 12));
          setFocus(drivers[index]?.id ?? null);
        }}>
        {drivers.map((driver, index) => <DriverCard key={driver.id} driver={driver} now={now} index={index}
          selected={focus === driver.id} onPress={() => onOpenDriver(driver.id)} />)}
      </ScrollView> : <View style={[parent.card, { width: undefined, marginHorizontal: 14, padding: 18, gap: 10 }]}>
        <Body>No connected drivers yet. Accept your teen&apos;s invitation in Family.</Body>
        <Action compact title="Open Family" onPress={onFamily} />
      </View>}
    </View>
  </View>;
}

const CARD_WIDTH = 268;

// ---- driver detail ------------------------------------------------------------------------------------

type Page = 'home' | 'factors' | 'location' | 'history';

function VideoHero({ driver, live, height }: { driver: FamilyDriver; live: boolean; height: number }) {
  const { frame, status } = useDriverVideo(driver.id, live && driver.videoSharing && !driver.id.startsWith('demo-'));
  const state = driver.status;
  const tone = live && state ? conditionTone(state.condition) : 'off';
  const color = tone === 'off' ? colors.faint : toneColor(tone);
  return <View style={[parent.hero, { height, borderColor: live ? tint(color, 0.7) : colors.lineSoft }]}>
    {frame
      ? <Image source={{ uri: `data:image/jpeg;base64,${frame.jpg}` }} style={StyleSheet.absoluteFill} resizeMode="cover" fadeDuration={0} />
      : <SensorFace face={live ? state?.face ?? null : null} tone={live ? tone : 'off'} height={height} />}
    <View style={parent.heroTop} pointerEvents="none">
      <HudChip color={frame ? colors.critical : live ? colors.brand : colors.muted}>
        {frame ? '● LIVE CAMERA' : live ? 'MODEL VIEW' : 'NOT DRIVING'}</HudChip>
      {state ? <HudChip>{ago(state.updatedAt)}</HudChip> : null}
    </View>
    {!frame && <View style={parent.heroBottom} pointerEvents="none">
      <HudChip color={colors.muted}>{driver.id.startsWith('demo-') ? 'Demo driver · no camera'
        : !driver.videoSharing ? `${driver.name} has not shared the camera with you`
          : !live ? 'Camera starts when they share a drive'
            : status === 'SUBSCRIBED' ? 'Waiting for the camera' : 'Connecting to camera'}</HudChip>
    </View>}
    {frame && live && state ? <View style={[parent.heroBottom, { alignItems: 'flex-start' }]} pointerEvents="none">
      <HudChip color={color}>{conditionLabel[state.condition].toUpperCase()}</HudChip>
    </View> : null}
  </View>;
}

const SHORT: Record<LiveCondition, string> = {
  none: 'Clean', distracted: 'Distracted', drowsy: 'Drowsy', possible_impairment: 'Impairment signs', possible_unresponsiveness: 'Critical',
};

function TripRows({ trips }: { trips: TripSummary[] }) {
  return <ListGroup>
    {trips.map((trip, index) => {
      const tone = trip.worstCondition === 'none' ? (trip.driverAlerts ? 'watch' : 'safe') : conditionTone(trip.worstCondition);
      const start = new Date(trip.startedAt);
      return <ListRow key={trip.sessionId} icon="car-outline"
        title={start.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' })}
        subtitle={`${start.toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' })} · ${Math.round(trip.durationS / 60)} min${trip.driverAlerts ? ` · ${trip.driverAlerts} warning${trip.driverAlerts > 1 ? 's' : ''}` : ''}`}
        value={trip.worstCondition === 'none' && trip.driverAlerts ? 'Brief alerts' : SHORT[trip.worstCondition]}
        valueTone={tone} tone={tone} last={index === trips.length - 1} />;
    })}
  </ListGroup>;
}

export function DriverDetail({ driver, demo, onBack, loadTrips }: {
  driver: FamilyDriver; demo: boolean; onBack: () => void; loadTrips: (id: string) => void;
}) {
  const [page, setPage] = useState<Page>('home');
  useEffect(() => { loadTrips(driver.id); }, [driver.id, loadTrips]);
  const now = Date.now();
  const status = driver.status;
  const state = driverState(driver, now);
  const color = state.tone === 'off' ? colors.muted : toneColor(state.tone);
  const week = driver.trips.filter((trip) => now - Date.parse(trip.startedAt) <= 7 * 24 * 60 * MINUTE);
  const clean = week.filter((trip) => trip.worstCondition === 'none' && !trip.driverAlerts).length;
  const minutes = Math.round(week.reduce((sum, trip) => sum + trip.durationS, 0) / 60);
  const markers = driverMarkers([driver], now);
  const tile = (title: string) => status?.indicators.find((item) => item.title === title);
  const flagged = (status?.indicators ?? []).filter((item) => item.tone === 'alert' || item.tone === 'critical' || item.tone === 'watch').length;

  const back = <Pressable accessibilityRole="button" onPress={page === 'home' ? onBack : () => setPage('home')} style={parent.back}>
    <Icon name="chevron-left" size={22} color={colors.text} />
    <Text style={parent.backText}>{page === 'home' ? 'Family' : driver.name}</Text>
    <View style={{ flex: 1 }} />
    {demo ? <Pill label="Demo" tone="watch" /> : null}
  </Pressable>;

  if (page === 'factors') {
    return <ScrollView style={styles.page} contentContainerStyle={styles.content}>
      {back}
      <Text style={parent.pageTitle}>Risk factors</Text>
      {state.live && status?.indicators.length ? <IndicatorList tiles={status.indicators} />
        : <Body>Risk factors appear while {driver.name} is driving.</Body>}
      <Body small>Driven reports observable driving-risk signs. It does not measure alcohol or identify a medical condition.</Body>
    </ScrollView>;
  }
  if (page === 'location') {
    return <ScrollView style={styles.page} contentContainerStyle={styles.content}>
      {back}
      <Text style={parent.pageTitle}>Location</Text>
      {markers.length ? <LiveMap markers={markers} focus={driver.id} style={{ height: 420, borderRadius: 24 }} />
        : <Body>{driver.locationSharing ? `${driver.name} has not shared a location on this drive.` : `${driver.name} has not turned on location for you.`}</Body>}
      {status?.location ? <Body small>{status.location.latitude.toFixed(4)}, {status.location.longitude.toFixed(4)} · {ago(status.updatedAt, now)}</Body> : null}
    </ScrollView>;
  }
  if (page === 'history') {
    return <ScrollView style={styles.page} contentContainerStyle={styles.content}>
      {back}
      <Text style={parent.pageTitle}>Driving history</Text>
      <StatusStrip items={[
        { icon: 'car-outline', value: String(week.length), label: 'THIS WEEK' },
        { icon: 'check-circle-outline', value: week.length ? `${clean}/${week.length}` : '--', label: 'CLEAN', tone: 'safe' },
        { icon: 'history', value: String(minutes), label: 'MINUTES' },
      ]} />
      {driver.trips.length ? <TripRows trips={driver.trips} /> : <Body>No shared drives yet.</Body>}
    </ScrollView>;
  }

  return <ScrollView style={styles.page} contentContainerStyle={styles.content}>
    {back}
    <FadeIn style={{ gap: 14 }}>
    <View style={[styles.row, { justifyContent: 'flex-start', gap: 14 }]}>
      <Avatar initials={driver.initials} color={driver.color} size={50} ring={color} />
      <View style={{ flex: 1, gap: 4 }}>
        <Text style={parent.name}>{driver.name}</Text>
        <View style={[styles.row, { justifyContent: 'flex-start', gap: 10 }]}>
          {state.live ? <LiveDot color={state.tone === 'safe' ? colors.safe : color} label="DRIVING" /> : <Text style={parent.parked}>PARKED</Text>}
          {status ? <Text style={parent.updated}>updated {ago(status.updatedAt, now)}</Text> : null}
        </View>
      </View>
    </View>

    <VideoHero driver={driver} live={state.live} height={300} />

    <View style={[styles.row, { alignItems: 'flex-end' }]}>
      <View style={{ flex: 1, gap: 4 }}>
        <Text style={[parent.condition, { color }]} numberOfLines={2}>{state.live ? state.label : 'Not driving'}</Text>
        <Text style={parent.updated}>{state.live ? 'Current condition' : status ? `Last drive ${ago(status.updatedAt, now)}` : 'Waiting for a shared drive'}</Text>
      </View>
      <Metric value={state.live && status ? String(Math.round(status.score * 100)) : '--'} label="Risk" size={52}
        tone={state.live && status && status.score >= 0.6 ? 'critical' : undefined} />
    </View>
    <Gauge value={state.live ? status?.score ?? 0 : 0} />

    <StatusStrip items={[
      { icon: 'speedometer', value: state.live && status?.speedKmh != null ? `${Math.round(status.speedKmh)}` : '--', label: 'KM/H' },
      { icon: 'eye-outline', value: state.live ? (tile('Eyes')?.value ?? '--').split(' ')[0] : '--', label: 'EYES', tone: state.live ? tile('Eyes')?.tone : undefined },
      { icon: 'head-outline', value: state.live ? tile('Head position')?.value ?? '--' : '--', label: 'HEAD', tone: state.live ? tile('Head position')?.tone : undefined },
      { icon: 'hand-back-right-outline', value: state.live ? (tile('Response check')?.value ?? '--').split(' ')[0] : '--', label: 'RESPONSE', tone: state.live ? tile('Response check')?.tone : undefined },
    ]} />

    <SectionHeader title="Escalation" />
    <EscalationLadder tier={state.live ? status?.tier ?? 0 : 0} />

    <ListGroup>
      <ListRow icon="heart-pulse" title="Risk factors" subtitle="Eyes, eye movement, head, breathing, response, camera"
        value={state.live ? (flagged ? `${flagged} flagged` : 'All clear') : 'Offline'} valueTone={flagged ? 'alert' : undefined}
        onPress={() => setPage('factors')} tone="brand" />
      <ListRow icon="crosshairs-gps" title="Location" subtitle={driver.locationSharing ? 'Shared on alerts and live drives' : 'Not shared with you'}
        value={markers.length ? (state.live ? 'Live' : 'Last known') : 'Off'} onPress={() => setPage('location')} tone="brand" />
      <ListRow icon="history" title="Driving history" subtitle={`${week.length} drives this week · ${minutes} min`}
        value={week.length ? `${clean}/${week.length} clean` : undefined} onPress={() => setPage('history')} tone="brand" />
      <ListRow icon="cctv" title="Camera" subtitle={driver.videoSharing ? 'Live during drives they share with you' : 'They have not allowed camera sharing'}
        value={driver.videoSharing ? 'Allowed' : 'Off'} tone={driver.videoSharing ? 'critical' : 'off'} last />
    </ListGroup>
    </FadeIn>
  </ScrollView>;
}

// ---- alerts ------------------------------------------------------------------------------------------

export type ParentAlert = GuardianTestAlert & { sample?: true };

export function sampleAlerts(now = Date.now()): ParentAlert[] {
  const base = { linkId: 'sample', level: 2 as const, isTest: true as const, sample: true as const };
  return [
    { ...base, id: 'sample-1', sessionId: 'sample-a', cue: 'eye-closure', evidenceMs: 6100,
      locationLat: 32.9300, locationLng: -96.9500, createdAt: new Date(now - 14 * MINUTE).toISOString() },
    { ...base, id: 'sample-2', sessionId: 'sample-b', cue: 'head-away', evidenceMs: 7600,
      locationLat: 32.9545, locationLng: -96.9950, createdAt: new Date(now - 26 * 60 * MINUTE).toISOString() },
    { ...base, id: 'sample-3', sessionId: 'sample-c', cue: 'eye-closure', evidenceMs: 3200,
      locationLat: null, locationLng: null, createdAt: new Date(now - 3 * 24 * 60 * MINUTE).toISOString() },
  ];
}

export function ParentAlerts({ guardian }: { guardian: GuardianState }) {
  const live = Boolean(guardianGateway && guardian.session);
  const alerts = useMemo<ParentAlert[]>(() => live
    ? [...guardian.inbox].sort((a, b) => b.createdAt.localeCompare(a.createdAt)) : sampleAlerts(), [live, guardian.inbox]);
  const [selected, setSelected] = useState<string | null>(null);
  const markers: MapMarker[] = alerts.flatMap((alert) => alert.locationLat !== null && alert.locationLng !== null ? [{
    id: alert.id, kind: 'alert' as const, latitude: alert.locationLat, longitude: alert.locationLng,
    label: guardianAlertTitle(alert.cue, alert.evidenceMs),
    color: toneColor(guardianAlertTone(alert.cue, alert.evidenceMs)), ring: colors.bg,
  }] : []);
  return <ScrollView style={styles.page} contentContainerStyle={styles.content}>
    <View style={[styles.row, { paddingTop: 6 }]}>
      <Text style={parent.name}>Alerts</Text>
      {live ? <Action compact tone="ghost" title="Refresh" onPress={() => void guardian.refresh()} /> : <Pill label="Sample data" tone="watch" />}
    </View>
    {markers.length > 0 && <LiveMap markers={markers} focus={selected} onSelect={setSelected} style={{ height: 240, borderRadius: 24 }} />}
    {!alerts.length ? <Body>No alerts yet. Alerts arrive when a teen does not respond to a warning or keeps their eyes closed for 5 seconds.</Body>
      : <ListGroup>
        {alerts.map((alert, index) => {
          const tone = guardianAlertTone(alert.cue, alert.evidenceMs);
          const located = alert.locationLat !== null;
          return <ListRow key={alert.id} icon={alert.cue === 'eye-closure' ? 'eye-off-outline' : 'head-alert-outline'}
            title={guardianAlertTitle(alert.cue, alert.evidenceMs)}
            subtitle={`${alert.cue === 'eye-closure' ? 'Eyes closed' : 'Head away'} ${(alert.evidenceMs / 1000).toFixed(1)} s · ${ago(alert.createdAt)}${located ? ' · on map' : ''}`}
            tone={tone}
            onPress={located ? () => setSelected(alert.id === selected ? null : alert.id) : undefined}
            last={index === alerts.length - 1} />;
        })}
      </ListGroup>}
    <Body small>Driven reports observable driving-risk signs. It does not measure alcohol or identify a medical condition. In an emergency, call 911 yourself.</Body>
  </ScrollView>;
}

const parent = StyleSheet.create({
  topBar: { position: 'absolute', top: 12, left: 14, right: 14 },
  titleCard: { alignSelf: 'flex-start', paddingHorizontal: 16, paddingVertical: 12, borderRadius: 22, gap: 2,
    backgroundColor: 'rgba(18,35,45,0.94)', borderWidth: 1, borderColor: colors.line },
  mapTitle: { color: colors.text, fontSize: 28, fontWeight: '700', letterSpacing: -0.6 },
  mapSub: { color: colors.muted, fontSize: 13, fontWeight: '600' },
  cards: { position: 'absolute', left: 0, right: 0, bottom: 12, gap: 8 },
  card: { width: 268, borderRadius: 24, overflow: 'hidden', backgroundColor: colors.surface, borderWidth: 1.5,
    shadowColor: '#000', shadowOpacity: 0.35, shadowRadius: 16, shadowOffset: { width: 0, height: 8 }, elevation: 8 },
  thumb: { height: 112, backgroundColor: '#0D1A22' },
  thumbChip: { position: 'absolute', top: 8, left: 8, paddingHorizontal: 8, paddingVertical: 4, borderRadius: 999, backgroundColor: 'rgba(9,20,28,0.78)' },
  thumbChipText: { fontSize: 10, fontWeight: '800', letterSpacing: 0.8 },
  cardBody: { padding: 14, gap: 9 },
  cardName: { color: colors.text, fontSize: 18, fontWeight: '700', flex: 1 },
  cardState: { fontSize: 14, fontWeight: '600' },
  cardStat: { flexDirection: 'row', alignItems: 'center', gap: 5 },
  cardStatText: { color: colors.muted, fontSize: 13, fontWeight: '600' },
  demo: { alignSelf: 'flex-start', marginTop: 8, paddingHorizontal: 10, paddingVertical: 5, borderRadius: 999,
    backgroundColor: tint(colors.watch, 0.18), borderWidth: 1, borderColor: tint(colors.watch, 0.5) },
  demoText: { color: colors.watch, fontSize: 10, fontWeight: '700', letterSpacing: 1 },
  sheet: { position: 'absolute', left: 10, right: 10, bottom: 10, paddingTop: 8, paddingBottom: 4,
    backgroundColor: 'rgba(12,19,34,0.96)', borderRadius: 26, borderWidth: 1, borderColor: colors.line, overflow: 'hidden' },
  grabber: { alignSelf: 'center', width: 40, height: 4, borderRadius: 2, backgroundColor: colors.line, marginBottom: 4 },
  driverRow: { flexDirection: 'row', alignItems: 'center', gap: 12, paddingHorizontal: 16, paddingVertical: 12 },
  driverRowLine: { borderTopWidth: StyleSheet.hairlineWidth, borderColor: colors.line },
  driverName: { color: colors.text, fontSize: 18, fontWeight: '600' },
  driverState: { fontSize: 13, fontWeight: '500' },
  error: { color: colors.watch, fontSize: 12, fontWeight: '600', paddingHorizontal: 16, paddingBottom: 6 },
  back: { flexDirection: 'row', alignItems: 'center', gap: 4, paddingVertical: 4 },
  backText: { color: colors.text, fontSize: 16, fontWeight: '500' },
  name: { color: colors.text, fontSize: 34, fontWeight: '600', letterSpacing: -0.8 },
  pageTitle: { color: colors.text, fontSize: 30, fontWeight: '600', letterSpacing: -0.6 },
  parked: { color: colors.faint, fontSize: 11, fontWeight: '700', letterSpacing: 1.2 },
  updated: { color: colors.muted, fontSize: 13, fontWeight: '500' },
  hero: { borderRadius: 28, overflow: 'hidden', borderWidth: 2, backgroundColor: '#0A0D12' },
  heroTop: { position: 'absolute', top: 14, left: 14, right: 14, flexDirection: 'row', justifyContent: 'space-between' },
  heroBottom: { position: 'absolute', bottom: 14, left: 14, right: 14, alignItems: 'center' },
  condition: { fontSize: 24, lineHeight: 28, fontWeight: '600', letterSpacing: -0.5 },
});
