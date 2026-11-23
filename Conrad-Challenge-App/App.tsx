import { useCallback, useEffect, useRef, useState } from 'react';
import { Alert, AppState, Linking, Pressable, ScrollView, Share, StatusBar, Switch, Text, View } from 'react-native';
import { initialWindowMetrics, SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useCameraPermissions } from 'expo-camera';
import { useKeepAwake } from 'expo-keep-awake';
import * as Location from 'expo-location';
import * as SMS from 'expo-sms';
import CameraMonitor from './src/CameraMonitor';
import { GuardianScreen, useGuardian } from './src/GuardianScreen';
import { readLocalAsset } from './src/localAsset';
import { MapScreen } from './src/MapScreen';
import { cueLabel } from './src/risk';
import { defaultPreferences, formatDuration, sessionLabel } from './src/trips';
import type { Delivery, Preferences, SessionSource, Trip } from './src/trips';
import { Action, Body, Card, colors, Field, Heading, Label, styles } from './src/ui';
import { sessionStatus, useTrip } from './src/useTrip';

const TRIPS_KEY = 'conrad.trips.v1';
const PROFILE_KEY = 'conrad.profile.v1';
const tabs = ['Drive', 'Map', 'History', 'Guardian', 'Settings'] as const;
type Tab = typeof tabs[number];

function ActiveSession({ session }: { session: ReturnType<typeof useTrip> }) {
  useKeepAwake();
  const { run, alert, coverage, gpsStatus, observe, markTrackingDelayed, stopOnFault, end } = session;
  const drill = run?.mode === 'drill';
  const mlReplay = run?.mode === 'ml-replay';
  const camera = run?.mode === 'camera';
  return (
    <View style={styles.page}>
      <View style={styles.header}><Text style={styles.brand}>{mlReplay ? 'Driven ML replay' : drill ? 'Parked drill' : 'Camera session'}</Text></View>
      <ScrollView contentContainerStyle={styles.content} removeClippedSubviews={false}>
        <Heading>{mlReplay ? (alert ? cueLabel[alert.cue] : 'Replaying ML signals…') : sessionStatus(coverage, camera ? null : alert)}</Heading>
        {coverage === 'no-face' && <Body urgent>Monitoring is limited. Adjust the mount once safely parked.</Body>}
        {coverage === 'calibrating' && <Body>While parked, face forward with your eyes open for a few seconds.</Body>}
        {coverage === 'delayed' && <Body urgent>Waiting for fresh frames before checking for new cues.</Body>}
        {camera && <CameraMonitor onObservation={observe} onTrackingDelay={markTrackingDelayed} onFault={stopOnFault} />}
        {camera && alert && <Card>
          <Label>Latest alert · {eventTitle(alert.level)}</Label>
          <Body urgent>{cueLabel[alert.cue]}</Body>
        </Card>}
        {!camera && coverage === 'tracking' && alert && (
          <Body urgent>{mlReplay ? `Driven ML issued a warning after ${(alert.evidenceMs / 1000).toFixed(1)} s of evidence.` :
            alert.level === 3 ? 'Drill complete. No emergency call was placed.' : 'If needed, pull over safely.'}</Body>
        )}
        {gpsStatus ? <Body>{gpsStatus}</Body> : null}
      </ScrollView>
      <View style={styles.sessionFooter}>
        <Body>{mlReplay ? 'This replay uses synthetic data. The warning is saved in History.' :
          'Leave this screen open. Stop or adjust the phone only when safely parked.'}</Body>
        <Action title={mlReplay ? 'End replay' : drill ? 'End drill' : 'End session'} onPress={() => end()} />
      </View>
    </View>
  );
}

function Toggle({ label, value, onChange, disabled = false }: {
  label: string; value: boolean; onChange: (next: boolean) => void; disabled?: boolean;
}) {
  return (
    <View style={styles.row}>
      <Text style={[styles.label, { flex: 1 }]}>{label}</Text>
      <Switch accessibilityLabel={label} value={value} onValueChange={onChange}
        disabled={disabled} trackColor={{ true: colors.accent, false: colors.line }} />
    </View>
  );
}

function eventTitle(level: number) {
  if (level === 1) return 'Driver warning';
  if (level === 2) return 'Guardian test escalation';
  return 'Simulated emergency escalation';
}

function deliveryText(delivery?: Delivery) {
  if (delivery?.state === 'delivered') return 'Test alert saved to guardian inbox · ' + delivery.elapsedMs + ' ms';
  if (delivery?.state === 'pending') return 'Waiting for guardian inbox confirmation';
  if (delivery?.state === 'failed') return 'Guardian inbox confirmation failed';
  if (delivery?.state === 'unknown') return 'App closed before guardian inbox confirmation';
  return 'Recorded locally · no linked guardian delivery';
}

function Settings({ prefs, save, cameraStatus, playWarning, clearTrips, trips }: {
  prefs: Preferences; save: (p: Preferences) => void; cameraStatus: string;
  playWarning: () => void; clearTrips: () => void; trips: Trip[];
}) {
  const [driver, setDriver] = useState(prefs.driver);
  const [name, setName] = useState(prefs.contact?.name ?? '');
  const [phone, setPhone] = useState(prefs.contact?.phone ?? '');
  const [message, setMessage] = useState('');
  const [license, setLicense] = useState('');

  function saveContact() {
    const digits = phone.trim().replace(/[\s().-]/g, '');
    if (!name.trim() || !/^\+?\d{8,15}$/.test(digits)) {
      setMessage('Enter a name and a full test mobile number, including the country code if needed.');
      return;
    }
    save({ ...prefs, contact: { name: name.trim(), phone: digits } });
    setMessage('Test contact saved on this phone.');
  }

  async function testText() {
    if (!prefs.contact) return;
    try {
      if (!(await SMS.isAvailableAsync())) {
        setMessage('The system SMS composer is unavailable on this device.');
        return;
      }
      const latest = trips.find((trip) => trip.events.some((event) => event.level === 2))
        ?.events.find((event) => event.level === 2);
      const summary = latest ? ' Recorded cue: ' + cueLabel[latest.cue] + '. Time: ' + new Date(latest.at).toLocaleString() + '.' : '';
      const result = await SMS.sendSMSAsync([prefs.contact.phone],
        'Conrad driver-monitor TEST. This is a planned test message. No assistance is requested.' + summary);
      setMessage(result.result === 'cancelled' ? 'Test message cancelled.' : 'Check Messages for the status of your test text.');
    } catch {
      setMessage('The test message composer could not open.');
    }
  }

  return <>
    <Heading>Settings</Heading>
    <Card>
      <Label>Driving profile</Label>
      <Field label="Driver nickname" maxLength={48} value={driver} onChangeText={setDriver} />
      <Action title="Save profile" secondary onPress={() => {
        save({ ...prefs, driver: driver.trim() });
        setMessage('Profile saved on this phone.');
      }} />
      <Toggle label="Estimate speed with GPS" value={prefs.gps} onChange={(gps) => save({ ...prefs, gps })} />
      <Toggle label="Spoken warning after tone" value={prefs.voice} onChange={(voice) => save({ ...prefs, voice })} />
      <Action title="Test warning tone" secondary onPress={playWarning} />
      <Body>Camera access {cameraStatus}. You can change permissions in your phone settings.</Body>
      <Action title="Open phone settings" secondary onPress={() => { void Linking.openSettings().catch(() => setMessage('Open Settings on your phone.')); }} />
    </Card>
    <Card>
      <Label>Emergency contact test</Label>
      <Body>Use a consenting test contact. Sending opens Messages for you to review while parked.</Body>
      <Field label="Contact name" maxLength={64} value={name} onChangeText={setName} />
      <Field label="Mobile number" keyboardType="phone-pad" maxLength={30} value={phone} onChangeText={setPhone} />
      <Action title="Save test contact" secondary onPress={saveContact} />
      {prefs.contact && <>
        <Body>Saved contact {prefs.contact.name} · {prefs.contact.phone}</Body>
        <Action title="Compose test SMS" secondary onPress={() => void testText()} />
        <Action title="Remove test contact" secondary onPress={() => {
          save({ ...prefs, contact: null });
          setName('');
          setPhone('');
          setMessage('Test contact removed.');
        }} />
      </>}
    </Card>
    <Card>
      <Label>Data on this phone</Label>
      <Body>The app keeps trip summaries and alert timestamps on this phone. It analyzes camera samples locally and discards them. Location can be shared for a chosen guardian test session if both the session and the link allow it.</Body>
      <Action title="Clear local trip history" secondary onPress={clearTrips} />
    </Card>
    <Card>
      <Label>Third-party software</Label>
      <Body>Includes Google MediaPipe Tasks Vision, Face Landmarker, and Leaflet. Map data comes from OpenStreetMap.</Body>
      <Action title={license ? 'Hide licenses' : 'Read licenses'} secondary onPress={() => {
        if (license) { setLicense(''); return; }
        void Promise.all([
          readLocalAsset(require('./assets/mediapipe-license.txt'), 'text'),
          readLocalAsset(require('./assets/runtime/leaflet-license.txt'), 'text'),
        ]).then(([vision, map]) => setLicense(`MediaPipe\n\n${vision}\n\nLeaflet\n\n${map}`))
          .catch(() => setMessage('The bundled licenses could not be opened.'));
      }} />
      {license ? <Body>{license}</Body> : null}
    </Card>
    {message ? <Body>{message}</Body> : null}
  </>;
}

function History({ trips }: { trips: Trip[] }) {
  return <>
    <Heading>Trip history</Heading>
    <Body>{trips.length ? 'Review event timing and camera coverage while parked.' : 'Completed sessions and drills will appear here.'}</Body>
    {trips.length > 0 && <Action title="Export trip metadata" secondary onPress={() => {
      void Share.share({ message: JSON.stringify({ schemaVersion: 1, trips }, null, 2) }).catch(() =>
        Alert.alert('Export unavailable', 'The system share sheet could not open.'));
    }} />}
    {trips.map((trip) => <Card key={trip.id}>
      <Label>{sessionLabel(trip.mode)} · {formatDuration(trip.durationMs)}</Label>
      <Body>{new Date(trip.startedAt).toLocaleString()} · {trip.end === 'background' ? 'App left foreground' : trip.end === 'fault' ? 'Camera interrupted' : 'Ended'}</Body>
      {trip.mode === 'ml-replay' ? <Body>{trip.samples} replayed ML signal snapshots</Body> :
        <Body>{trip.samples} samples · {trip.trackingSamples} tracked · {trip.noFaceSamples} without face · {trip.maximumGapMs} ms longest sample gap</Body>}
      {trip.cameraTiming && <Body>
        {trip.cameraTiming.firstUsableFrameMs === null ? 'No usable frame' :
          `First usable frame in ${(trip.cameraTiming.firstUsableFrameMs / 1000).toFixed(1)} s`}
        {' · '}{trip.cameraTiming.droppedSlowFrames} slow frames dropped
        {trip.cameraTiming.longestFrameMs !== null && ` · ${trip.cameraTiming.longestFrameMs} ms longest capture to result`}
      </Body>}
      {trip.maxSpeedKmh !== null && <Body>Peak estimated speed {Math.round(trip.maxSpeedKmh)} km/h</Body>}
      {!trip.events.length && <Body>No risk alerts recorded.</Body>}
      {trip.events.map((event, index) => <View key={event.at + index}>
        <Label>{eventTitle(event.level)}</Label>
        <Body>{cueLabel[event.cue]} · {(event.elapsedMs / 1000).toFixed(1)} s into session · {(event.evidenceMs / 1000).toFixed(1)} s of evidence</Body>
      </View>)}
      {trip.delivery && <Body>{deliveryText(trip.delivery)}</Body>}
    </Card>)}
  </>;
}

function Main() {
  const [tab, setTab] = useState<Tab>('Drive');
  const [prefs, setPrefs] = useState(defaultPreferences);
  const [trips, setTrips] = useState<Trip[]>([]);
  const tripsRef = useRef<Trip[]>([]);
  const writeQueue = useRef(Promise.resolve());
  const [storageError, setStorageError] = useState('');
  const [loaded, setLoaded] = useState(false);
  const [starting, setStarting] = useState(false);
  const startLock = useRef(false);
  const [status, setStatus] = useState('');
  const [linkId, setLinkId] = useState<string>();
  const [shareLocation, setShareLocation] = useState(false);
  const [permission, requestPermission] = useCameraPermissions();
  const guardian = useGuardian();

  const persist = useCallback((key: string, value: unknown) => {
    const serialized = JSON.stringify(value);
    writeQueue.current = writeQueue.current.then(() => AsyncStorage.setItem(key, serialized)).catch(() => {
      setStorageError('The phone could not save changes to local storage.');
    });
  }, []);

  useEffect(() => {
    let active = true;
    void AsyncStorage.multiGet([PROFILE_KEY, TRIPS_KEY]).then(([[, profile], [, saved]]) => {
      if (!active) return;
      if (profile) setPrefs({ ...defaultPreferences, ...JSON.parse(profile) });
      if (saved) {
        tripsRef.current = (JSON.parse(saved) as Trip[]).map((trip) =>
          trip.delivery?.state === 'pending' ? { ...trip, delivery: { state: 'unknown' } } : trip);
        setTrips(tripsRef.current);
      }
    }).catch(() => {
      if (active) setStorageError('Saved data could not be opened.');
    }).finally(() => {
      if (active) setLoaded(true);
    });
    return () => { active = false; };
  }, []);

  const replaceTrips = useCallback((updated: Trip[]) => {
    tripsRef.current = updated;
    setTrips(updated);
    persist(TRIPS_KEY, updated);
  }, [persist]);

  const saveTrip = useCallback((trip: Trip) => {
    replaceTrips([trip, ...tripsRef.current].slice(0, 40));
  }, [replaceTrips]);

  const updateDelivery = useCallback((id: string, delivery: Delivery) => {
    if (tripsRef.current.some((trip) => trip.id === id)) {
      replaceTrips(tripsRef.current.map((trip) => trip.id === id ? { ...trip, delivery } : trip));
    }
  }, [replaceTrips]);

  const session = useTrip(saveTrip, updateDelivery);
  const outgoing = guardian.links.filter((link) => link.driverId === guardian.session?.user.id && link.status === 'accepted');
  const selectedLink = outgoing.find((link) => link.id === linkId);

  async function start(source: SessionSource) {
    if (startLock.current) return;
    startLock.current = true;
    setStarting(true);
    setStatus('');
    try {
      if (source.mode === 'ml-replay') {
        if (!session.start({ ...source, gps: false, shareLocation: false, voice: false })) {
          setStatus('Return to the app and tap Start when safely parked.');
        }
        return;
      }
      if (source.mode === 'camera') {
        const granted = permission?.granted ? permission : await requestPermission();
        if (!granted.granted) {
          setStatus('Allow camera access in phone settings to run a camera session. You can still run a parked drill.');
          return;
        }
      }
      const requestLocation = (source.mode === 'camera' && prefs.gps) || (shareLocation && selectedLink?.locationSharing);
      const location = requestLocation ? await Location.requestForegroundPermissionsAsync().catch(() => null) : null;
      const gps = Boolean(location?.granted);
      if (requestLocation && !gps) setStatus('Location was unavailable. The session can run without GPS.');
      if (AppState.currentState !== 'active' ||
        !session.start({ ...source, gps, shareLocation: gps && shareLocation && Boolean(selectedLink?.locationSharing),
          linkId: selectedLink?.id, voice: prefs.voice })) {
        setStatus('Return to the app and tap Start when safely parked.');
      } else {
        setLinkId(undefined);
        setShareLocation(false);
      }
    } catch {
      setStatus('The session could not start. Check permissions and try again while parked.');
    } finally {
      setStarting(false);
      startLock.current = false;
    }
  }

  const cameraStatus = permission?.granted ? 'allowed' : permission?.status === 'denied' ? 'denied' : 'requested when you start';
  if (session.run) return <ActiveSession session={session} />;
  return <>
    <View style={styles.header}><Text style={styles.brand}>Conrad Driver Monitor</Text></View>
    <ScrollView key={tab} style={styles.page} contentContainerStyle={styles.content}
      keyboardShouldPersistTaps="handled" keyboardDismissMode="on-drag">
      {storageError ? <Body urgent>{storageError}</Body> : null}
      {!loaded ? <Body>Loading saved data…</Body> : <>
        {tab === 'Drive' && <>
          <Heading>{prefs.driver ? 'Ready, ' + prefs.driver : 'Drive'}</Heading>
          <Card>
            <Label>Camera monitoring</Label>
            <Body>While parked, mount the phone with your face in the front camera. Keep the monitoring screen open during the test.</Body>
            {session.fault ? <Body urgent>{session.fault}</Body> : null}
            {status ? <Body urgent>{status}</Body> : null}
            {outgoing.length > 0 && <>
              <Label>Optional guardian test for the next session</Label>
              {outgoing.map((link) => <Action key={link.id} secondary title={
                (link.id === selectedLink?.id ? 'Selected · ' : 'Select · ') + guardian.linkLabel(link)
              } onPress={() => {
                setLinkId(link.id === selectedLink?.id ? undefined : link.id);
                setShareLocation(false);
              }} disabled={starting} />)}
              {selectedLink?.locationSharing && <Toggle label="Include a recent location in this test alert"
                value={shareLocation} onChange={setShareLocation} disabled={starting} />}
            </>}
            <Action title={starting ? 'Starting…' : 'Start camera session'} onPress={() => void start({ mode: 'camera' })} disabled={starting} />
          </Card>
          <Card>
            <Label>Parked drills</Label>
            <Body>Run a 25-second synthetic scenario to see the warning and guardian test escalation. The eye-closure drill also simulates emergency escalation.</Body>
            <Action title="Run eye-closure drill" secondary onPress={() => void start({ mode: 'drill', scenario: 'eye-closure' })} disabled={starting} />
            <Action title="Run head-turn drill" secondary onPress={() => void start({ mode: 'drill', scenario: 'head-away' })} disabled={starting} />
          </Card>
          <Card>
            <Label>Driven ML replay</Label>
            <Body>Play a synthetic warning generated by the Driven ML pipeline. Each replay takes about eight seconds and saves the warning in History.</Body>
            <Action title="Replay ML eye closure" secondary onPress={() => void start({ mode: 'ml-replay', scenario: 'eye-closure' })} disabled={starting} />
            <Action title="Replay ML head turn" secondary onPress={() => void start({ mode: 'ml-replay', scenario: 'head-away' })} disabled={starting} />
          </Card>
          <Card>
            <Label>Recent activity</Label>
            {trips.length ? <>
              <Body>{sessionLabel(trips[0].mode)} · {formatDuration(trips[0].durationMs)} · {trips[0].events.length} events</Body>
              <Action title="View trip history" secondary onPress={() => setTab('History')} />
            </> : <Body>Your first trip or drill will appear here.</Body>}
          </Card>
        </>}
        {tab === 'Map' && <MapScreen />}
        {tab === 'History' && <History trips={trips} />}
        {tab === 'Guardian' && <GuardianScreen guardian={guardian} />}
        {tab === 'Settings' && <Settings prefs={prefs} save={(p) => {
          setPrefs(p);
          persist(PROFILE_KEY, p);
        }} cameraStatus={cameraStatus} playWarning={() => session.playWarning()} trips={trips} clearTrips={() =>
          Alert.alert('Clear local trip history?', 'This removes trip summaries and events from this phone.', [
            { text: 'Keep history', style: 'cancel' },
            { text: 'Clear history', style: 'destructive', onPress: () => replaceTrips([]) },
          ])
        } />}
      </>}
    </ScrollView>
    <View style={styles.tabs} accessibilityRole="tablist">
      {tabs.map((item) => <Pressable key={item} style={[styles.tab, tab === item && styles.tabSelected]} disabled={starting}
        android_ripple={{ color: '#FFFFFF20' }}
        onPress={() => setTab(item)} accessibilityRole="tab" accessibilityState={{ selected: tab === item }}>
        <Text numberOfLines={1} maxFontSizeMultiplier={1.25} style={[styles.tabText, tab === item && { color: colors.accent }]}>{item}</Text>
      </Pressable>)}
    </View>
  </>;
}

export default function App() {
  return <SafeAreaProvider initialMetrics={initialWindowMetrics}>
    <SafeAreaView style={styles.page}>
      <StatusBar barStyle="light-content" />
      <Main />
    </SafeAreaView>
  </SafeAreaProvider>;
}
