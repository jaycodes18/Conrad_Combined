import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Alert, AppState, ScrollView, StatusBar, Text, View } from 'react-native';
import { initialWindowMetrics, SafeAreaProvider, SafeAreaView } from 'react-native-safe-area-context';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { useCameraPermissions } from 'expo-camera';
import * as Location from 'expo-location';
import { FamilyScreen, useGuardian } from './src/GuardianScreen';
import { ActiveSession, DriveHome, Trips } from './src/screens/DriverScreens';
import { DriverDetail, FamilyMap, ParentAlerts } from './src/screens/ParentScreens';
import { RoleSelect, Settings } from './src/screens/Settings';
import { defaultPreferences } from './src/trips';
import type { Delivery, Preferences, Role, SessionSource, Trip } from './src/trips';
import { Body, FadeIn, Logo, Pill, styles, TabBar } from './src/ui';
import { useFamily } from './src/useFamily';
import { useTrip } from './src/useTrip';

const TRIPS_KEY = 'conrad.trips.v1';
const PROFILE_KEY = 'conrad.profile.v1';
const driverTabs = ['Drive', 'Trips', 'Family', 'Settings'] as const;
const parentTabs = ['Map', 'Alerts', 'Family', 'Settings'] as const;
type DriverTab = typeof driverTabs[number];
type ParentTab = typeof parentTabs[number];

function Brand({ role }: { role: Role | null }) {
  return <View style={styles.header}>
    <Logo size={34} />
    <Text style={[styles.brand, { flex: 1 }]}>Driven</Text>
    {role && <Pill label={role === 'driver' ? 'Teen' : 'Parent'} tone={role === 'driver' ? 'brand' : 'safe'} />}
  </View>;
}

function Main() {
  const [driverTab, setDriverTab] = useState<DriverTab>('Drive');
  const [parentTab, setParentTab] = useState<ParentTab>('Map');
  const [openDriver, setOpenDriver] = useState<string>();
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
  const [shareVideo, setShareVideo] = useState(false);
  const [permission, requestPermission] = useCameraPermissions();
  const guardian = useGuardian();
  const family = useFamily(guardian);

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

  const savePrefs = useCallback((next: Preferences) => {
    setPrefs(next);
    persist(PROFILE_KEY, next);
  }, [persist]);

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
          setStatus('Allow camera access in phone settings to start monitoring. Drills still work without it.');
          return;
        }
      }
      const requestLocation = (source.mode === 'camera' && prefs.gps) || (shareLocation && selectedLink?.locationSharing);
      const location = requestLocation ? await Location.requestForegroundPermissionsAsync().catch(() => null) : null;
      const gps = Boolean(location?.granted);
      if (requestLocation && !gps) setStatus('Location was unavailable. Monitoring runs without it.');
      if (AppState.currentState !== 'active' ||
        !session.start({ ...source, gps, shareLocation: gps && shareLocation && Boolean(selectedLink?.locationSharing),
          linkId: selectedLink?.id, voice: prefs.voice,
          shareVideo: shareVideo && Boolean(selectedLink?.videoSharing) })) {
        setStatus('Return to the app and tap Start when safely parked.');
      } else {
        setLinkId(undefined);
        setShareLocation(false);
        setShareVideo(false);
      }
    } catch {
      setStatus('Monitoring could not start. Check permissions and try again while parked.');
    } finally {
      setStarting(false);
      startLock.current = false;
    }
  }

  const cameraStatus = permission?.granted ? 'allowed' : permission?.status === 'denied' ? 'denied' : 'requested when you start';
  if (session.run) return <ActiveSession session={session} />;
  if (!loaded) return <View style={styles.page}><Brand role={null} /><View style={styles.content}><Body>Loading…</Body></View></View>;
  if (!prefs.role) {
    return <View style={styles.page}>
      <Brand role={null} />
      <ScrollView contentContainerStyle={{ flexGrow: 1 }}>
        <RoleSelect choose={(role) => savePrefs({ ...prefs, role })} />
      </ScrollView>
    </View>;
  }

  const settings = <Settings prefs={prefs} save={savePrefs} cameraStatus={cameraStatus}
    playWarning={() => session.playWarning()} trips={trips} clearTrips={() =>
      Alert.alert('Clear trip history?', 'This removes trip summaries and events from this phone.', [
        { text: 'Keep history', style: 'cancel' },
        { text: 'Clear history', style: 'destructive', onPress: () => replaceTrips([]) },
      ])} />;

  const driver = prefs.role === 'driver';
  const scrolled = (key: string, children: React.ReactNode) => <ScrollView key={key} style={styles.page}
    contentContainerStyle={styles.content} keyboardShouldPersistTaps="handled" keyboardDismissMode="on-drag">
    {storageError ? <Body urgent>{storageError}</Body> : null}
    <FadeIn style={{ gap: 14 }}>{children}</FadeIn>
  </ScrollView>;

  if (driver) {
    return <>
      <Brand role={prefs.role} />
      {scrolled('d-' + driverTab, <>
        {driverTab === 'Drive' && <DriveHome prefs={prefs} trips={trips} starting={starting} status={status}
          fault={session.fault} links={outgoing} linkLabel={guardian.linkLabel} selectedLink={selectedLink}
          onSelectLink={(id) => { setLinkId(id); setShareLocation(false); setShareVideo(false); }}
          shareLocation={shareLocation} setShareLocation={setShareLocation}
          shareVideo={shareVideo} setShareVideo={setShareVideo} setVoice={(voice) => savePrefs({ ...prefs, voice })}
          start={(source) => void start(source)} openTrips={() => setDriverTab('Trips')} openFamily={() => setDriverTab('Family')}
          cameraGranted={Boolean(permission?.granted)} requestCamera={() => { void requestPermission(); }} />}
        {driverTab === 'Trips' && <Trips trips={trips} />}
        {driverTab === 'Family' && <FamilyScreen guardian={guardian} role="driver" />}
        {driverTab === 'Settings' && settings}
      </>)}
      <TabBar items={driverTabs} active={driverTab} onChange={setDriverTab} disabled={starting}
        icons={{ Drive: 'steering', Trips: 'road-variant', Family: 'account-group-outline', Settings: 'cog-outline' }} />
    </>;
  }

  const selectedDriver = family.drivers.find((item) => item.id === openDriver);
  return <>
    <Brand role={prefs.role} />
    {parentTab === 'Map' && (selectedDriver
      ? <DriverDetail driver={selectedDriver} demo={family.demo} onBack={() => setOpenDriver(undefined)} loadTrips={family.loadTrips} />
      : <FamilyMap drivers={family.drivers} demo={family.demo} error={family.error}
          onOpenDriver={setOpenDriver} onFamily={() => setParentTab('Family')} />)}
    {parentTab === 'Alerts' && <ParentAlerts guardian={guardian} />}
    {parentTab === 'Family' && scrolled('p-family', <FamilyScreen guardian={guardian} role="guardian" />)}
    {parentTab === 'Settings' && scrolled('p-settings', settings)}
    <TabBar items={parentTabs} active={parentTab} onChange={(tab) => { setOpenDriver(undefined); setParentTab(tab); }}
      icons={{ Map: 'map-marker-radius-outline', Alerts: 'bell-ring-outline', Family: 'account-group-outline', Settings: 'cog-outline' }} />
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
