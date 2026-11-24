import { useState } from 'react';
import { Alert, Linking, Pressable, Switch, Text, View } from 'react-native';
import * as SMS from 'expo-sms';
import { readLocalAsset } from '../localAsset';
import { cueLabel } from '../risk';
import type { Preferences, Role, Trip } from '../trips';
import { Action, Body, Card, colors, Field, Heading, Label, Logo, ScreenTitle, SectionHeader, styles, tint } from '../ui';

function Toggle({ label, value, onChange }: { label: string; value: boolean; onChange: (next: boolean) => void }) {
  return <View style={styles.row}>
    <Text style={[styles.label, { flex: 1, fontSize: 15 }]}>{label}</Text>
    <Switch accessibilityLabel={label} value={value} onValueChange={onChange} trackColor={{ true: colors.brand, false: colors.line }} />
  </View>;
}

export function Settings({ prefs, save, cameraStatus, playWarning, clearTrips, trips }: {
  prefs: Preferences; save: (p: Preferences) => void; cameraStatus: string;
  playWarning: () => void; clearTrips: () => void; trips: Trip[];
}) {
  const driverRole = prefs.role === 'driver';
  const [driver, setDriver] = useState(prefs.driver);
  const [name, setName] = useState(prefs.contact?.name ?? '');
  const [phone, setPhone] = useState(prefs.contact?.phone ?? '');
  const [message, setMessage] = useState('');
  const [license, setLicense] = useState('');

  function saveContact() {
    const digits = phone.trim().replace(/[\s().-]/g, '');
    if (!name.trim() || !/^\+?\d{8,15}$/.test(digits)) {
      setMessage('Enter a name and a full mobile number, including the country code if needed.');
      return;
    }
    save({ ...prefs, contact: { name: name.trim(), phone: digits } });
    setMessage('Contact saved on this phone.');
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
        'Driven TEST. This is a planned test message. No assistance is requested.' + summary);
      setMessage(result.result === 'cancelled' ? 'Test message cancelled.' : 'Check Messages for the status of your test text.');
    } catch {
      setMessage('The test message composer could not open.');
    }
  }

  const other: Role = driverRole ? 'guardian' : 'driver';
  return <>
    <ScreenTitle eyebrow={driverRole ? 'Teen driver' : 'Parent'} title="Settings" />
    {driverRole && <>
      <SectionHeader title="Driving" />
      <Card>
        <Field label="Nickname" maxLength={48} value={driver} onChangeText={setDriver} />
        <Action compact secondary title="Save nickname" onPress={() => {
          save({ ...prefs, driver: driver.trim() });
          setMessage('Nickname saved.');
        }} />
        <Toggle label="Estimate speed with GPS" value={prefs.gps} onChange={(gps) => save({ ...prefs, gps })} />
        <Toggle label="Spoken warning after the tone" value={prefs.voice} onChange={(voice) => save({ ...prefs, voice })} />
        <Action compact secondary title="Play warning tone" onPress={playWarning} />
        <Body small>Camera access {cameraStatus}.</Body>
        <Action compact tone="ghost" title="Open phone settings" onPress={() => { void Linking.openSettings().catch(() => setMessage('Open Settings on your phone.')); }} />
      </Card>
      <SectionHeader title="Emergency contact test" />
      <Card>
        <Body small>Use a contact who agreed to receive test texts. Sending opens Messages so you review it while parked.</Body>
        <Field label="Contact name" maxLength={64} value={name} onChangeText={setName} />
        <Field label="Mobile number" keyboardType="phone-pad" maxLength={30} value={phone} onChangeText={setPhone} />
        <Action compact secondary title="Save contact" onPress={saveContact} />
        {prefs.contact && <>
          <Body small>Saved: {prefs.contact.name} · {prefs.contact.phone}</Body>
          <Action compact secondary title="Compose test text" onPress={() => void testText()} />
          <Action compact tone="ghost" title="Remove contact" onPress={() => {
            save({ ...prefs, contact: null });
            setName('');
            setPhone('');
            setMessage('Contact removed.');
          }} />
        </>}
      </Card>
    </>}
    <SectionHeader title="Privacy" />
    <Card flat>
      <Body small>Camera frames are analyzed on this phone and discarded. The app keeps trip summaries and alert times. A location goes with a guardian alert only when the link and that drive both allow it. The microphone is never used.</Body>
      {driverRole && <Action compact tone="danger" title="Clear trip history" onPress={clearTrips} />}
    </Card>
    <SectionHeader title="This phone" />
    <Card flat>
      <Body small>This phone is set up for the {driverRole ? 'teen driver' : 'parent'}.</Body>
      <Action compact secondary title={`Switch to ${other === 'driver' ? 'teen driver' : 'parent'} mode`} onPress={() =>
        Alert.alert('Switch mode?', `This phone will open the ${other === 'driver' ? 'teen driver' : 'parent'} screens.`, [
          { text: 'Cancel', style: 'cancel' },
          { text: 'Switch', onPress: () => save({ ...prefs, role: other }) },
        ])} />
    </Card>
    <SectionHeader title="Software" />
    <Card flat>
      <Body small>Includes Google MediaPipe Tasks Vision, Face Landmarker and Leaflet. Map data from OpenStreetMap.</Body>
      <Action compact tone="ghost" title={license ? 'Hide licenses' : 'Read licenses'} onPress={() => {
        if (license) { setLicense(''); return; }
        void Promise.all([
          readLocalAsset(require('../../assets/mediapipe-license.txt'), 'text'),
          readLocalAsset(require('../../assets/runtime/leaflet-license.txt'), 'text'),
        ]).then(([vision, map]) => setLicense(`MediaPipe\n\n${vision}\n\nLeaflet\n\n${map}`))
          .catch(() => setMessage('The bundled licenses could not be opened.'));
      }} />
      {license ? <Body small>{license}</Body> : null}
    </Card>
    {message ? <Body>{message}</Body> : null}
  </>;
}

function RoleCard({ title, body, points, onPress }: { title: string; body: string; points: string[]; onPress: () => void }) {
  return <Pressable accessibilityRole="button" onPress={onPress}
    style={({ pressed }) => [styles.card, { gap: 10 }, pressed && { borderColor: colors.brand, backgroundColor: tint(colors.brand, 0.08) }]}>
    <Label>{title}</Label>
    <Body>{body}</Body>
    {points.map((point) => <View key={point} style={[styles.row, { justifyContent: 'flex-start' }]}>
      <View style={[styles.dot, { backgroundColor: colors.brand }]} />
      <Body small>{point}</Body>
    </View>)}
  </Pressable>;
}

export function RoleSelect({ choose }: { choose: (role: Role) => void }) {
  return <View style={[styles.content, { flex: 1, justifyContent: 'center' }]}>
    <View style={{ gap: 8, marginBottom: 8 }}>
      <Logo size={64} />
      <Heading>Who uses this phone?</Heading>
      <Body>Teens and parents get different screens. You can switch later in Settings.</Body>
    </View>
    <RoleCard title="I’m the driver" body="The phone on the dash. Watches for eye closure and head turns and warns you out loud."
      points={['Front-camera monitoring with a response check', 'Choose who sees each drive', 'Your trip history']}
      onPress={() => choose('driver')} />
    <RoleCard title="I’m a parent or guardian" body="Your own phone. See each teen on a live map, then tap their car for the live camera, risk and driving history."
      points={['Live family map', 'Per-driver risk and condition', 'Driving history and alerts']}
      onPress={() => choose('guardian')} />
  </View>;
}
