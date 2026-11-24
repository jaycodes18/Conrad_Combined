import { useCallback, useEffect, useRef, useState } from 'react';
import { Alert, AppState, Text, View } from 'react-native';
import type { Session } from '@supabase/supabase-js';
import { guardianGateway } from './guardian';
import type { GuardianLink, GuardianTestAlert } from './guardian';
import type { Role } from './trips';
import { Action, Body, Card, colors, Field, Label, Pill, ScreenTitle, SectionHeader, styles } from './ui';

export function useGuardian() {
  const [session, setSession] = useState<Session | null>(null);
  const [links, setLinks] = useState<GuardianLink[]>([]);
  const [inbox, setInbox] = useState<GuardianTestAlert[]>([]);
  const [error, setError] = useState('');
  const userId = session?.user.id;
  const activeUser = useRef<string | undefined>(undefined);
  const generation = useRef(0);

  const refresh = useCallback(async () => {
    if (!guardianGateway || !userId || activeUser.current !== userId) return;
    const request = ++generation.current;
    try {
      const [nextLinks, nextInbox] = await Promise.all([guardianGateway.listLinks(), guardianGateway.listTestAlerts()]);
      if (activeUser.current !== userId || generation.current !== request) return;
      setLinks(nextLinks);
      setInbox(nextInbox);
      setError('');
    } catch {
      if (activeUser.current === userId && generation.current === request) {
        setError('Could not refresh guardian data. Check the connection and try again.');
      }
    }
  }, [userId]);

  useEffect(() => {
    if (!guardianGateway) return;
    return guardianGateway.onAuthStateChange((next) => {
      if (activeUser.current !== next?.user.id) {
        activeUser.current = next?.user.id;
        generation.current += 1;
        setLinks([]);
        setInbox([]);
        setError('');
      }
      setSession(next);
    });
  }, []);

  useEffect(() => {
    if (!guardianGateway || !userId) return;
    void refresh();
    const unsubscribe = guardianGateway.subscribeToInbox(() => {
      if (AppState.currentState === 'active') void refresh();
    });
    const foreground = AppState.addEventListener('change', (state) => {
      if (state === 'active') void refresh();
    });
    return () => {
      unsubscribe();
      foreground.remove();
      generation.current += 1;
    };
  }, [refresh, userId]);

  const linkLabel = useCallback((link: GuardianLink) => {
    const role = link.driverId === userId ? 'Guardian' : 'Driver';
    return link.otherEmail ? role + ' · ' + link.otherEmail : role + ' link ' + link.id.slice(0, 8).toUpperCase();
  }, [userId]);

  return { session, links, inbox, error, refresh, linkLabel };
}

export type GuardianState = ReturnType<typeof useGuardian>;

function useTask() {
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);
  const lock = useRef(false);
  const perform = useCallback(async (task: () => Promise<string>) => {
    if (lock.current) return;
    lock.current = true;
    setBusy(true);
    setMessage('');
    try {
      setMessage(await task());
    } catch (reason) {
      setMessage(reason instanceof Error ? reason.message : 'The request could not be completed.');
    } finally {
      lock.current = false;
      setBusy(false);
    }
  }, []);
  return { message, busy, perform };
}

const COPY: Record<Role, { title: string; subtitle: string; signIn: string }> = {
  driver: {
    title: 'Family',
    subtitle: 'Invite a parent or guardian. They see alerts only after they accept, and only what you allow.',
    signIn: 'Sign in so a parent can receive your alerts on their own phone.',
  },
  guardian: {
    title: 'Family',
    subtitle: 'Accept your teen’s invitation to receive their alerts on this phone.',
    signIn: 'Sign in with the email your teen invited.',
  },
};

function SignIn({ role }: { role: Role }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const { message, busy, perform } = useTask();
  const api = guardianGateway;
  if (!api) return null;
  const valid = email.includes('@') && password.length >= 8;
  return <Card>
    <Label>Account</Label>
    <Body>{COPY[role].signIn}</Body>
    <Field label="Email" value={email} onChangeText={setEmail} autoCapitalize="none" autoComplete="email" keyboardType="email-address" />
    <Field label="Password (8+ characters)" value={password} onChangeText={setPassword} secureTextEntry autoCapitalize="none" />
    <Action title="Sign in" disabled={busy || !valid} onPress={() => void perform(async () => {
      await api.signIn(email, password);
      setPassword('');
      return 'Signed in.';
    })} />
    <Action title="Create account" secondary disabled={busy || !valid} onPress={() => void perform(async () => {
      const { needsEmailVerification } = await api.signUp(email, password);
      setPassword('');
      return needsEmailVerification ? 'Check your email for a confirmation link, then sign in here.' : 'Account created.';
    })} />
    {message ? <Body small>{message}</Body> : null}
  </Card>;
}

function LinkCard({ link, guardian, role, busy, perform }: {
  link: GuardianLink; guardian: GuardianState; role: Role; busy: boolean;
  perform: (task: () => Promise<string>) => Promise<void>;
}) {
  const api = guardianGateway;
  if (!api) return null;
  const pending = link.status === 'pending';
  const own = link.driverId === guardian.session?.user.id;
  const who = link.otherEmail || 'Link ' + link.id.slice(0, 8).toUpperCase();
  return <Card flat>
    <View style={styles.row}>
      <View style={{ flex: 1, gap: 4 }}>
        <Text style={{ color: colors.text, fontWeight: '700', fontSize: 15 }} numberOfLines={1}>{who}</Text>
        <Body small>{own ? 'Your guardian' : 'Your driver'} · since {new Date(link.acceptedAt ?? link.createdAt).toLocaleDateString()}</Body>
      </View>
      <Pill label={pending ? 'Pending' : 'Connected'} tone={pending ? 'watch' : 'safe'} />
    </View>
    {!pending && <Pill label={link.locationSharing ? 'Location shared on alerts' : 'Location not shared'}
      tone={link.locationSharing ? 'brand' : 'off'} />}
    {pending && !own && <>
      <Body small>Confirm code {link.id.slice(0, 8).toUpperCase()} with your teen before accepting.</Body>
      <Action title="Accept invitation" disabled={busy} onPress={() => void perform(async () => {
        await api.acceptInvite(link.id);
        await guardian.refresh();
        return 'Invitation accepted. Alerts will appear on Home.';
      })} />
    </>}
    {pending && own && <Body small>Waiting for the guardian to accept code {link.id.slice(0, 8).toUpperCase()}.</Body>}
    {own && !pending && role === 'driver' && <Action compact secondary
      title={link.locationSharing ? 'Stop sharing location on alerts' : 'Allow location on alerts'}
      disabled={busy} onPress={() => void perform(async () => {
        await api.setLocationSharing(link.id, !link.locationSharing);
        await guardian.refresh();
        return 'Location preference updated.';
      })} />}
    {!pending && <Pill label={link.videoSharing ? 'Camera shared on drives' : 'Camera not shared'}
      tone={link.videoSharing ? 'critical' : 'off'} />}
    {own && !pending && role === 'driver' && <Action compact secondary
      title={link.videoSharing ? 'Stop camera sharing' : 'Allow camera on shared drives'}
      disabled={busy} onPress={() => void perform(async () => {
        await api.setVideoSharing(link.id, !link.videoSharing);
        await guardian.refresh();
        return link.videoSharing ? 'Camera sharing turned off.' : 'Camera sharing allowed. You still turn the camera on for each drive.';
      })} />}
    <Action compact tone="ghost" title={pending ? 'Remove invitation' : 'Disconnect'} disabled={busy} onPress={() =>
      Alert.alert(pending ? 'Remove invitation?' : 'Disconnect?', 'This ends access to alerts shared through this link.', [
        { text: 'Cancel', style: 'cancel' },
        { text: 'Remove', style: 'destructive', onPress: () => void perform(async () => {
          await api.revokeLink(link.id);
          await guardian.refresh();
          return 'Connection removed.';
        }) },
      ])
    } />
  </Card>;
}

export function FamilyScreen({ guardian, role }: { guardian: GuardianState; role: Role }) {
  const [target, setTarget] = useState('');
  const { message, busy, perform } = useTask();
  const { session, links, error, refresh } = guardian;
  const api = guardianGateway;

  if (!api) {
    return <>
      <ScreenTitle eyebrow="Connections" title={COPY[role].title} subtitle={COPY[role].subtitle} />
      <Card tone="watch">
        <Label>Local mode</Label>
        <Body>Pairing needs the Supabase test project in .env.local. Until then, alerts stay in this phone&apos;s history{role === 'guardian' ? ' and Home shows a labeled sample' : ''}.</Body>
      </Card>
    </>;
  }
  const gateway = api;

  if (!session) {
    return <>
      <ScreenTitle eyebrow="Connections" title={COPY[role].title} subtitle={COPY[role].subtitle} />
      <SignIn role={role} />
    </>;
  }

  const active = links.filter((link) => link.status !== 'revoked');
  const mine = active.filter((link) => role === 'driver' ? link.driverId === session.user.id : link.driverId !== session.user.id);
  return <>
    <ScreenTitle eyebrow="Connections" title={COPY[role].title} subtitle={COPY[role].subtitle} />
    {role === 'driver' && <Card>
      <Label>Invite a parent or guardian</Label>
      <Field label="Their account email" value={target} onChangeText={setTarget} autoCapitalize="none" keyboardType="email-address" />
      <Action title="Send invitation" disabled={busy || !target.includes('@')} onPress={() => void perform(async () => {
        const id = await gateway.inviteGuardian(target);
        setTarget('');
        await refresh();
        return 'Invitation sent. Share code ' + id.slice(0, 8).toUpperCase() + ' with them in person.';
      })} />
    </Card>}
    <SectionHeader title={role === 'driver' ? 'Your guardians' : 'Your drivers'}
      action={<Action compact tone="ghost" title="Refresh" disabled={busy} onPress={() => void refresh()} />} />
    {!mine.length && <Card flat><Body>{role === 'driver' ? 'No guardians yet.' : 'No invitations yet. Ask your teen to invite this email.'}</Body></Card>}
    {mine.map((link) => <LinkCard key={link.id} link={link} guardian={guardian} role={role} busy={busy} perform={perform} />)}
    {error ? <Body urgent>{error}</Body> : null}
    {message ? <Body small>{message}</Body> : null}
    <SectionHeader title="Account" />
    <Card flat>
      <Body small>Signed in as {session.user.email ?? 'test account'}</Body>
      <Action compact secondary title="Sign out" disabled={busy} onPress={() => void perform(async () => {
        await gateway.signOut();
        return 'Signed out.';
      })} />
      <Action compact tone="danger" title="Delete account" disabled={busy} onPress={() =>
        Alert.alert('Delete this account?', 'This removes your account, links, and associated alerts.', [
          { text: 'Cancel', style: 'cancel' },
          { text: 'Delete account', style: 'destructive', onPress: () => void perform(async () => {
            await gateway.deleteAccount();
            return 'Account deleted.';
          }) },
        ])
      } />
    </Card>
  </>;
}
