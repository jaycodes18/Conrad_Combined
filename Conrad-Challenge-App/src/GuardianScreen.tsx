import { useCallback, useEffect, useRef, useState } from 'react';
import { Alert, AppState } from 'react-native';
import type { Session } from '@supabase/supabase-js';
import { guardianGateway } from './guardian';
import type { GuardianLink, GuardianTestAlert } from './guardian';
import { cueLabel } from './risk';
import { Action, Body, Card, Field, Heading, Label } from './ui';

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

export function GuardianScreen({ guardian }: { guardian: ReturnType<typeof useGuardian> }) {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [target, setTarget] = useState('');
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);
  const lock = useRef(false);
  const { session, links, inbox, error, refresh, linkLabel } = guardian;

  if (!guardianGateway) {
    return <>
      <Heading>Guardian</Heading>
      <Card>
        <Label>Local mode</Label>
        <Body>Pairing and the guardian test inbox are available when the app is connected to a Supabase test project. Test escalations appear in local trip history in the meantime.</Body>
      </Card>
    </>;
  }
  const api = guardianGateway;

  async function perform(task: () => Promise<string>) {
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
  }

  if (!session) {
    const valid = email.includes('@') && password.length >= 8;
    return <>
      <Heading>Guardian access</Heading>
      <Body>Sign in on each phone with separate accounts to test an alert between a driver and guardian.</Body>
      <Card>
        <Field label="Email" value={email} onChangeText={setEmail} autoCapitalize="none" autoComplete="email" keyboardType="email-address" />
        <Field label="Password" value={password} onChangeText={setPassword} secureTextEntry autoCapitalize="none" />
        <Action title="Sign in" disabled={busy || !valid} onPress={() => void perform(async () => {
          await api.signIn(email, password);
          setPassword('');
          return 'Signed in.';
        })} />
        <Action title="Create test account" secondary disabled={busy || !valid} onPress={() => void perform(async () => {
          const { needsEmailVerification } = await api.signUp(email, password);
          setPassword('');
          return needsEmailVerification ? 'Check your email for a confirmation link, then return here to sign in.' : 'Test account created.';
        })} />
        <Body>Use a password with at least eight characters.</Body>
      </Card>
      {message ? <Body>{message}</Body> : null}
    </>;
  }

  return <>
    <Heading>Guardian access</Heading>
    <Body>Signed in as {session.user.email ?? 'test account'}. Paired guardians can see test alerts after accepting an invitation.</Body>
    <Card>
      <Label>Invite a consenting guardian</Label>
      <Field label="Guardian account email" value={target} onChangeText={setTarget} autoCapitalize="none" keyboardType="email-address" />
      <Action title="Create invitation" disabled={busy || !target.includes('@')} onPress={() => void perform(async () => {
        const id = await api.inviteGuardian(target);
        setTarget('');
        await refresh();
        return 'Ask the guardian to sign in and look for invitation ' + id.slice(0, 8).toUpperCase() + '.';
      })} />
    </Card>
    <Card>
      <Label>Connections</Label>
      <Action title="Refresh links and inbox" secondary disabled={busy} onPress={() => void refresh()} />
      {!links.some((link) => link.status !== 'revoked') && <Body>No invitations or active links.</Body>}
      {links.filter((link) => link.status !== 'revoked').map((link) => {
        const driver = link.driverId === session.user.id;
        const pending = link.status === 'pending';
        return <Card key={link.id}>
          <Label>{linkLabel(link)}</Label>
          <Body>{pending ? 'Awaiting guardian acceptance' : 'Accepted'} · {new Date(link.createdAt).toLocaleDateString()}</Body>
          {pending && !driver && <>
            <Body>Confirm invitation {link.id.slice(0, 8).toUpperCase()} with the driver before accepting.</Body>
            <Action title="Accept this invitation" disabled={busy} onPress={() => void perform(async () => {
              await api.acceptInvite(link.id);
              await refresh();
              return 'Guardian invitation accepted.';
            })} />
          </>}
          {driver && !pending && <Action title={link.locationSharing ? 'Disable location for this link' : 'Allow opt-in location for this link'}
            secondary disabled={busy} onPress={() => void perform(async () => {
              await api.setLocationSharing(link.id, !link.locationSharing);
              await refresh();
              return 'Location preference updated.';
            })} />}
          <Action title={pending ? 'Remove invitation' : 'Revoke link'} secondary disabled={busy} onPress={() =>
            Alert.alert(pending ? 'Remove invitation?' : 'Revoke this link?', 'This ends access to alerts shared through this link.', [
              { text: 'Cancel', style: 'cancel' },
              { text: 'Remove', style: 'destructive', onPress: () => void perform(async () => {
                await api.revokeLink(link.id);
                await refresh();
                return 'Connection removed.';
              }) },
            ])
          } />
        </Card>;
      })}
    </Card>
    <Card>
      <Label>Guardian test inbox</Label>
      <Body>Open this screen on the guardian phone during the drill. Test alerts update while the app is open.</Body>
      {!inbox.length && <Body>No guardian test alerts received.</Body>}
      {inbox.map((entry) => <Card key={entry.id}>
        <Label>TEST · {cueLabel[entry.cue]}</Label>
        <Body>{new Date(entry.createdAt).toLocaleString()} · {(entry.evidenceMs / 1000).toFixed(1)} s of evidence</Body>
        {entry.locationLat !== null && entry.locationLng !== null &&
          <Body>Shared location {entry.locationLat.toFixed(4)}, {entry.locationLng.toFixed(4)}</Body>}
      </Card>)}
    </Card>
    {error ? <Body urgent>{error}</Body> : null}
    {message ? <Body>{message}</Body> : null}
    <Card>
      <Label>Account</Label>
      <Action title="Sign out" secondary disabled={busy} onPress={() => void perform(async () => {
        await api.signOut();
        return 'Signed out.';
      })} />
      <Action title="Delete test account" secondary disabled={busy} onPress={() =>
        Alert.alert('Delete this test account?', 'This removes your account, links, and associated guardian test alerts.', [
          { text: 'Cancel', style: 'cancel' },
          { text: 'Delete account', style: 'destructive', onPress: () => void perform(async () => {
            await api.deleteAccount();
            return 'Test account deleted.';
          }) },
        ])
      } />
    </Card>
  </>;
}
