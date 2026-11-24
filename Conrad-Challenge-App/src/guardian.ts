import 'react-native-url-polyfill/auto';

import AsyncStorage from '@react-native-async-storage/async-storage';
import { createClient, type Session, type SupabaseClient } from '@supabase/supabase-js';
import * as SecureStore from 'expo-secure-store';
import { Platform } from 'react-native';

const CHUNK_SIZE = 1500;
const manifestKey = (key: string) => `${key}_chunks`;
const chunkKey = (key: string, index: number) => `${key}_chunk_${index}`;

// SecureStore has a small per-value limit on some platforms. Keep the full
// Supabase auth session in encrypted SecureStore shards with a tiny manifest.
const secureSessionStorage = {
  async getItem(key: string): Promise<string | null> {
    const countText = await SecureStore.getItemAsync(manifestKey(key));
    if (!countText) return null;
    const count = Number(countText);
    if (!Number.isSafeInteger(count) || count < 1 || count > 100) {
      await this.removeItem(key);
      return null;
    }
    const chunks = await Promise.all(Array.from({ length: count }, (_, i) => SecureStore.getItemAsync(chunkKey(key, i))));
    if (chunks.some((chunk) => chunk === null)) {
      await this.removeItem(key);
      return null;
    }
    return chunks.join('');
  },
  async setItem(key: string, value: string): Promise<void> {
    const count = Math.max(1, Math.ceil(value.length / CHUNK_SIZE));
    if (count > 100) throw new Error('Auth session exceeds secure storage capacity.');
    const previous = Number(await SecureStore.getItemAsync(manifestKey(key)) ?? 0);
    await Promise.all(Array.from({ length: count }, (_, i) =>
      SecureStore.setItemAsync(chunkKey(key, i), value.slice(i * CHUNK_SIZE, (i + 1) * CHUNK_SIZE))));
    await SecureStore.setItemAsync(manifestKey(key), String(count));
    await Promise.all(Array.from({ length: Math.max(0, previous - count) }, (_, i) =>
      SecureStore.deleteItemAsync(chunkKey(key, count + i))));
  },
  async removeItem(key: string): Promise<void> {
    const count = Number(await SecureStore.getItemAsync(manifestKey(key)) ?? 0);
    await Promise.all(Array.from({ length: Math.min(count, 100) }, (_, i) => SecureStore.deleteItemAsync(chunkKey(key, i))));
    await SecureStore.deleteItemAsync(manifestKey(key));
  },
};

export type GuardianLink = {
  id: string;
  otherEmail: string;
  driverId: string;
  guardianId: string | null;
  status: 'pending' | 'accepted' | 'revoked';
  locationSharing: boolean;
  videoSharing: boolean;
  createdAt: string;
  acceptedAt: string | null;
};

export type GuardianTestAlert = {
  id: string;
  linkId: string;
  sessionId: string;
  cue: 'eye-closure' | 'head-away';
  level: 2;
  evidenceMs: number;
  locationLat: number | null;
  locationLng: number | null;
  createdAt: string;
  isTest: true;
};

export type TestAlertInput = {
  cue: GuardianTestAlert['cue'];
  level: 2;
  evidenceMs: number;
  location?: { latitude: number; longitude: number };
};

export type LiveStatusInput = {
  sessionId: string;
  active: boolean;
  condition: string;
  tier: number;
  score: number;
  indicators: Record<string, unknown>;
  speedKmh: number | null;
  location?: { latitude: number; longitude: number; heading: number | null };
};

export type TripSummaryInput = {
  sessionId: string;
  startedAt: string;
  durationS: number;
  driverAlerts: number;
  guardianAlerts: number;
  worstCondition: string;
  trackedPct: number | null;
};

export type DriverStatusRow = {
  driverId: string;
  driverEmail: string;
  locationSharing: boolean;
  sessionId: string | null;
  active: boolean;
  condition: string | null;
  tier: number | null;
  score: number | null;
  indicators: Record<string, unknown> | null;
  speedKmh: number | null;
  locationLat: number | null;
  locationLng: number | null;
  heading: number | null;
  updatedAt: string | null;
  videoSharing: boolean;
};

export type DriverTripRow = {
  sessionId: string;
  startedAt: string;
  durationS: number;
  driverAlerts: number;
  guardianAlerts: number;
  worstCondition: string;
  trackedPct: number | null;
};

export type VideoFrame = { at: number; jpg: string };
export type VideoSender = { send(jpgBase64: string): void; close(): void };

export type GuardianGateway = {
  client: SupabaseClient;
  signUp(email: string, password: string): Promise<{ needsEmailVerification: boolean }>;
  signIn(email: string, password: string): Promise<Session>;
  signOut(): Promise<void>;
  getSession(): Promise<Session | null>;
  onAuthStateChange(callback: (session: Session | null) => void): () => void;
  inviteGuardian(email: string): Promise<string>;
  listLinks(): Promise<GuardianLink[]>;
  acceptInvite(linkId: string): Promise<void>;
  revokeLink(linkId: string): Promise<void>;
  setLocationSharing(linkId: string, enabled: boolean): Promise<void>;
  setVideoSharing(linkId: string, enabled: boolean): Promise<void>;
  /** Opens the driver's private video topic for sending. Frames are relayed, never stored. */
  openVideoSender(): Promise<VideoSender>;
  /** Joins a driver's private video topic. RLS refuses the join unless the link allows video. */
  subscribeVideo(driverId: string, onFrame: (frame: VideoFrame) => void, onStatus?: (status: string) => void): () => void;
  sendTestAlert(linkId: string, sessionId: string, alert: TestAlertInput): Promise<void>;
  listTestAlerts(): Promise<GuardianTestAlert[]>;
  subscribeToInbox(onChange: () => void): () => void;
  publishStatus(status: LiveStatusInput): Promise<void>;
  recordTrip(trip: TripSummaryInput): Promise<void>;
  listDriverStatus(): Promise<DriverStatusRow[]>;
  listDriverTrips(driverId: string): Promise<DriverTripRow[]>;
  deleteAccount(): Promise<void>;
};

function throwIfError(error: { message: string } | null): void {
  if (error) throw new Error(error.message);
}

function createGateway(client: SupabaseClient): GuardianGateway {
  const rpc = async <T>(name: string, args?: Record<string, unknown>, signal?: AbortSignal) => {
    const request = client.rpc(name, args);
    const { data, error } = await (signal ? request.abortSignal(signal) : request);
    throwIfError(error);
    return data as T;
  };
  return {
    client,
    async signUp(email, password) {
      const { data, error } = await client.auth.signUp({ email: email.trim().toLowerCase(), password });
      throwIfError(error);
      return { needsEmailVerification: !data.session };
    },
    async signIn(email, password) {
      const { data, error } = await client.auth.signInWithPassword({ email: email.trim().toLowerCase(), password });
      throwIfError(error);
      if (!data.session) throw new Error('Sign-in did not return a session. Verify your email and try again.');
      return data.session;
    },
    async signOut() {
      const { error } = await client.auth.signOut();
      throwIfError(error);
    },
    async getSession() {
      const { data, error } = await client.auth.getSession();
      throwIfError(error);
      return data.session;
    },
    onAuthStateChange(callback) {
      const { data } = client.auth.onAuthStateChange((_event, session) => callback(session));
      return () => data.subscription.unsubscribe();
    },
    async inviteGuardian(email) {
      return await rpc<string>('guardian_invite', { p_email: email });
    },
    async listLinks() {
      const rows = await rpc<Array<Record<string, unknown>>>('guardian_list_links');
      return rows.map((row) => ({
        id: String(row.id), driverId: String(row.driver_id),
        otherEmail: String(row.other_email),
        guardianId: row.guardian_id ? String(row.guardian_id) : null,
        status: row.status as GuardianLink['status'], locationSharing: Boolean(row.location_sharing),
        videoSharing: Boolean(row.video_sharing),
        createdAt: String(row.created_at), acceptedAt: row.accepted_at ? String(row.accepted_at) : null,
      }));
    },
    async acceptInvite(linkId) {
      await rpc('guardian_accept', { p_link_id: linkId });
    },
    async revokeLink(linkId) {
      await rpc('guardian_revoke', { p_link_id: linkId });
    },
    async setLocationSharing(linkId, enabled) {
      await rpc('guardian_set_location_sharing', { p_link_id: linkId, p_enabled: enabled });
    },
    async setVideoSharing(linkId, enabled) {
      await rpc('guardian_set_video_sharing', { p_link_id: linkId, p_enabled: enabled });
    },
    async openVideoSender() {
      const { data } = await client.auth.getUser();
      if (!data.user) throw new Error('Sign in to share the camera.');
      await client.realtime.setAuth();
      const channel = client.channel(`video:${data.user.id}`, { config: { private: true, broadcast: { self: false, ack: false } } });
      let joined = false;
      channel.subscribe((status) => { joined = status === 'SUBSCRIBED'; });
      return {
        send(jpg) {
          if (!joined || jpg.length > 240_000) return;
          void channel.send({ type: 'broadcast', event: 'frame', payload: { at: Date.now(), jpg } });
        },
        close() { void client.removeChannel(channel); },
      };
    },
    subscribeVideo(driverId, onFrame, onStatus) {
      let channel: ReturnType<SupabaseClient['channel']> | null = null;
      let closed = false;
      void client.realtime.setAuth().then(() => {
        if (closed) return;
        channel = client.channel(`video:${driverId}`, { config: { private: true } })
          .on('broadcast', { event: 'frame' }, ({ payload }) => {
            const frame = payload as Partial<VideoFrame>;
            if (typeof frame?.jpg === 'string' && typeof frame.at === 'number') onFrame({ at: frame.at, jpg: frame.jpg });
          })
          .subscribe((status) => onStatus?.(status));
      });
      return () => {
        closed = true;
        if (channel) void client.removeChannel(channel);
      };
    },
    async sendTestAlert(linkId, sessionId, alert) {
      if (alert.level !== 2) throw new Error('Only L2 test alerts can be sent to a guardian.');
      if (!Number.isInteger(alert.evidenceMs) || alert.evidenceMs < 0 || alert.evidenceMs > 600_000) {
        throw new Error('Alert evidence duration is invalid.');
      }
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 12_000);
      try {
        await rpc('guardian_send_test_alert', {
          p_link_id: linkId, p_session_id: sessionId, p_cue: alert.cue,
          p_level: alert.level, p_evidence_ms: alert.evidenceMs,
          p_location_lat: alert.location?.latitude ?? null,
          p_location_lng: alert.location?.longitude ?? null,
        }, controller.signal);
      } finally {
        clearTimeout(timeout);
      }
    },
    async listTestAlerts() {
      const { data, error } = await client.from('guardian_test_alerts').select(
        'id, link_id, session_id, cue, level, evidence_ms, location_lat, location_lng, created_at, is_test',
      ).eq('is_test', true).order('created_at', { ascending: false }).limit(100);
      throwIfError(error);
      return (data ?? []).map((row) => ({
        id: row.id, linkId: row.link_id, sessionId: row.session_id,
        cue: row.cue as GuardianTestAlert['cue'], level: row.level as GuardianTestAlert['level'],
        evidenceMs: row.evidence_ms, locationLat: row.location_lat, locationLng: row.location_lng,
        createdAt: row.created_at, isTest: true,
      }));
    },
    subscribeToInbox(onChange) {
      let active = true;
      const channel = client.channel(`guardian-inbox-${Math.random().toString(36).slice(2)}`)
        .on('postgres_changes', { event: '*', schema: 'public', table: 'guardian_test_alerts' }, onChange)
        .subscribe();
      const poll = setInterval(() => { if (active) onChange(); }, 15_000);
      return () => {
        active = false;
        clearInterval(poll);
        void client.removeChannel(channel);
      };
    },
    async publishStatus(status) {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 8_000);
      try {
        await rpc('driver_publish_status', {
          p_session_id: status.sessionId, p_active: status.active, p_condition: status.condition,
          p_tier: status.tier, p_score: Math.min(1, Math.max(0, status.score)), p_indicators: status.indicators,
          p_speed_kmh: status.speedKmh, p_location_lat: status.location?.latitude ?? null,
          p_location_lng: status.location?.longitude ?? null, p_heading: status.location?.heading ?? null,
        }, controller.signal);
      } finally {
        clearTimeout(timeout);
      }
    },
    async recordTrip(trip) {
      await rpc('driver_record_trip', {
        p_session_id: trip.sessionId, p_started_at: trip.startedAt, p_duration_s: trip.durationS,
        p_driver_alerts: trip.driverAlerts, p_guardian_alerts: trip.guardianAlerts,
        p_worst_condition: trip.worstCondition, p_tracked_pct: trip.trackedPct,
      });
    },
    async listDriverStatus() {
      const rows = await rpc<Array<Record<string, unknown>>>('guardian_list_driver_status');
      const num = (value: unknown) => typeof value === 'number' && Number.isFinite(value) ? value : null;
      return (rows ?? []).map((row) => ({
        driverId: String(row.driver_id), driverEmail: String(row.driver_email ?? ''),
        locationSharing: Boolean(row.location_sharing),
        sessionId: row.session_id ? String(row.session_id) : null, active: Boolean(row.active),
        condition: row.condition ? String(row.condition) : null, tier: num(row.tier), score: num(row.score),
        indicators: row.indicators && typeof row.indicators === 'object' ? row.indicators as Record<string, unknown> : null,
        speedKmh: num(row.speed_kmh), locationLat: num(row.location_lat), locationLng: num(row.location_lng),
        heading: num(row.heading), updatedAt: row.updated_at ? String(row.updated_at) : null,
        videoSharing: Boolean(row.video_sharing),
      }));
    },
    async listDriverTrips(driverId) {
      const rows = await rpc<Array<Record<string, unknown>>>('guardian_list_driver_trips', { p_driver_id: driverId });
      return (rows ?? []).map((row) => ({
        sessionId: String(row.session_id), startedAt: String(row.started_at), durationS: Number(row.duration_s),
        driverAlerts: Number(row.driver_alerts), guardianAlerts: Number(row.guardian_alerts),
        worstCondition: String(row.worst_condition),
        trackedPct: typeof row.tracked_pct === 'number' ? row.tracked_pct : null,
      }));
    },
    async deleteAccount() {
      const { error } = await client.functions.invoke('delete-account', { method: 'POST' });
      throwIfError(error);
      await client.auth.signOut();
    },
  };
}

const supabaseUrl = process.env.EXPO_PUBLIC_SUPABASE_URL?.trim();
const publishableKey = process.env.EXPO_PUBLIC_SUPABASE_PUBLISHABLE_KEY?.trim();

// expo-secure-store has no web implementation; the browser parent dashboard keeps the session in local storage.
const authStorage = Platform.OS === 'web' ? AsyncStorage : secureSessionStorage;

export const guardianGateway: GuardianGateway | null = supabaseUrl && publishableKey
  ? createGateway(createClient(supabaseUrl, publishableKey, {
      auth: {
        storage: authStorage,
        autoRefreshToken: true,
        persistSession: true,
        detectSessionInUrl: false,
      },
    }))
  : null;
