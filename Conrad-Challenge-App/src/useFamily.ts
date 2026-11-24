import { useCallback, useEffect, useState } from 'react';
import { AppState } from 'react-native';
import { demoFamily, displayName, DRIVER_COLORS, initialsFor } from './family';
import type { DriverStatus, FamilyDriver, IndicatorReading, TripSummary } from './family';
import { guardianGateway } from './guardian';
import type { DriverStatusRow, DriverTripRow } from './guardian';
import type { GuardianState } from './GuardianScreen';
import type { LiveCondition, LiveFace, TileTone } from './liveView';
import type { Tier } from './risk';

const CONDITIONS: LiveCondition[] = ['none', 'distracted', 'drowsy', 'possible_impairment', 'possible_unresponsiveness'];
const TONES: TileTone[] = ['safe', 'watch', 'alert', 'critical', 'off'];
const POLL_MS = 3000;

function asCondition(value: unknown): LiveCondition {
  return CONDITIONS.includes(value as LiveCondition) ? value as LiveCondition : 'none';
}

function readTiles(indicators: Record<string, unknown> | null): IndicatorReading[] {
  const tiles = Array.isArray(indicators?.tiles) ? indicators.tiles : [];
  return tiles.slice(0, 8).flatMap((tile: unknown) => {
    if (!tile || typeof tile !== 'object') return [];
    const t = tile as Record<string, unknown>;
    const tone = TONES.includes(t.tone as TileTone) ? t.tone as TileTone : 'off';
    return [{ title: String(t.title ?? ''), value: String(t.value ?? ''), note: String(t.note ?? ''), tone }];
  });
}

function readFace(indicators: Record<string, unknown> | null): LiveFace | null {
  const face = indicators?.face as Record<string, unknown> | null | undefined;
  if (!face || typeof face !== 'object') return null;
  const yaw = Number(face.yaw);
  const pitch = Number(face.pitch);
  return { eyes: face.eyes === 'closed' ? 'closed' : 'open', yaw: Number.isFinite(yaw) ? yaw : 0, pitch: Number.isFinite(pitch) ? pitch : 0 };
}

export function statusFromRow(row: DriverStatusRow): DriverStatus | null {
  if (!row.sessionId || !row.updatedAt) return null;
  return {
    sessionId: row.sessionId,
    active: row.active,
    condition: asCondition(row.condition),
    tier: (Math.max(0, Math.min(3, row.tier ?? 0)) as Tier),
    score: Math.max(0, Math.min(1, row.score ?? 0)),
    indicators: readTiles(row.indicators),
    face: readFace(row.indicators),
    speedKmh: row.speedKmh,
    location: row.locationLat !== null && row.locationLng !== null
      ? { latitude: row.locationLat, longitude: row.locationLng, heading: row.heading } : null,
    updatedAt: row.updatedAt,
  };
}

function tripFromRow(row: DriverTripRow): TripSummary {
  return { ...row, worstCondition: asCondition(row.worstCondition) };
}

/** Live family from the backend when signed in; otherwise the labeled demo family, ticking once a second. */
export function useFamily(guardian: GuardianState) {
  const live = Boolean(guardianGateway && guardian.session);
  const [drivers, setDrivers] = useState<FamilyDriver[]>(() => live ? [] : demoFamily());
  const [trips, setTrips] = useState<Record<string, TripSummary[]>>({});
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    if (!live || !guardianGateway) {
      setDrivers(demoFamily());
      return;
    }
    try {
      const rows = await guardianGateway.listDriverStatus();
      setDrivers(rows.map((row, index) => {
        const name = displayName(row.driverEmail);
        return {
          id: row.driverId, name, initials: initialsFor(row.driverEmail), color: DRIVER_COLORS[index % DRIVER_COLORS.length],
          locationSharing: row.locationSharing, videoSharing: row.videoSharing, status: statusFromRow(row), trips: [],
        };
      }));
      setError('');
    } catch {
      setError('Could not reach your family’s live status. Retrying.');
    }
  }, [live]);

  useEffect(() => {
    void refresh();
    const timer = setInterval(() => {
      if (AppState.currentState === 'active') void refresh();
    }, live ? POLL_MS : 1000);
    return () => clearInterval(timer);
  }, [refresh, live]);

  const loadTrips = useCallback(async (driverId: string) => {
    if (!live || !guardianGateway) return;
    try {
      const rows = await guardianGateway.listDriverTrips(driverId);
      setTrips((current) => ({ ...current, [driverId]: rows.map(tripFromRow) }));
    } catch {
      setError('Could not load driving history.');
    }
  }, [live]);

  const withTrips = drivers.map((driver) => live ? { ...driver, trips: trips[driver.id] ?? [] } : driver);
  return { drivers: withTrips, demo: !live, error, refresh, loadTrips };
}
