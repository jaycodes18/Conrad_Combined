import { useCallback, useEffect, useRef, useState } from 'react';
import { AppState } from 'react-native';
import { setAudioModeAsync, useAudioPlayer } from 'expo-audio';
import { randomUUID } from 'expo-crypto';
import * as Location from 'expo-location';
import * as Speech from 'expo-speech';
import { DEMO_DURATION_MS, drillObservation } from './demo';
import { drivenMlReplays } from './drivenMl/replays';
import { guardianGateway } from './guardian';
import { mlReplayWarning } from './mlReplay';
import { cueLabel, RiskEngine } from './risk';
import type { Coverage, Observation, RiskAlert } from './risk';
import type { Delivery, SessionSource, Trip, TripEvent } from './trips';

type Options = SessionSource & { gps: boolean; shareLocation: boolean; linkId?: string; voice: boolean };
type SessionCoverage = Coverage | 'starting' | 'delayed';

type LiveTrip = {
  trip: Trip;
  started: number;
  engine: RiskEngine;
  options: Options;
  lastAt?: number;
  location?: Location.LocationObject;
  watch?: Location.LocationSubscription;
};

export function useTrip(save: (trip: Trip) => void, setDelivery: (id: string, delivery: Delivery) => void) {
  const live = useRef<LiveTrip | null>(null);
  const player = useAudioPlayer(require('../assets/warning.wav'));
  const voiceTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [run, setRun] = useState<(SessionSource & { id: string }) | null>(null);
  const [coverage, setCoverage] = useState<SessionCoverage>('starting');
  const [alert, setAlert] = useState<TripEvent | null>(null);
  const [fault, setFault] = useState('');
  const [gpsStatus, setGpsStatus] = useState('');

  useEffect(() => {
    void setAudioModeAsync({ playsInSilentMode: true, interruptionMode: 'duckOthers' }).catch(() => {});
  }, []);

  const clearVoice = useCallback(() => {
    if (voiceTimer.current) clearTimeout(voiceTimer.current);
    voiceTimer.current = null;
    void Speech.stop().catch(() => {});
  }, []);

  const end = useCallback((reason: Trip['end'] = 'finished') => {
    const state = live.current;
    if (!state) return;
    live.current = null;
    state.watch?.remove();
    clearVoice();
    player.pause();
    save({ ...state.trip, durationMs: Math.round(performance.now() - state.started), end: reason });
    setRun(null);
    if (reason === 'background') setFault('Monitoring stopped when the app left the foreground.');
  }, [clearVoice, player, save]);

  useEffect(() => {
    const listener = AppState.addEventListener('change', (status) => {
      if (status !== 'active') end('background');
    });
    return () => listener.remove();
  }, [end]);

  const playWarning = useCallback((state?: LiveTrip) => {
    clearVoice();
    void player.seekTo(0).then(() => {
      if (!state || live.current === state) player.play();
    }).catch(() => {});
    if (state?.options.voice) {
      voiceTimer.current = setTimeout(() => {
        if (live.current === state) {
          Speech.speak('Attention. If needed, pull over safely.', { rate: 0.93 });
        }
      }, 570);
    }
  }, [clearVoice, player]);

  const start = useCallback((options: Options) => {
    if (live.current || (AppState.currentState && AppState.currentState !== 'active')) return false;
    const id = randomUUID();
    const state: LiveTrip = {
      started: performance.now(),
      options,
      engine: new RiskEngine(options.mode === 'drill'),
      trip: {
        id,
        mode: options.mode,
        scenario: options.scenario,
        startedAt: new Date().toISOString(),
        end: 'finished',
        durationMs: 0,
        samples: 0,
        trackingSamples: 0,
        noFaceSamples: 0,
        maximumGapMs: 0,
        cameraTiming: options.mode === 'camera' ? {
          firstUsableFrameMs: null,
          longestFrameMs: null,
          droppedSlowFrames: 0,
        } : undefined,
        maxSpeedKmh: null,
        events: [],
      },
    };

    live.current = state;
    setFault('');
    setAlert(null);
    setCoverage('starting');
    setGpsStatus(options.gps ? 'Waiting for GPS' : '');
    setRun(options.mode === 'camera' ? { id, mode: options.mode } : { id, mode: options.mode, scenario: options.scenario });
    if (options.gps) {
      void Location.watchPositionAsync({ accuracy: Location.Accuracy.Balanced, timeInterval: 2000 }, (fix) => {
        if (live.current !== state) return;
        state.location = fix;
        setGpsStatus('');
        const speed = fix.coords.speed;
        if (typeof speed === 'number' && Number.isFinite(speed) && speed >= 0) {
          state.trip.maxSpeedKmh = Math.max(state.trip.maxSpeedKmh ?? 0, speed * 3.6);
        }
      }, () => {
        if (live.current === state) setGpsStatus('GPS unavailable');
      }).then((watch) => {
        if (live.current === state) state.watch = watch;
        else watch.remove();
      }).catch(() => {
        if (live.current === state) setGpsStatus('GPS unavailable');
      });
    }
    return true;
  }, []);

  const recordResult = useCallback((state: LiveTrip, at: number, coverage: Coverage, alerts: RiskAlert[]) => {
    if (state.lastAt !== undefined && at <= state.lastAt) return;
    const { trip, options } = state;
    const now = performance.now();
    const processingMs = Math.max(0, Math.round(now - at));
    if (trip.cameraTiming) {
      trip.cameraTiming.firstUsableFrameMs ??= Math.round(now - state.started);
      trip.cameraTiming.longestFrameMs = Math.max(trip.cameraTiming.longestFrameMs ?? 0, processingMs);
    }
    setCoverage(coverage);
    trip.samples += 1;
    if (coverage === 'tracking') trip.trackingSamples += 1;
    if (coverage === 'no-face') trip.noFaceSamples += 1;
    if (state.lastAt !== undefined) {
      trip.maximumGapMs = Math.max(trip.maximumGapMs, Math.round(at - state.lastAt));
    }
    state.lastAt = at;

    for (const signal of alerts) {
      const event: TripEvent = {
        ...signal,
        at: new Date().toISOString(),
        elapsedMs: Math.round(at - state.started),
        processingMs,
      };
      if (trip.events.length >= 120) {
        const oldestWarning = trip.events.findIndex((item) => item.level === 1);
        if (oldestWarning >= 0) trip.events.splice(oldestWarning, 1);
      }
      trip.events.push(event);
      setAlert(event);
      if (signal.level === 1) playWarning(state);

      if (signal.level === 2) {
        if (!guardianGateway || !options.linkId) {
          trip.delivery = { state: 'local' };
          continue;
        }

        const startedSending = performance.now();
        const fix = state.location;
        const location = options.shareLocation && fix && Date.now() - fix.timestamp <= 30_000 &&
          fix.coords.accuracy !== null && fix.coords.accuracy <= 100
          ? { latitude: fix.coords.latitude, longitude: fix.coords.longitude }
          : undefined;
        trip.delivery = { state: 'pending' };
        void guardianGateway.sendTestAlert(options.linkId, trip.id, {
          cue: signal.cue,
          level: 2,
          evidenceMs: Math.round(signal.evidenceMs),
          location,
        }).then(() => {
          const delivery: Delivery = { state: 'delivered', elapsedMs: Math.round(performance.now() - startedSending) };
          if (live.current === state) trip.delivery = delivery;
          else setDelivery(trip.id, delivery);
        }).catch(() => {
          const delivery: Delivery = { state: 'failed', elapsedMs: Math.round(performance.now() - startedSending) };
          if (live.current === state) trip.delivery = delivery;
          else setDelivery(trip.id, delivery);
        });
      }
    }
  }, [playWarning, setDelivery]);

  const observe = useCallback((sample: Observation) => {
    const state = live.current;
    if (!state || sample.at < state.started || (state.lastAt !== undefined && sample.at <= state.lastAt)) return;
    const result = state.engine.update(sample);
    recordResult(state, sample.at, result.coverage, result.alerts);
  }, [recordResult]);

  const markTrackingDelayed = useCallback((slowFrameMs?: number) => {
    const state = live.current;
    if (state?.options.mode !== 'camera') return;
    const timing = state.trip.cameraTiming;
    if (timing && slowFrameMs !== undefined) {
      timing.droppedSlowFrames += 1;
      timing.longestFrameMs = Math.max(timing.longestFrameMs ?? 0, Math.round(slowFrameMs));
    }
    state.engine.interruptTracking();
    setCoverage('delayed');
  }, []);

  useEffect(() => {
    if (run?.mode !== 'drill') return;
    const scenario = run.scenario;
    const timer = setInterval(() => {
      const state = live.current;
      if (!state || state.trip.id !== run.id) return;
      const now = performance.now();
      const elapsed = now - state.started;
      if (elapsed >= DEMO_DURATION_MS) {
        end();
      } else {
        observe({ ...drillObservation(elapsed, scenario), at: now });
      }
    }, 200);
    return () => clearInterval(timer);
  }, [run, end, observe]);

  useEffect(() => {
    if (run?.mode !== 'ml-replay') return;
    const signals = drivenMlReplays[run.scenario];
    const first = signals[0].t_seconds;
    const duration = (signals[signals.length - 1].t_seconds - first) * 1000 + 3500;
    let index = 0;
    const tick = () => {
      const state = live.current;
      if (!state || state.trip.id !== run.id) return;
      const elapsed = performance.now() - state.started;
      while (index < signals.length) {
        const signal = signals[index];
        const offset = (signal.t_seconds - first) * 1000;
        if (offset > elapsed) break;
        const warning = mlReplayWarning(signal);
        recordResult(state, state.started + offset, 'tracking', warning ? [warning] : []);
        index += 1;
      }
      if (elapsed >= duration) end();
    };
    tick();
    const timer = setInterval(tick, 100);
    return () => clearInterval(timer);
  }, [run, end, recordResult]);

  const stopOnFault = useCallback((message: string) => {
    if (!live.current) return;
    setFault(message);
    end('fault');
  }, [end]);

  return { run, alert, coverage, fault, gpsStatus, start, end, observe, markTrackingDelayed, stopOnFault, playWarning };
}

export function sessionStatus(coverage: SessionCoverage, event: TripEvent | null) {
  switch (coverage) {
    case 'starting': return 'Starting camera and model…';
    case 'delayed': return 'Camera frames delayed';
    case 'calibrating': return 'Look ahead to calibrate';
    case 'no-face': return 'Face not visible';
    case 'tracking':
      if (event?.level === 3) return 'Emergency escalation simulated';
      if (event?.level === 2) return 'Guardian test alert recorded';
      if (event) return cueLabel[event.cue];
      return 'Monitoring active';
  }
}
