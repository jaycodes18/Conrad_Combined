# Audit: the ML repository and the team app, before the indicator and risk-signal work

Written from the code on 2026-11-23, at commit `9914af9` (ML) and `d8d8a74` (app,
`saharshambati1/Conrad-Challenge-App`, cloned read-only to `../Conrad-Challenge-App`).
"Imported by" counts only imports from other package modules, not tests. "Tests" names the
`ml/tests/` files that exercise the module directly; the pipeline modules are exercised end to end by
`tools/check.py --dev-run` (the full synthetic pipeline through staging), which is not a unit test.

## A. ML repository as it stands

### Pipeline modules (`driven_ml/`)

| Module | What it computes | Product feature | Tests | Imported by |
| --- | --- | --- | --- | --- |
| `spec.py` | Landmark indices, signal thresholds, window and baseline lengths, feature order | Contract shared by training and the phone | via `test_studio*.py` | 20 modules |
| `features/geometry.py` | EAR, MAR, head pose, horizontal iris gaze per frame | Per-frame face signals | `test_studio_wave_b.py` (`no_face_frame`) | app_fixtures, conditions, live, extraction |
| `features/extraction.py` | MediaPipe Face Landmarker over video at the target fps | Real-data features | none (dev run does not use video) | frames, conditions, live |
| `features/frames.py` | The frame table: synthetic people or licensed video | Training data | dev run | windows, replay |
| `features/preprocess.py` | CLAHE low light, seat crop, fisheye correction | Robustness to cabin optics | `test_conditions.py` | server, conditions, live, extraction |
| `features/synthetic.py` | Synthetic people, camera placements, scripted team sessions | Dev data with the camera confound | dev run | app_fixtures, sources, replay, frames, null_control |
| `windows.py` | 60 s windows, personal baseline deltas, labels | Model inputs | dev run | 7 modules |
| `splits.py` | Person split: development, locked test by id hash, replay people | Locked test (invariant 3) | dev run | pipeline, replay, null_control |
| `modeling/training.py` | Person-grouped CV of every candidate, temperature, threshold, export | Drowsiness head | dev run | pipeline, modeling `__init__`, null_control |
| `modeling/candidates.py`, `calibration.py`, `alert_simulation.py`, `artifact.py`, `store.py`, `null_control.py` | Candidates, temperature scaling, false-alert threshold, JSON model, candidate store, null-signal control | Drowsiness head and its gates | `artifact` in `test_studio.py`; rest via dev run | training, gate, evaluation |
| `evaluation.py`, `scoring.py` | Locked test scoring; metrics by baseline scope and within one recording | Accuracy gate | dev run | pipeline, gate, replay |
| `staging/gate.py`, `packaging.py`, `replay.py` | Every staging check; packaging; scripted replay | Staging gates (invariant 6) | dev run | release, app_fixtures |
| `release.py` | Promotion with a named approver, commercial track only | Human-approved promotion | dev run (not promoted) | launch, pipeline |
| `compliance/rules.py`, `checks.py` | Legal flags, banned claims, forbidden code patterns | Compliance gate (invariant 1) | dev run stage 1 | gate, launch, product |
| `data/registry.py`, `discovery.py`, `consent.py` | Dataset licences; Hugging Face / Kaggle search; team consent | Licence filtering, consent (invariant 5) | `consent` via product surface tests; dev run with `--no-discovery` | pipeline, catalog |
| `app_fixtures.py` | Frame, window, replay and model parity fixtures | On-device port parity | dev run (parity gate) | launch, pipeline |
| `economics.py`, `product.py`, `settings.py`, `workspace.py`, `errors.py` | Unit economics; typed product.json; run settings; atomic I/O; typed errors | Business gate; single source of prices and flags | dev run | many |
| `launch.py`, `leads.py`, `cli.py` | Launch readiness check; lead summary; CLI | Go-to-market review | none | console scripts only (`driven-launch-check`, `driven-ml`); `leads.py` has no importer |

### Studio modules (`driven_ml/studio/`)

| Module | What it computes | Product feature | Tests | Imported by |
| --- | --- | --- | --- | --- |
| `live.py` | Camera loop, calibration, 60 s window scoring, eyes-off 2 s rule, observation publishing, fused risk per frame, 10-minute parent-text simulation | Live demo of the shipped model and rules | `test_studio.py`, `test_studio_wave_b.py` | server, training |
| `monitor.py` | `TemporalTracker`: continuous gaze-away and closure timers, camera-invalid state, re-arm | Research timers (3 s gaze, 1.5 s closure) | `test_product_surface.py` | live, server |
| `risk.py` | `RiskFusionEngine`: confidence-weighted scores for impairment / unresponsiveness / driving, hysteresis, staleness | Experimental risk fusion | `test_studio_wave_a.py`, `test_studio_wave_b.py` | live |
| `observations.py` | Timestamped `SensorObservation`, `ObservationBus` with stale expiry | Studio metrics panel; fusion input | wave A/B | 8 modules |
| `audio_health.py` | Mic fault classification: mute, gain, clipping, speech, road noise, frozen, rate change | Gated research audio | `test_studio_wave_a.py` | breathing, microphone, scenarios |
| `breathing.py` | Envelope peak detection of breath candidates, rate and regularity | Research breathing (audio) | via `scenarios` in wave A | microphone, scenarios |
| `visual_respiration.py` | Zero-crossing rate of head pitch + jaw motion | Research breathing (visual) | **none** | **none** (dead code) |
| `microphone.py` | Gated `sounddevice` capture, retry, RAM-only chunks | Gated research audio | wave A, wave B | server |
| `research_audio.py` | The two-flag legal gate for the microphone | Gated research audio | wave A | microphone, server |
| `contacts.py` | `ContactBook` with consent, `SimulatedEscalation` (local warning, simulated SMS and call, 911 refusal) | Family escalation simulation | `test_product_surface.py` | live, server |
| `conditions.py` | OpenCV condition transforms run through MediaPipe; eye visibility, separation, verdicts | Glasses / light / lens robustness | `test_conditions.py` | live, server, scenarios |
| `scenarios.py` | Synthetic PCM scenarios and captured-frame condition scenarios, `NOT RUN` rows | Scenario lab | wave A | server |
| `server.py` | HTTP server, SSE `/events`, MJPEG, 9 GET and 22 POST routes | Studio dashboard | wave B (`Studio`) | studio `__init__` |
| `sources.py`, `cameras.py` | Frame sources (webcam, RTSP/HTTP, file, synthetic); camera kinds, gap rules, hardware registry | Camera input | wave B; product surface | live, server, status |
| `catalog.py` | Dataset cards and a named approval store; ingest plans never download | Dataset review | product surface, wave A | server, status |
| `profiles.py` | Driver profiles and saved calibration | Saved baseline | product surface | server |
| `status.py` | Compatibility matrix and weakness audit | Honest status tab | product surface | server |
| `telemetry.py` | Optional driving-stability inputs | none in use | **none** | **none** (dead code; `live.py` publishes `driving_stability` as always unavailable) |
| `training.py` | Runs the real pipeline stage by stage for the Train tab | Train tab | none | server |
| `events.py` | SSE publish/subscribe hub | Studio transport | `test_studio.py` | server, live, training |

### Duplication, confirmed or corrected

1. **`monitor.py` and `live.py` both track closure: corrected.** `live.py` does not time closure itself. It
   calls `TemporalTracker.push` (`studio/live.py:799-820`) and republishes `closure_for_seconds` as the
   `eye_closure` observation (`studio/live.py:312-324`). The real duplicate is the gaze-away timer:
   `TemporalTracker` times off-forward gaze against `off_forward_gaze_seconds` (3 s, research), and
   `_Session._eyes_off` (`studio/live.py:829-845`) times the 2 s NHTSA rule separately with its own
   `eyes_off_since`. Two timers measure the same thing with different thresholds.
2. **`breathing.py` and `visual_respiration.py` both publish `breathing_rate`: confirmed, with a
   correction.** Both build a `SensorObservation` named `breathing_rate`
   (`studio/breathing.py` `rate_observation`, `studio/visual_respiration.py` `observation(name="breathing_rate")`),
   so they would overwrite each other on the bus. Nothing imports `visual_respiration.py`, so in practice only
   the audio path publishes.
3. **`contacts.py` and the 10-minute parent-text simulation in `live.py` both model escalation: confirmed.**
   `_Session._escalate` (`studio/live.py:915-926`) emits an `escalation` alert on a second drowsiness alert
   within 600 s. `SimulatedEscalation.note_warning` (`studio/contacts.py`) sends a simulated SMS on a second
   qualifying warning within 60 s. Two different rules for the same product promise.
4. **Also found:** staleness is applied twice, in `ObservationBus.expire` and again in
   `RiskFusionEngine.assess`, both with the 2 s / 15 s special case for `drowsiness_probability`. The eye
   visibility cutoff lives only in `conditions.py` (`FAIL_EYE_VISIBILITY = 0.25`), so nothing on the live path
   uses it.

## B. The app (`../Conrad-Challenge-App`)

**Stack.** Expo SDK 58, React Native 0.88, TypeScript (`package.json`: `"expo": "~58.0.23"`,
`"react-native": "0.88.1"`, `"typescript": "~6.0.3"`). Face landmarks run in a WebView with
`@mediapipe/tasks-vision` 0.10.32 (`src/visionHtml.ts:80-87`). Optional backend is Supabase (Auth, Postgres
with row-level security, Realtime, one Edge Function): `src/guardian.ts:4`
`import { createClient, ... } from '@supabase/supabase-js'`, `supabase/migrations/20261107010000_guardian_test_backend.sql`,
`supabase/functions/delete-account/index.ts`.

**Where the app receives anything from the ML side.** Nothing yet. The app runs its own heuristic engine,
`src/risk.ts`, on `{ eyesClosed, yaw, pitch }` from blendshapes (`src/visionHtml.ts:87`). Its alert type is
`src/risk.ts:8` `export type RiskAlert = { cue: Cue; level: Level; evidenceMs: number };` with
`Cue = 'eye-closure' | 'head-away'` and `Level = 1 | 2 | 3`. The only server write is the RPC
`guardian_send_test_alert(p_link_id, p_session_id, p_cue, p_level, p_evidence_ms, p_location_lat, p_location_lng)`
(`supabase/migrations/20261107010000_guardian_test_backend.sql:197-200`), called from
`src/useTrip.ts:170` and `src/guardian.ts:162`, inserting into table `guardian_test_alerts` with
`cue text ... check (cue in ('eye-closure', 'head-away'))`, `level smallint ... check (level = 2)`,
`evidence_ms integer` (`:35-37`). The guardian inbox subscribes with
`.on('postgres_changes', { event: '*', schema: 'public', table: 'guardian_test_alerts' }, onChange)`
(`src/guardian.ts:187`). The Phase 3 contract therefore carries an `app_alert` block with exactly these
names (`cue`, `level`, `evidence_ms`) rather than renaming them.

**Emergency contacts and consent.** Two separate things. (1) A local test contact,
`src/trips.ts:3` `export type Contact = { name: string; phone: string };`, saved on the phone and used only by
the manual SMS composer. No consent record. (2) Guardians, stored server-side in `guardian_links`: the
driver invites by email (`guardian_invite`, migration `:88`), the guardian must accept (`guardian_accept`,
`:117`), and the state check allows only `pending | accepted | revoked` (`:8`, `:14-18`). An accepted link is
the consent; `location_sharing boolean not null default false` (`:9`) is a separate opt-in.

**Location.** Foreground only. Permission is requested when a session starts, if GPS or location sharing is
on: `App.tsx:273` `Location.requestForegroundPermissionsAsync()`. Fixes come from
`src/useTrip.ts:108` `Location.watchPositionAsync({ accuracy: Location.Accuracy.Balanced, timeInterval: 2000 }, ...)`.
A fix is attached to a guardian alert only if it is under 30 s old and its accuracy is 100 m or better
(`src/useTrip.ts:165-166`), and the server refuses coordinates unless the link has `location_sharing`
(migration `:216-217`). No route is stored. "Precise" in the contract means this path.

**Twilio, ElevenLabs, SMS, calls.** No Twilio or ElevenLabs code. SMS is the system composer only:
`App.tsx:114` `SMS.sendSMSAsync([prefs.contact.phone], 'Conrad driver-monitor TEST. ...')`, triggered by the
Settings button "Compose test SMS" (`App.tsx:145`); the person presses Send. No call code; level 3 is a
simulated "emergency escalation" drill after 18 s of synthetic closure (`src/risk.ts:91`). The in-car voice is
on-device TTS: `src/useTrip.ts:72` `Speech.speak('Attention. If needed, pull over safely.', ...)`.

**What a parent sees during an alert.** The Guardian tab lists rows from `guardian_test_alerts`:
`src/GuardianScreen.tsx:186` `TEST · {cueLabel[entry.cue]}`, the time and seconds of evidence (`:187`), and
`Shared location {lat.toFixed(4)}, {lng.toFixed(4)}` when present (`:189`). It updates while the screen is
open (Realtime plus a 15 s poll, `src/guardian.ts:189`). There is no push notification.

## C. Lean Canvas coverage (CONTEXT.md claims)

Status key: **built**, **partial**, **blocked** (law or evidence, cited), **missing**.

| Section | Claim | Status | Where / why |
| --- | --- | --- | --- |
| Solution | Dashboard-mounted phone camera | partial | App uses the front camera (`src/CameraMonitor.tsx`); Studio runs a webcam. No in-car bench test. |
| Solution | Deep learning detects distraction | built (as a rule) | Eyes off the road > 2 s, NHTSA 78 FR 24818; a fixed rule, not a model. |
| Solution | Detects intoxication | blocked | No licensed, labeled video exists; an unsubstantiated claim violates FTC Act section 5. Reported as `possible_impairment`: observable, baseline-relative indicators, no claim about alcohol. |
| Solution | Other dangerous visual cues | partial | Drowsiness head trained on synthetic data only; possible unresponsiveness added in this change, synthetic-only. |
| Solution | Autonomously alerts emergency contacts | partial | App sends a TEST row to an accepted guardian; the ML signal now recommends tiers. No automated SMS provider. |
| Solution | Autonomously alerts emergency services | blocked | Automated calls to emergency lines: 47 CFR 64.1200(a)(1)(i). `auto_call_911` is false; `emergency_services` is always `never_automated`. |
| UVP | Proprietary deep learning models | missing | Only a synthetic-data candidate exists; no real-data commercial model. |
| UVP | Analyzes observable risky behavior | partial | Closure, gaze, head pose, drowsiness model; validated on synthetic data only. |
| UVP | Audible warnings | built | App tone plus TTS (`src/useTrip.ts:64-76`). |
| UVP | Notify a contact when repeated alerts go unanswered | partial | App: second warning within 2 min (`src/risk.ts:14`, `:87`). ML: Tier 2 on no response or a second alert within `notify.min_gap_s`. |
| Key Metrics | Detection accuracy | partial | Locked-test accuracy gate exists; measured only on synthetic data. |
| Key Metrics | False-alert frequency | partial | False alerts per hour gate (0.2/h); synthetic only. |
| Key Metrics | Notification latency | partial | App records database acknowledgement time; ML now reports driver response latency. Delivery latency to a phone is unmeasured. |
| Key Metrics | Retention, licensing deals | missing | Outside the ML repository. |
| Early adopters | Parents of teen drivers | partial | App guardian prototype; `product.json` pilot plan. No families enrolled. |
| Early adopters | Small and medium dashcam makers | missing | VIOFO A229 Plus is `documented_only`; its front camera faces the road. |
| Early adopters | Small and medium auto insurers | missing | No integration or data agreement. |
