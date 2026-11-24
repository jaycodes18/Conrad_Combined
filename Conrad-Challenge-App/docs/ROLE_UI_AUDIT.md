# Audit: teen and parent interfaces, model alignment, map

Scope: `~/Desktop/Conrad_Combined` at commit `c6cf08e` ("Import the Conrad phone app and Driven ML pipeline"), the Expo app in `Conrad-Challenge-App/` and the Python package in `ml/`.

## 1. What the combined code did before this change

| Area | Finding | Evidence |
|---|---|---|
| Interfaces | One interface for everyone. Drive, Map, History, Guardian and Settings tabs were shown to teen and parent alike. The Guardian tab mixed the teen's invite flow and the parent's inbox on one screen. | `App.tsx` `tabs = ['Drive', 'Map', 'History', 'Guardian', 'Settings']` |
| Map | Showed the phone's own position or a Dallas demo pin. It never showed where an alert happened. The parent saw shared coordinates as text. | `src/MapScreen.tsx` `DEMO_PIN`; `src/GuardianScreen.tsx` "Shared location {lat}, {lng}" |
| Live risk engine | The phone ran its own two-cue heuristic: eye closure 2.1 s, head turn 2.6 s, guardian step after 8 s of continuous evidence or a repeat within 2 min. It had no 5-second critical line, no response check and no condition or tier. | `src/risk.ts` `WARNING_AFTER`, `evidenceMs >= 8000` |
| ML link | The Python `DriverRiskSignal` reached the app only through replay. Only `app_alert.level === 1` was used. Condition, the six indicators, the tier, `share_location` and `voice_call_recommended` were ignored. | `src/mlReplay.ts` |
| Indicators the phone can measure | Eyelid closure (blendshapes) and head yaw and pitch (transform matrix) only. No iris, so no gaze instability. The microphone is blocked in `app.json`, so no breathing. | `src/visionHtml.ts:98-105`, `app.json` `blockedPermissions: RECORD_AUDIO` |
| Database | `guardian_test_alerts` stores cue, `level = 2` and `evidence_ms`, plus an optional location. It has no condition, tier, score or response-check result. | migration `:29-45` |
| Wording | The app follows the no-diagnosis rule. No "drunk", "intoxicated" or "unconscious" appears in UI strings. | grep over `src/`, `App.tsx` |

## 2. What changed

### 2.1 Model alignment (`src/risk.ts`)

- **5-second critical closure.** Five seconds of closed eyes escalates straight to the guardian step (`CRITICAL_CLOSURE_MS = 5000`). That matches `closure_critical_seconds = 5.0` in the ML eyes indicator. The 8-second line stays for head turns.
- **Response check.** After every driver alert, a 5-second window opens (`RESPONSE_WINDOW_MS`), matching the ML `response_window_seconds`.
  - **Responded:** eyes open and head back to centre, held for 0.5 s. The latency is recorded.
  - **No response:** the window closes first. This triggers the guardian step, so a teen who stays turned away escalates at about 7.6 s instead of 8 s.
  - **Unavailable:** the face is lost during the window. It is not counted as no response.
- **Snapshot.** Every update returns a snapshot with:
  - `condition`: `none`, `distracted`, `drowsy` or `possible_unresponsiveness`;
  - `tier`: 0 to 3;
  - the held seconds for eyes and head;
  - the response-check state.

### 2.2 One live view for phone and ML (`src/liveView.ts`)

- **Same six tiles.** `viewFromEngine` (the phone engine) and `viewFromSignal` (a Driven ML replay) produce the same six tiles, named as in the ML contract:
  - eyes;
  - eye movement;
  - head;
  - breathing;
  - response check;
  - camera.
- **Unmeasured tiles show as off.** Eye movement and breathing read **Off**, with the reason, on the phone path. They are never shown as healthy.
- **Parent titles.** Guardian alert titles come from cue and evidence: eye closure of 5 s or more reads "Possible unresponsiveness", eye closure under 5 s reads "Drowsiness signs", and a head turn reads "Eyes off the road".

### 2.3 Two interfaces

- **Role choice.** The first launch asks who uses the phone. The role is a device preference and can be switched in Settings.
- **Teen tabs:**
  - **Drive:** start monitoring, choose who gets alerts this trip, the location toggle, parked drills and ML replays.
  - **Trips:** history with summary statistics.
  - **Family:** invite a guardian and control location sharing.
  - **Settings.**
- **Teen live screen:**
  - a full-screen response banner;
  - the camera preview;
  - a status orb coloured by condition;
  - the six-indicator grid;
  - the escalation ladder (watching, driver alert, guardian alert, call step).
- **Parent tabs:**
  - **Home:** a status orb for the last 24 h, 7-day statistics, the latest alerts, connected drivers and pending invitations.
  - **Alerts:** a dark map with a pin for every alert that shared a location, coloured by severity; tapping an alert focuses its pin.
  - **Family:** accept an invitation, see drivers, disconnect.
  - **Settings.**
- **Sample data.** A parent phone without a connected account sees three alerts clearly labeled **Sample data**. These are never mixed with real rows.
- **Access is still enforced by the server.** A teen who switches the phone to parent mode sees only the inbox their own account is allowed to read under the existing row-level security.

### 2.4 Design system (`src/ui.tsx`)

- **Palette:** a deep navy base, one electric-blue brand colour, and green, amber, rose and red status colours used only for status.
- **Shapes:** 22 pt card radius, 16 pt buttons.
- **Text:** heavy tight headings, uppercase section labels.
- **Components:** status orb, pills, stat tiles, chips, a tab bar with an active marker, a danger button.
- **Map:** OSM tiles are dark-filtered to match. Labels are inserted as text, not HTML.

## 3. Verification

- `tsc --noEmit`: passes.
- `node --test tests/*.test.mjs`: 28 of 28 pass. This adds `tests/engineAlignment.test.mjs` (7 tests) and leaves the original 21 unchanged. The new tests cover:
  - 5 s escalation;
  - no-response escalation;
  - a response inside the window;
  - face loss during the check;
  - the six-tile view;
  - the replay mapping;
  - banned wording.
- `expo export --platform ios`: Metro resolved and bundled every module. The Hermes bytecode step failed only because the audit VM is arm64 Linux and `hermesc/linux64-bin` is x86-64. Run the export on the Mac to confirm.
- No physical phone, camera, guardian delivery or map tile load was exercised in this audit.

## 4. Gaps that remain, most important first

1. **No push notification to the parent.** Alerts arrive only while the parent app is open (Realtime plus a 15 s poll, `src/guardian.ts`). A parent product needs server-sent push (Expo push or APNs/FCM) from a Supabase function on insert.
2. **The alert row cannot say what the model concluded.** Add nullable `condition`, `tier` and `response_check` columns to `guardian_test_alerts`, and a v2 of `guardian_send_test_alert` that accepts them. Until then the parent title is derived from cue and evidence.
3. **Erratic eye movement is not measured on the phone.** `visionHtml.ts` reads blendshapes and the head matrix only. Reading iris landmarks 468–477 and porting `ml/driven_ml/indicators/ocular.py` would light up the Eye movement tile and the ML impairment rule, which needs two indicator families.
4. **Breathing stays off.** `RECORD_AUDIO` is blocked and the ML audio path is gated behind legal review. This is by design.
5. **Two engines.** Live phone sessions use the TypeScript engine, and replays use the Python fusion output. Their thresholds now match for closure and response, but the impairment rule exists only in Python.
6. **The call step is a drill only.** `voice_call_recommended` is not wired, and `guardian_links` has no consent-for-calls field. Add that consent before any ElevenLabs integration.
7. **CI still runs `make setup`** in `.github/workflows/ci.yml`.

## 5. Files

Changed:
- `App.tsx`
- `src/ui.tsx`
- `src/risk.ts`
- `src/trips.ts` (adds `role`)
- `src/useTrip.ts` (exposes `view`)
- `src/mapHtml.ts` (multi-pin, dark)
- `src/MapScreen.tsx` (now `AlertMap`)
- `src/GuardianScreen.tsx` (role-aware `FamilyScreen`)

New:
- `src/liveView.ts`
- `src/screens/DriverScreens.tsx`
- `src/screens/ParentScreens.tsx`
- `src/screens/Settings.tsx`
- `src/screens/parts.tsx`
- `tests/engineAlignment.test.mjs`
