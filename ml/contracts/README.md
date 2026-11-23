# DriverRiskSignal v1: integration note for the app

The ML side emits one `DriverRiskSignal` per camera frame (about 10 per second). It describes what the camera
and, when allowed, the microphone can observe, and it recommends what to do. The app decides what to do and
does it.

- Schema: `driver_risk_signal.v1.schema.json` (JSON Schema 2020-12)
- TypeScript type: `driver_risk_signal.v1.ts`
- Golden examples: `fixtures/` (none, distracted, drowsy, possible impairment, possible unresponsiveness, and
  degraded with sunglasses). They come from running scripted sessions; `python -m driven_ml.risk.fixtures`
  regenerates them, and `tests/test_contract.py` fails if they drift.

## Phone demo export

From `ml/`, run `python -m driven_ml.risk.phone_demo --app-dir ../Conrad-Challenge-App` to refresh the
app's bundled contract and parked replays. Add `--check` to verify the app checkout matches the Python
output. The Expo app plays snapshots from just before the warning in each synthetic session, using
the signal's own timestamp, cue and evidence duration. The app loads these snapshots from its bundle.

## Fields to read

| Field | Use |
| --- | --- |
| `condition`, `level` | What to show the driver and the parent. `level` is `none`, `low`, `elevated` or `high`. |
| `escalation.in_car_alert` | Play the warning tone and speech now. True on one signal per episode. |
| `escalation.notify_contacts` | Text the consented contacts now. True on one signal per episode. |
| `escalation.share_location` | `precise`: attach the phone's current location to that text. |
| `escalation.voice_call_recommended` | The hook for a future voice call. True on one signal per episode. |
| `escalation.responsiveness_check` | `waiting` during the 5 s response check; show a prompt. |
| `escalation.reason` | One sentence explaining the tier, safe to show a parent. |
| `indicators.<name>.evidence` | Plain sentences a parent can read. Show them under the alert. |
| `monitoring_degraded` | Tell the driver that monitoring is limited (for example, sunglasses), and why. |
| `app_alert` | The app's existing `cue`, `level` and `evidence_ms`, unchanged. |
| `disclaimer` | Show it wherever a condition is shown. |

`emergency_services` is always `never_automated`. The app must not call 911 or any emergency line
automatically (47 CFR 64.1200(a)(1)(i)). It can show the parent a button to call 911 themselves.

## When to text and when to share location

Text when `escalation.notify_contacts` is true. The ML side sets it only on the one signal that enters Tier 2,
and only if a contact with recorded consent exists and the daily cap (`notify.max_sms_per_driver_per_day` in
`product.json`) is not reached. Tier 2 starts when the driver does not respond to the in-car alert, when a second
qualifying alert lands within `notify.min_gap_s` (600 s), or when possible impairment reaches high.

Location is the app's job. The signal never carries coordinates. When `share_location` is `precise`, read the
phone's location the way the app already does:
- `src/useTrip.ts:108` `Location.watchPositionAsync({ accuracy: Location.Accuracy.Balanced, timeInterval: 2000 }, …)`;
- attach a fix only if it is under 30 s old and accurate to 100 m (`src/useTrip.ts:165-166`);
- send it only on a link with `location_sharing` on, which the server already enforces in
  `guardian_send_test_alert` (`supabase/migrations/20261107010000_guardian_test_backend.sql:216-217`).

## Mapping onto what the app already has

The app's own engine is `src/risk.ts` (`RiskAlert = { cue, level, evidenceMs }`). To switch to the ML signal:

1. Read `app_alert` where the app now reads `RiskEngine.update(...).alerts`. `app_alert.level` is the
   escalation tier, the same 1-3 scale as the app's `Level`.
2. On `notify_contacts`, call `guardianGateway.sendTestAlert(linkId, sessionId, { cue, level: 2, evidenceMs,
   location })` (`src/guardian.ts:154`) with `cue` and `evidence_ms` from `app_alert`. The table
   `guardian_test_alerts` accepts only `cue` in `eye-closure | head-away` and `level = 2` (migration `:35-37`).
   Possible impairment and possible unresponsiveness map to those cues today; a column for `condition` is the
   next migration if the parent should see the condition name.
3. The guardian screen (`src/GuardianScreen.tsx:186-189`) can show `escalation.reason` and the evidence
   sentences next to the existing cue label and shared location.
4. Set `session_id` to the app's trip id when the app runs the session: `guardian_test_alerts.session_id` is a
   `uuid` column, and the fixtures' ids (`fixture-…`) are not UUIDs. `driver_ref` is pseudonymous and never a
   name or an email.

## Where the voice call would hook in

There is no ElevenLabs, Twilio or calling code in the app today (only the SMS composer at `App.tsx:114`). When
`voice_call_recommended` is true, a server-side function, not the phone, would place an artificial-voice call
to the contacts listed in the Tier 3 event. The ML side sets the flag only when a contact's consent record covers
automated calls, because an artificial-voice call to a wireless number needs the called party's prior express
consent (47 CFR 64.1200(a)(1)(iii)). That consent must be collected in the app (`guardian_links` has no field
for it yet), and provider keys stay on the server.

## What the signal never contains

Pixels, landmarks, audio, transcripts or GPS coordinates. `tests/test_contract.py` checks every field name.
