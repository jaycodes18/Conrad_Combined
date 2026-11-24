# Conrad Driver Monitor

A React Native and Expo SDK 58 prototype for the Conrad Challenge. Run the prototype with Expo Go on an iPhone or Android phone. Use a stationary, parked vehicle or tabletop setup for every demonstration and evaluation.

## Run on a phone

1. Install Node.js 22.13 or later in the 22.x line, or an Expo SDK 58-supported newer Node.js release.
2. From the app directory, run `npm ci`, then `npm start`. The install prepares the bundled vision and map assets. The phone and computer should be on the same network.
3. Sign in to an Expo account in Expo Go if prompted. On iPhone, scan the terminal QR code with Camera. On Android, scan it from Expo Go. Use an Expo Go build that supports SDK 58.
4. On first launch, choose **I’m the driver** on the dashboard phone and **I’m a parent or guardian** on the parent’s phone. Run **Settings → Play warning tone**, then return to **Drive**. The eye-closure drill plays the tone, logs the guardian test escalation, and demonstrates a simulated emergency escalation in about 25 seconds.
5. Start a camera session, allow camera access, and hold a neutral forward pose while parked. Briefly blink and make a short head turn. Then, still parked, hold a head turn and inspect History after ending the session. See [the validation protocol](docs/VALIDATION.md) for the full test procedure.

In **Drive → Demo while parked**, tap **ML replay · eyes** or **ML replay · head**. The app plays the original warning timing from a short synthetic scenario, sounds the local warning tone, and adds a **Driven ML replay** entry to **Trips**. The live screen shows the six Driven ML indicators, the response check and the escalation ladder. The replay uses bundled `DriverRiskSignal` v1 output from the Python project. It runs without camera access, GPS, a guardian account or a Python server.

On the parent phone, **Map** shows each connected teen as a car. Tap a car or its card to open that driver: the live model view (eyes open or closed and head direction, redrawn from the model's reading; video never leaves the teen's phone), current condition and risk score, the six risk factors, location, and driving history. A teen shares a drive by choosing the parent under **Share this drive with** before starting; the phone then publishes its status every 2.5 seconds and a trip summary at the end (`supabase/migrations/20261124000000_family_live_status.sql`). Live camera: in **Family** the teen allows camera sharing for a parent, then turns on **Camera** on the Drive screen for each drive. The phone relays the same 640×480 frames the face model reads, about three per second, on a private Supabase Realtime Broadcast channel (`video:<driver id>`); nothing is stored, and the RLS policies in `supabase/migrations/20261124010000_family_video.sql` let only that driver send and only accepted guardians with camera sharing receive. The teen's screen shows **FAMILY CAN SEE** while it is on. In the Supabase dashboard, turn off public channel access in Realtime settings so every channel needs these policies. The parent view also runs in a laptop browser with `npm run web`. Without a connected account the parent screens show a labeled **Demo family** of two simulated drivers, and **Alerts** shows three alerts labeled **Sample data**. The bundled map renderer loads tiles for the visible area from [OpenStreetMap](https://www.openstreetmap.org/) and uses an internet connection. If you open an alert location in your browser, OpenStreetMap receives it rounded to four decimal places. The **Settings → Estimate speed with GPS** preference applies to camera sessions.

To refresh this data, run `python -m driven_ml.risk.phone_demo --app-dir <path-to-this-checkout>` from the Driven ML project's `ml/` directory. Commit the resulting files in `src/drivenMl/` together. Add `--check` to verify that the app's copy matches the Python export. Live phone-camera sessions use the app's on-device risk engine.

Local mode works without an account. The phone fetches the app and assets from Metro in Expo Go. A standalone build bundles the model and runtime so inference can run without internet. Optional location, linked guardian delivery, and the SMS composer each depend on device permission or connectivity.

## What this prototype detects

- The foreground front camera supplies sequential low-resolution images to the bundled MediaPipe Face Landmarker. Inference runs inside a local WebView. The monitor removes temporary JPEGs after capture and stores only trip and event metadata.
- The risk engine consumes timestamped bilateral eye-closure and head-orientation signals. A roughly two-second neutral calibration accounts for mount angle. A warning requires about 2.1 seconds of eye closure or 2.6 seconds of head turn/tilt. Missing faces and gaps over 900 ms clear current evidence. If a frame takes more than 900 ms to arrive, the app discards it and shows that tracking is delayed. A new warning needs a clear interval of 1.5 seconds.
- Every warning opens a 5-second response check: eyes open and head forward, held for half a second. A second separated warning within two minutes, no response within the window, five seconds of eye closure, or eight seconds of head turn records one guardian test escalation per session. Automatic in-app delivery requires an accepted guardian link and driver opt-in for that session. Eighteen seconds of synthetic eye closure can demonstrate the simulated emergency workflow.
- Optional GPS estimates speed. A recent, reasonably accurate fix can accompany a guardian test alert when the link and the session both allow it. The app keeps no location route. Local history is capped at 40 sessions and can be cleared or exported from the app.

The model is pretrained for face landmarks and expressions. The risk thresholds are prototype heuristics for observable cues. Head orientation is only a proxy for distraction. The prototype makes no determination about intoxication, consciousness, or fitness to drive. Audio volume, face visibility, sampling rate, glasses, mount placement, and lighting need device testing.

Camera monitoring ends on app switch or screen lock. Plan the test around keeping the app open. Background monitoring, navigation integration, CarPlay, vehicle telemetry, and compatible dashcams are separate future work. A native frame processor in an Expo development build would be the next path if device tests show that the Expo Go snapshot cadence or battery cost is inadequate.

## Optional guardian test project

The two-phone test uses Supabase Auth, Postgres with row-level security, and a foreground guardian inbox. It sends a clearly marked **TEST** row to the paired guardian. The guardian screen can refresh manually and polls while foregrounded. The local SMS composer lets you review and manually send a standalone test text while parked.

1. Set up a dedicated Supabase test project. Keep email confirmation enabled and configure auth email delivery for the test accounts.
2. From this app directory, use the Supabase CLI to sign in, link the test project, apply the migration, and deploy account deletion. The hosted function uses Supabase's built-in server environment variables.
3. Copy `.env.example` to `.env.local`. Fill in the project URL and **publishable** key. Start Metro again after changing the environment.
4. On two phones, create and verify separate test accounts. In **Family**, the driver creates an invitation using the guardian account's email. The guardian confirms the sender and accepts it.
5. On the driver phone, select the accepted link for the next session. Start the parked drill. On the parent phone, leave **Home** or **Alerts** open and verify the received test alert. The driver trip history records database acknowledgement time. Measure guardian screen arrival separately.

```sh
supabase login
supabase link --project-ref YOUR_TEST_PROJECT_REF
supabase db push --remote
supabase functions deploy delete-account --use-api
```

The test environment needs its own release review before it handles real accounts or alerts. This prototype includes no automatic SMS, push delivery, or emergency calls. For a production service, keep any future notification provider credentials on the server and add verified delivery, opt-in, abuse controls, and retention before expanding beyond controlled tests.

## Verify and build

```sh
npm run typecheck
npm test
npx expo export --platform ios --output-dir dist/ios
npx expo export --platform android --output-dir dist/android
```

The deterministic tests cover warning timing, short cues, lost observations, escalation, and guardian SQL access rules. Run the parked protocol on each target phone before reporting real-world detection results. Record runtime latency, sample gaps, false positives, missed cues, battery/thermal behavior, and actual guardian receipt time.

The Expo config includes an icon, app identifiers, foreground camera/location permission descriptions, and EAS profiles. Use your Apple team and confirm the iOS bundle identifier when preparing an internal build or TestFlight release.

```sh
npx eas-cli build --platform ios --profile production
```

For App Store review, complete physical-device validation and add the app's public privacy policy URL, privacy disclosures, support URL, and review notes. If the guardian feature is included in the release, verify the configured account deletion function and provide review access. The EAS profile prepares the standalone build path. Apple signing, a signed build, TestFlight, and App Review are separate checks.

## Technical sources

- [Expo Camera API](https://docs.expo.dev/versions/v58.0.0/sdk/camera/)
- [MediaPipe Face Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker)
- [iOS camera interruptions in the background](https://developer.apple.com/documentation/avfoundation/avcapturesession/interruptionreason/videodevicenotavailableinbackground)
- [Supabase Auth with Expo React Native](https://supabase.com/docs/guides/getting-started/tutorials/with-expo-react-native)
- [Expo iOS production builds](https://docs.expo.dev/tutorial/eas/ios-production-build/)
- [Apple App Review Guidelines](https://developer.apple.com/app-store/review/guidelines/)
