# Conrad Driver Monitor

## Project goal

Build a phone-based driver-monitoring prototype for the Conrad Challenge. The demonstration should show that a dashboard-mounted phone can use its front camera to detect sustained eye closure and head turns or tilts, warn the driver audibly, and record what happened. With explicit consent, a controlled test can send an in-app alert to a paired guardian. Prioritize the detection and warning pipeline, measurable performance, and a repeatable parked demo before expanding the UI or integrations.

The app uses React Native and Expo SDK 58. Keep the core demo runnable in Expo Go on iOS and Android and keep the standalone iOS EAS build path viable. Local monitoring and the simulated drill work without an account. App Store readiness depends on physical-device validation, signing, privacy disclosures, and review.

## Repository map

- `App.tsx` contains the Drive, History, and Settings screens and the active-session UI. `src/useTrip.ts` coordinates session lifecycle, audio, optional GPS, and test alert delivery.
- `src/CameraMonitor.tsx` takes sequential front-camera samples and removes temporary JPEGs. `src/visionHtml.ts` loads bundled MediaPipe Face Landmarker in a local WebView and returns timestamped face features.
- `src/risk.ts` is the deterministic risk engine. It calibrates a neutral pose and uses elapsed time across observations to classify sustained cues. `src/demo.ts` supplies synthetic observations for parked drills.
- `src/trips.ts` defines trip metadata and preferences. `src/GuardianScreen.tsx` handles pairing and the test inbox. `src/guardian.ts` is the optional Supabase client boundary.
- `supabase/` contains the guardian SQL migration and account deletion function. `app.json` controls native configuration and permissions. `eas.json` defines standalone build profiles.
- `README.md` documents setup and deployment. `docs/VALIDATION.md` defines the parked test protocol. `docs/THIRD_PARTY.md` records model provenance and licensing.

## Safety and privacy constraints

- Run demonstrations and evaluations with the vehicle parked and stationary. Describe eye closure and head position as observed cues. Head orientation is a proxy for distraction. Avoid claims about intoxication, consciousness, fitness to drive, or guaranteed incident prevention.
- Keep monitoring foreground-only. End the session when the app leaves the foreground or the camera fails. Surface lost tracking clearly. Missing observations and excessive gaps must break evidence accumulation. Keep transient blinks and brief glances below alert thresholds.
- Keep the active-session UI simple and the warning audible. Level 1 warns the driver. Level 2 records one guardian test escalation per session and can deliver it to the selected, accepted link when the driver opts in. Level 3 is a labeled drill simulation. The SMS composer is a separate, manual parked test.
- Process frames on the phone and discard temporary images. Persist trip and event metadata only. Keep GPS optional and avoid storing a location route. Attach a recent location fix to a guardian test alert only when both the link and the session allow it.
- Require verified accounts and accepted links for guardian access. Preserve row-level security, revocation, and one-alert-per-driver-session deduplication. Use test accounts and a dedicated test project. Keep service-role credentials server-side. Expo public environment variables can contain the project URL and publishable key.
- Treat background capture, CarPlay, vehicle telemetry, dashcams, automatic SMS and push, and emergency-service integration as future work. Changes to those boundaries need their own design and validation.

## Working in this repo

- Use Node.js 22.13+ in the 22.x line or an Expo SDK 58-supported newer release. Run `npm ci` to install locked dependencies and generate `assets/runtime/` via `scripts/prepare-assets.cjs`. Run `npm start` for the Expo Go QR code.
- Check Expo Go compatibility before adding native dependencies. Use `npx expo install` for Expo-managed package versions and update the lockfile with dependency changes.
- Keep camera capture and feature extraction separate from risk rules. Reuse the timestamped observation interface for drills and any future camera adapters. Keep risk logic independently testable without React Native or a network.
- Keep secrets, local environment files, generated runtime assets, native build output, and private test data out of Git. Use `.env.example` for documented configuration. Review the target project before any hosted migration or function deployment.
- Update the README or validation protocol when setup steps, permissions, test flows, or alert behavior change.

## Verification

Run `npm run typecheck` and `npm test` for code changes. The tests exercise risk transitions and guardian SQL access rules locally. For changes to native dependencies, asset packaging, or permissions, also run:

```sh
npx expo-doctor
npx expo export --platform ios --output-dir dist/ios
npx expo export --platform android --output-dir dist/android
```

Use [the parked validation protocol](docs/VALIDATION.md) for camera cadence, audio, app switching, tracking loss, false alarms, and end-to-end guardian receipt. Report which checks ran. A bundle export, desktop inference test, and deterministic drill cover different parts of the system. Report phone behavior, detector accuracy, live delivery, and App Store readiness only after their respective checks.
