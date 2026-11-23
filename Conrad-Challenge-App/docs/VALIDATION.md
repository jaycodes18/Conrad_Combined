# Validation protocol

## Safe demo boundary

Run the demo only while parked, with the vehicle in Park, parking brake set, and engine off. Keep it on the demo screen: its cues are simulated and do not establish that the camera detector, notification path, or any driving behavior is safe. Do not use the drill while driving or present a demo alert as a real incident. Verify that the screen identifies the drill as simulated before starting it.

1. Start with **Settings → Test warning tone**. Verify the tone at the device volume you will use. Test the iPhone silent switch separately.
2. Run the eye-closure drill. From start, expect a warning near 6 seconds, a guardian test escalation near 12 seconds, and the simulated emergency escalation near 22 seconds.
3. Run a separate camera session. After calibration, record a short blink and a short glance, then try a sustained head turn that keeps the face visible. End the session and inspect the events and the longest sample gap.
4. To test guardian delivery, pair two verified test accounts, accept the invitation, select the guardian for the next drill, and keep the guardian inbox open on the second phone. Use synchronized clocks and record the time of receipt there. The driver's delivery timer measures server acknowledgement.
5. Switch apps during a short camera session and then return. Check that the session ended and that History records the interruption.
6. On Android, repeat with larger display text. Check that the camera preview stays in place after calibration and the **End session** button stays visible. Confirm that a lost face or delayed frames show the corresponding status instead of advancing the cue. Check subsequent sustained cues after tracking returns.
7. Open **Map** with internet access and confirm you can pan and zoom the Dallas demo pin. Deny location on the first try and confirm the demo stays visible. Then grant location and confirm the pin moves on **Use my location**. Open another tab and return to verify that the map resets to Dallas. Test the retry message by opening the map with internet temporarily disabled.

## Empirical measurement

Treat the deterministic tests as checks of programmed timing and state transitions. They do not measure detector accuracy or prove that alerts reach a person. Collect real-world measurements in a stationary, supervised session first, with consent from every participant and no driving task underway.

For each session, log timestamps for camera frames, detected face/eyes/head cues, app alert creation, notification request, and observed notification receipt. Keep participant/session identifiers pseudonymous. Record device model, OS/app build, camera position, lighting, glasses status, app foreground/background state, and any tracking gaps. Review video or have a second observer label the true cue intervals so detector outcomes can be compared against an independent reference.

Use **History → Export trip metadata** for event times, evidence duration, processing time, sample counts, maximum sample gap, and server acknowledgement. Camera sessions also include the time to the first usable frame, longest capture-to-result time, and count of results dropped for taking more than 900 ms. A clear frame without a face is usable for tracking status. The export contains aggregate timing and event metadata. To collect per-frame labels for a research run, use independent observation or temporary instrumentation on a dedicated test build.

Report these metrics with counts and denominators:

- **Precision:** true positive alert episodes divided by all alert episodes.
- **Recall:** reference-positive episodes detected divided by all reference-positive episodes.
- **False positives per hour:** false alert episodes divided by valid monitored hours; exclude time with no face/tracking coverage and report that excluded duration separately.
- **End-to-end timing:** time from reference cue onset to in-app alert creation, including median and 95th percentile.
- **Notification delivery latency:** time from notification request to confirmed receipt on the target device, including median, 95th percentile, and delivery failures.
- **Sampling/thermal behavior:** observed frame/sample rate, dropped or delayed samples, tracking coverage, device temperature/thermal state, and session duration.

Use a repeated-measures matrix rather than a single best-case run. Include app foreground and background/app switching; face temporarily missing and reacquired; bright, dim, and uneven lighting; with and without glasses; and multiple head/face positions. Repeat each condition across multiple sessions and participants. Report each condition separately as well as pooled results, along with sample sizes and confidence intervals. Do not claim accuracy from the simulated drill or from unreviewed detector output.

## Deterministic behavior checks

`npm test` covers calibration sampling, cue episode/rearm behavior, escalation limits, missing-face recovery, timestamp ordering, and the simulated demo boundaries. These tests validate software behavior only; empirical metrics above require device-side observation and independent labels.
