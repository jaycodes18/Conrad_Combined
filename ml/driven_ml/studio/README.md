# Driven Studio

A local dashboard for the team and judges to see the product work: the six indicators, the fused condition and the escalation tiers on a live face, the conditions they survive, and the pipeline training the drowsiness model. It runs the same package code the product uses (`driven_ml.indicators`, `driven_ml.risk.RiskSession`, `frame_features`, `predict_probability`, `STAGE_RUNNERS`), with OpenCV for video and MediaPipe Face Landmarker for the face. What you see is the product logic itself, not a copy.

## Start it

Only a virtual environment and Python. Run from the repository root:

```
python3.11 -m venv .venv                          # once
.venv/bin/python -m pip install -e "ml[dev]"      # OpenCV and MediaPipe install with the package
cd ml
../.venv/bin/python -m driven_ml.studio           # then open http://127.0.0.1:8765
```

On a Mac, the first webcam start asks for camera permission for the app that runs Python (Terminal, iTerm or your IDE). The first live start downloads Google's Face Landmarker model (3.6 MB) into `driven_work/studio/mediapipe/`.

To test the selected local camera directly, run:

```
cd ml
../.venv/bin/python -m driven_ml.studio --source 0
```

In the Drive tab, type another camera index to `1`, `2`, and so on to select another OpenCV camera. Live streams request a one-frame capture buffer and retry after read/open failures; timing and dropped-frame counters are visible in the dashboard. This code path is not a hardware-compatibility claim until tested with the actual camera.

Microphone support is optional. Install it with `../.venv/bin/python -m pip install -e ".[studio-audio]"` from `ml/`. The Studio keeps both legal-opinion flags off by default; device selection and capture remain unavailable until they are explicitly authorized and enabled. Start is a separate user action, and the OS may request microphone permission.

Options:
- `--source 0 | synthetic | rtsp://… | http://… | file.mp4` starts live at launch. `--speed` sets the playback speed for files and synthetic data.
- `--model path/to/model.json` loads a specific model. Without it, the studio uses the last candidate trained in its workspace.
- `--workspace` sets where the Training tab writes. The default is `driven_work/studio`.

The condition test also runs on its own:

```
../.venv/bin/python -m driven_ml.studio.conditions --source 0          # prompts: eyes open, then closed
../.venv/bin/python -m driven_ml.studio.conditions --video drive.mp4 --image me.jpg --closed-image me_closed.jpg
```

## Tabs

**Drive** (the default).
- **Strip above the stream:** source (`synthetic`, `0`, `1`, an `rtsp://` or `http://` stream, a file), Start, Stop, Recalibrate, and **Driving**, which stands in for GPS motion. Untick it to check that drowsy, distracted and impairment never fire while parked.
- **Stream:** the first 2 minutes are calibration (sit normally, alert, looking ahead). Under the stream, the coverage bar shows each indicator as available (green), degraded (amber) or unavailable (grey).
- **Condition:** one large label (none, distracted, drowsy, possible impairment, possible unresponsiveness) in its level colour, with score, confidence, the evidence behind it, and any degraded-monitoring situation.
- **Indicators:** six cards (eyes, ocular, head, breathing, responsiveness, sensor quality), each with state, main value and units, and a confidence bar. An unavailable indicator says why in grey; with sunglasses, eyes and ocular go grey and head keeps running.
- **Escalation:** Tier 0 to 3. During a response check it counts down the 5 s window. "Contacts would be texted with precise location" and "Voice call would be recommended" appear when those flags flip. Nothing is ever sent, and emergency services are never contacted.
- **Signal JSON:** the exact `DriverRiskSignal` the app would receive, with a copy button. The SSE stream carries it as `risk_signal` on every frame.
- **Contacts drawer:** add a contact with a consent date and, separately, consent to automated calls (Tier 3 needs it). It also holds the escalation log and the research microphone, which stays blocked until both legal flags are true.

**Test.** Merges the condition test and the scenario lab into one results table: condition or source, event, detected, latency, false alarms, verdict.
- **Scripted condition suite:** pick conditions (clean, sunglasses, IR, low light, backlit, glare, blur, fisheye, hand) and scripted events (5 s closure, head drop with no response, 10 s erratic gaze, road noise, parked silence, nodding), then Run. Each runs through indicators, fusion and escalation on synthetic frames and synthetic audio. These are modelled conditions, not cabin recordings.
- **Captured frames:** with Drive running on a camera, capture 4 s with eyes open and 4 s with eyes closed, then run the image conditions (15 OpenCV transforms through MediaPipe) or the vision scenarios.
- **Audio scenarios:** synthetic PCM through the microphone health and breathing analyzers.
- Hardware-only rows show `NOT RUN`.

**Train.** Runs `dev` (synthetic drivers), `train` or `staging`, stage by stage: stage chips, the person-grouped validation table with AUC and balanced accuracy twice (full features, and every eye input masked, as with sunglasses), held-out AUC by epoch, and the log with every release gate. The new model then loads into Drive.

The status report that used to be a tab prints with `../.venv/bin/python -m driven_ml.studio.status`.

## Using a VIOFO A229 Plus (or another Novatek dashcam)

1. The A229 Plus front and rear cameras face the road. To see the driver you need the **3CH version or the RF200 infrared interior camera**, mounted by the rear-view mirror.
2. Select the interior camera as the live-view channel in the VIOFO app. The camera streams one channel at a time.
3. Join this computer to the camera's Wi-Fi, or put the camera in Wi-Fi Station Mode (hold REC and Wi-Fi) on the same network.
4. Use the source `rtsp://192.168.1.254`. If it shows nothing, send the Novatek wake command first: `POST /api/live/start` with `"novatek_live_view": true`, or try the HTTP live view.
5. Set the driver region and fisheye FOV with `POST /api/live/preprocess`. Use 1CH or a lower resolution while testing: reviewers report RTSP failures and reboots at 1440p60+1440p30.

## Privacy

The server binds to 127.0.0.1. Frames stay in memory, and nothing is saved or uploaded (rule L6). The risk signal carries no pixels, landmarks, audio or location. Condition-test captures live in memory until you clear them or stop the studio. Research audio is off by default and is never written to disk. The studio is a development tool and is never shipped in the app.
