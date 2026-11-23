# What Driven can and cannot do today

An assessment of the product against the Lean Canvas, the conditions a car puts the camera in, and the VIOFO A229 Plus. Measured numbers come from `python -m driven_ml.studio.conditions`; everything else is stated as what exists in this repository on 2026-11-22. Rerun the condition test on your own face (see the end) before relying on any row.

## 1. The model has not met a real driver yet

The model has never been trained or tested on real drivers. Every model in this repository was trained on synthetic data, and synthetic models can never be promoted (rule L9).

- **Dev numbers are meaningless.** The dev pipeline passes staging with AUC near 1.0, but the synthetic data separates perfectly; the Training tab shows it.
- **No team recordings exist.** The real-data gates are built: within-recording AUC of at least 0.80, and at least 2 held-out replay people. With no recordings, a real model cannot pass them yet.
- **Next step:** record team sessions with `ml/RECORDING_PROTOCOL.md`. Nothing else on this page matters until that exists.

## 2. The Lean Canvas, item by item

| Canvas item | Status | Detail |
|---|---|---|
| Detect drowsiness from a phone camera | Built, untested on people | Face landmarks, 60 s windows, personal baseline, calibrated probability, 120 s cooldown. See section 1. |
| Detect distraction | Partly | Eyes off the road for 2 s or more, measured from the driver's own calibrated centre (NHTSA guideline). Phone-in-hand, eating and other distractions are not detected. |
| Detect intoxication or impairment | Not doable | No licensed, labeled video of impaired drivers exists, and the claim is barred without evidence (FTC Act section 5, rule L8). |
| Audible warning in the car | App side | The model decides when to warn; the app speaks it. The studio shows the decision but plays no sound. |
| Notify a contact when repeated alerts go unanswered | Simulated | The backend texts confirmed contacts per drowsiness alert. The "unanswered" rule (a second alert within 10 minutes) runs only in the studio. |
| Alert emergency services | Not doable | Automated calls to 911 are prohibited without consent (47 CFR 64.1200, rule L1). Parents get a text and decide whether to call. |
| Measure accuracy and false alerts | Built | AUC, sensitivity, false alerts per hour (gated at 0.2/h) and within-recording AUC, on locked test people. |
| Measure how fast notifications arrive | Not built | Nothing records the time from an alert to a delivered text. It needs Twilio delivery callbacks stored per notification. |
| Works on dashcams (licensing) | Not validated | The studio reads a dashcam stream, and the OpenCV crop and fisheye fixes work in the condition test. No real cabin-camera recording has been scored. |
| Deep models trained on large unique datasets | Partly | MediaPipe is a pre-trained deep face model; Driven trains a small logistic or 16-unit MLP head on it. The unique dataset is the team's recordings, which do not exist yet. |

## 3. Conditions, measured

Each condition is an OpenCV transform of real face frames, run through MediaPipe Face Landmarker and the same frame features the model uses. The run below used 12 jittered frames of one public-domain portrait (NASA, via scikit-image) with eyes open. That makes it a single face, with no eyes-closed frames, so **closure separation was not measured**.

Column meanings:
- **Face**: frames with a face. Windows go unscored below 60%.
- **Eyes** and **Mouth**: contrast inside the eye and mouth outlines, against the clean frames.
- **With OpenCV fix**: the same frames after the matching fix in `features/preprocess.py`.

| Condition | Face | Eyes | Mouth | As captured | With OpenCV fix |
|---|---|---|---|---|---|
| Clean daylight, phone in front | 100% | 100% | 100% | works | |
| Dim cabin (dusk, overcast) | 100% | 36% | 37% | degraded | **works** with CLAHE (93% eyes) |
| Night, no infrared light | 0% | 0% | 0% | **fails** | **fails**: no software fix |
| Night with an infrared cabin camera | 100% | 49% | 52% | degraded | works with CLAHE on the photo; on video frames CLAHE cut face detection to 92%, so test before enabling |
| Backlit (bright windows behind) | 100% | 30% | 31% | degraded | still degraded: needs camera HDR or exposure, not post-processing |
| Hard sun on one side | 100% | 128% | 107% | works | |
| Motion blur (bumps) | 100% | 50% | 70% | degraded (borderline) | none; a stiffer mount helps |
| Low resolution (one fifth) | 100% | 47% | 69% | degraded | none; use at least 640x480 on the face |
| Heavy stream compression | 100% | 97% | 97% | works | |
| Cabin camera at the mirror: small face | 0% | 0% | 0% | **fails** | **works** with a driver-seat crop (93% eyes) |
| 210-degree fisheye cabin camera | 0% | 0% | 0% | **fails** | **works** with fisheye correction and the seat crop (89% eyes) |
| Seen from the side (planar approximation) | 100% | 92% | 94% | works | a real 3D side view is likely worse: record it |
| Dark sunglasses | 100% | 3% | 100% | **fails** | none: MediaPipe still places eye points, but it is guessing |
| Glare on clear glasses | 100% | 96% | 99% | works | |
| Hand or mask over the lower face | 100% | 99% | 4% | degraded | none: yawns cannot be seen; eyes still work |

**What this means for the product.**

- **Works now:**
  - a phone in front of the driver in daylight, dusk (with the low-light fix) or hard sun;
  - compressed streams;
  - clear glasses.
- **Works only with the OpenCV fixes:**
  - a mirror-mounted or fisheye cabin camera, with the crop set once per car around the driver's head;
  - a dim cabin.
- **Fails, with no software fix:**
  - **Night without infrared light:** a phone camera alone cannot run at night. The VIOFO cabin camera's infrared LEDs are what make night possible.
  - **Dark sunglasses:** the drowsiness signal is lost. Many lenses pass near-infrared, so an IR camera may see through them; that is untested.
  - **Backlighting:** needs camera exposure settings or HDR, not post-processing.
- **Not measured yet:**
  - **Closure separation** on real closed eyes. It needs your own eyes-closed capture (below).
  - **Glasses, beards and face shapes** across different people.
  - **A real 3D side view.** The side-angle row is a flat approximation.

**The browser app cannot run these fixes yet.** The OpenCV fixes run in Python: the studio, the condition test and video extraction for training. A browser phone app would need OpenCV.js or a native app to do the same. Train and deploy with the same preparation, or the features shift.

## 4. The VIOFO A229 Plus

- **It needs a cabin camera.** The A229 Plus front and rear cameras face the road. Watching the driver needs the 3CH version or the RF200 infrared interior camera, which mounts by the rear-view mirror. That is the "small face" and "side view" rows above.
- **There is no WebRTC and no official API.** In Wi-Fi Station Mode the camera serves files over HTTP and live video over RTSP at `rtsp://192.168.1.254`. An undocumented Novatek command interface (`/?custom=1&cmd=...`) includes live-view start (2015).
- **It streams one channel at a time.** Select the cabin camera in the VIOFO app.
- **RTSP is unreliable at high settings.** Reviewers report RTSP failures and reboots on the A229 Plus at 1440p60+1440p30.
- **Browsers cannot play RTSP.** The studio reads it with OpenCV. A phone app needs a native video player, or a relay that converts RTSP to WebRTC.

## 5. Run the condition test on your own face

From the repository root, with the virtual environment:

```
.venv/bin/python -m pip install -e "ml[dev]"
cd ml
../.venv/bin/python -m driven_ml.studio.conditions --source 0 --json driven_work/conditions.json
```

The command first asks you to look at the camera with your eyes open, then to close your eyes. It then prints the same table with the closure-separation column filled in. The studio's Conditions tab does the same from the Live tab's camera, including the dashcam.

## 6. What each condition is based on, and what it is not

Every condition below comes from rules in `driven_ml/risk/fusion.py` over the six indicators in
`driven_ml/indicators`. The thresholds are named constants, not weights learned from Driven data, because no
labelled Driven sessions exist. Each was tested on synthetic frames and synthetic audio only. The signal's
`model.experimental` field lists every indicator for that reason.

**Distracted.** Based on the gaze or head turned off the driver's calibrated centre for more than 2 s while
moving: NHTSA's visual-manual guideline (78 FR 24818, 2013), applied as a fixed rule. It is not a measure of
attention, and it does not know where the road is. Erratic gaze with nothing else is shown at a low level and
never alerts.

**Drowsy.** Based on the 60 s drowsiness model (P80 PERCLOS, blinks, nods and the other window features against
the personal baseline), eyes closed 2 s or more, two such closures in 60 s, or repeated nods. The model was
trained and staged on synthetic people only, so its accuracy on real drivers is unknown.

**Possible impairment.** Based on two or more independent indicator families being abnormal at once against the
driver's own calibration, while moving: erratic eye movement (5 s or 10 s), head nodding, the drowsiness model
at its threshold, and repeated eye closures. It is not intoxication detection. It does not measure alcohol, and
nothing in the repository was trained or tested on impaired drivers. Field sobriety cues such as nystagmus need
saccade velocities a 30 fps phone cannot resolve; the ocular indicator measures gaze dispersion and reversals
instead, so Koch et al.'s eye-tracker results (CHI 2023, CHI 2025) do not transfer. One family alone, however
strong, is never reported as impairment.

**Possible unresponsiveness.** Based on eyes closed 5 s or more with the face tracked, plus one independent
confirming sign: a dropped or slumped head, no visible response within 5 s of the in-car alert, or (when the
gated microphone is on) a suspected agonal pattern or no breath sounds in a quiet parked cabin. With sunglasses
it is a head drop held 5 s with no response. It is not a finding of unconsciousness or of any medical state.
Breathing never triggers it alone and regular breathing never cancels it. A head-down pose that makes MediaPipe
lose the face is reported as unavailable, not as a drop, so a real slump can go unconfirmed.

**Monitoring degraded.** Based on heuristics: eye-region contrast against calibration, luminance, sharpness,
glare pixels, grayscale (IR) frames and a configured fisheye lens. The regression suite models sunglasses, IR, low
light, backlight, glare, blur, fisheye and a hand over the mouth as their effect on those inputs; with all eight
it raised no false possible-unresponsiveness or possible-impairment alarm in normal driving, caught the 5 s
closure whenever the eyes were visible, and caught the head drop with no response when they were not. Those are
modelled conditions, not recordings of real glasses on real faces in a real car.
