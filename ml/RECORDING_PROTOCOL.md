# Recording team sessions

This is how the team records its own data for Driven. It is the only data that can ship a model on the
commercial track, and the only data that shows the model working **inside one recording**, which a change of
camera cannot fake.

## Safety first (rule L11)

- **The car is parked, the engine is off, and nobody drives.** Every drowsy moment is acted.
- Nobody records while drowsy, after drinking, or on any medication that affects alertness.
- Stop if anyone feels unwell or uncomfortable.

## Before recording

1. Every person recorded has a row in `recordings/consent.csv` (rule L5):

   | column | what to write |
   |---|---|
   | `subject_id` | a short code such as `T07`, the same as their folder name |
   | `biometric_notice_given` | `yes` once they have read the face-data notice |
   | `subject_consent_date` | the date they signed, as `YYYY-MM-DD` |
   | `is_minor` | `yes` or `no`. Nothing else is accepted |
   | `guardian_consent_date` | for a minor, the date a parent or guardian signed, as `YYYY-MM-DD` |
   | `replay_only` | `yes` keeps this person out of training, validation and test, so their sessions only check the alerts. Otherwise `no` |
   | `recorded_by` | the team member who ran the session |

   A blank or unclear answer means the person's recordings are not used.

2. Put the phone in the dashboard mount used for the drive. **Do not move it between the two clips.**

## Recording one session

Each session gets its own folder, `recordings/<person>/<session>/`, for example `recordings/T07/2026-11-14a/`.
It holds three files.

### 1. `calibration.mp4`: at least 2 minutes

The driver sits normally, alert, looking ahead as if driving. Normal blinking is fine. Driven uses this clip
as the driver's personal baseline, just like the phone's setup step.

### 2. `drive.mp4`: straight after calibration

Same mount, same seat position, same sitting, recorded right away. The driver follows a printed script of
acted events, with **at least 60 seconds of normal, alert sitting between events**:

| event | what to act | length |
|---|---|---|
| `acted_drowsiness` | slow eye closures of about a second, again and again, with the head sinking a little | 30 to 60 s |
| `acted_yawn` | one or two big yawns | 5 to 15 s |
| `acted_nod` | the head drops and snaps back up, once | 1 to 3 s |
| `acted_eyes_off` | look at the passenger seat | 3 s |

A good drive is 10 to 15 minutes long and has at least two `acted_drowsiness` events.

### 3. `script.csv`: written live

A **second team member** writes the script **while watching**, not afterwards from memory. They start a
stopwatch when the drive recording starts and write one row per event:

```
start_s,end_s,event
150,190,acted_drowsiness
330,340,acted_yawn
480,483,acted_nod
600,603,acted_eyes_off
```

The times are seconds from the start of `drive.mp4`. Each event must end after it starts, and the event names
must be exactly the four above.

## How Driven uses the session

- The calibration clip becomes the baseline for this session only. It is never a training window.
- Drive windows that overlap a drowsiness, yawn or nod event are labeled drowsy. Windows at least 30 seconds
  from every event are labeled alert. Windows that are partly an event, or that overlap looking away, are left
  out: eyes-off is a fixed in-car rule, not the model's job.
- People marked `replay_only = yes` are held out of training, validation and test. Their sessions only check
  that the in-car alert catches the acted drowsiness. Mark at least two people this way: a model cannot ship
  without at least 2 replay people and 6 acted drowsiness events between them and the locked test people.
- A model can ship only if it separates drowsy from alert windows **inside the same drive clip** of people it
  never trained on (within-recording AUC of at least 0.80).

## Checklist

- [ ] Consent row filled in, with dates
- [ ] Phone in the mount, not moved between clips
- [ ] `calibration.mp4`: 2 minutes or more, alert
- [ ] `drive.mp4`: scripted events, 60 s or more of alert sitting between them
- [ ] `script.csv`: written live by a second person, seconds from the start of the drive
