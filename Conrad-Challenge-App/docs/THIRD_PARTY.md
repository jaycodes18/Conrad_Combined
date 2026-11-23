# Bundled vision components

The on-device prototype uses [MediaPipe Tasks Vision](https://www.npmjs.com/package/@mediapipe/tasks-vision/v/0.10.32) version 0.10.32 from Google. The package and its vision WASM runtime are installed through the locked npm dependency. The upstream [MediaPipe license](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE) is Apache 2.0.

The Face Landmarker bundle is the version 1 float16 model linked from Google's [Face Landmarker guide](https://developers.google.com/edge/mediapipe/solutions/vision/face_landmarker). It is stored at `assets/face-landmarker.task`. The source and checksum allow the team to trace the exact pretrained weights used in evaluation.

The app distributes the upstream license text in `assets/mediapipe-license.txt` and exposes it in Settings.

- [Face Landmarker v1 model bundle](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)
- SHA-256 `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`
