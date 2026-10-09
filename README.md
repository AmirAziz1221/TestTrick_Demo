# TestTrick Proctor v2

AI-assisted exam monitoring that runs in the browser. It watches the candidate's camera, checks that their
eyes stay on the screen, records the session, and gives the reviewer short evidence clips of only the
flagged moments instead of hours of video.

**Live demo:** https://testtrickdemo-production.up.railway.app
**Reviewer page:** https://testtrickdemo-production.up.railway.app/review.html

Built with FastAPI and the MediaPipe Face Landmarker (no YOLO).

---

## Features

- **Automatic screen fit (no calibration dots).** The browser reports the screen size and the server estimates
  the candidate's distance from the camera, so the "eyes on screen" area adapts to every device. The first
  few seconds are used to learn where the candidate's eyes rest; gaze is not judged during that time.
- **Live warnings** shown on the candidate's screen after an issue lasts a few seconds:
  - face missing or only partly visible
  - a second person in view
  - camera blocked or too dark
  - eyes or head turned away from the screen
  - looking down while not typing or using the mouse
  - looking up for a long time ("thinking" is allowed for a while)
- **Typing rule.** Looking down at the keyboard is allowed while the candidate is typing or using the mouse.
- **Recordings.** Camera and screen are recorded in the browser and uploaded in chunks.
- **Evidence clips.** Each flagged moment is cut into a short MP4 (with padding before and after) for review.
- **Reviewer page.** Pick a session and see the report and only the flagged clips.

## How it works

```
Browser (index.html)                        Server (FastAPI)
  camera frame every 0.7 s  ──WebSocket──►  detector.py  (MediaPipe face landmarks)
  screen size + key/mouse events            gaze.py      (eye + head direction, screen area)
  camera/screen recording   ──upload────►   analysis.py  (timers, issues, warnings, summary)
  ◄── warning / status ─────────────────    clips.py     (ffmpeg evidence clips)
                                            storage.py   (data/sessions/<id>/)
```

## Quick start (local)

Requirements: Python 3.10 to 3.12 and a webcam. Windows PowerShell shown; on Mac/Linux use
`source .venv/bin/activate`.

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                                   # logic tests, no camera needed
uvicorn app.main:app --host 127.0.0.1 --port 8000     # leave it running
```

**Candidate:** open http://localhost:8000 in Chrome or Edge, then
1. tick the consent box,
2. **Turn on camera**,
3. **Share your entire screen** (choose "Entire screen"),
4. **Start test in full screen**,
5. answer the questions (the screen is measured automatically in the first seconds),
6. **Submit**.

**Reviewer:** open http://localhost:8000/review.html and pick a session.

**Tuning view:** add `?debug=1` to the candidate URL to see live eye/head values, the detected zone and
the fitted screen under the small camera.

## Configuration

Every setting can be changed with an environment variable of the same name (no code change needed).

| Variable | Default | Meaning |
|---|---|---|
| `REVIEWER_TOKEN` | empty | Password for the reviewer page. **Empty = open, set it in production.** |
| `RETENTION_DAYS` | 0 | Delete sessions after N days (0 = keep) |
| `MAX_UPLOAD_MB` | 3000 | Upload limit per session (camera + screen) |
| `FACE_WARN_S` | 3 | Seconds before a face / camera issue warns |
| `AWAY_WARN_S` | 2 | Seconds eyes or head are away from the screen before a warning |
| `DOWN_WARN_S` | 2 | Seconds looking down without typing before a warning |
| `UP_WARN_S` | 12 | Seconds looking up before it is flagged |
| `ACTIVITY_WINDOW_S` | 3 | A key or mouse action this recent counts as "typing" |
| `AUTOFIT_S` | 6 | Seconds at the start used to learn the screen position |
| `EYE_MARGIN` | 0.10 | Extra eye-direction room around the screen area (raise it if it warns too often) |
| `EYE_MARGIN_PCT` | 0.20 | Same, as a share of the screen range (larger of the two is used) |
| `HEAD_TOL_DEG` | 10 | Extra head-turn room in degrees |
| `EYE_PER_DEG` | 0.012 | Eye-direction change per degree of eye rotation (screen-size model) |
| `DEFAULT_DISTANCE_M` | 0.55 | Assumed distance to the camera until it is measured |
| `CAM_HFOV_DEG` | 65 | Webcam horizontal field of view used for distance estimation |
| `HEAD_PITCH_SIGN` | 1 | Set to -1 if "looking down" is reported as "up" in `?debug=1` |
| `CLIP_PAD_BEFORE_S` / `CLIP_PAD_AFTER_S` | 3 / 3 | Seconds added around each evidence clip |
| `CLIP_MAX_S` | 120 | Maximum clip length |

## Deployment

The app needs a **long-running server with WebSocket support and HTTPS** (browsers only allow camera and
screen capture on HTTPS or localhost). It is deployed on Railway from the included `Dockerfile`; any
container host works. Serverless platforms such as Vercel are not suitable because they do not support
WebSockets or persistent disk.

Production checklist:
- set `REVIEWER_TOKEN`
- serve over HTTPS
- attach a persistent volume to `data/` if recordings must survive restarts
  (on most hosts the disk is wiped on every redeploy or restart)

## Where things are saved

`data/sessions/<id>/` contains `report.json`, `camera.webm`, `screen.webm` and `clips/*.mp4`.

## Project structure

```
app/
  main.py        FastAPI routes + WebSocket
  detector.py    MediaPipe Face Landmarker wrapper
  gaze.py        eye/head features, screen area estimation
  analysis.py    session monitor: timers, issues, warnings, report
  clips.py       evidence clip cutting (ffmpeg)
  storage.py     session files and upload handling
  config.py      all settings (environment variables)
frontend/
  index.html     candidate page (camera, screen share, test)
  review.html    reviewer page
models/          face_landmarker.task
tests/           logic tests (no camera needed)
```

## Limitations

- Gaze is estimated from a normal webcam, so it is most reliable for looking sideways or away.
  Looking up and down is less precise.
- Results depend on lighting and camera position (camera at eye level works best).
- This tool flags suspicious moments for a human reviewer. It does not decide on cheating by itself.
- Candidates must give consent before the camera and screen are recorded; keep recordings only as long
  as needed (`RETENTION_DAYS`).
