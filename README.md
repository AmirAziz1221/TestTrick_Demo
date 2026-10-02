# TestTrick Proctor v2: gaze, typing rule, recordings and evidence clips

MediaPipe Face Landmarker only (no YOLO). Full documentation: `docs/DOCUMENTATION.md` (also as `.docx`).

## Run (Windows PowerShell; Mac/Linux use `source .venv/bin/activate`)
1. Python 3.10 to 3.12.
2. `python -m venv .venv` then `.venv\Scripts\activate`
3. `pip install -r requirements.txt`
4. `python -m pytest -q`  (16 logic tests, no camera needed)
5. `uvicorn app.main:app --host 0.0.0.0 --port 8000`  (leave it running)
6. Candidate: open **http://localhost:8000** in Chrome or Edge, then
   1. tick the consent box, 2. **Turn on camera**, 3. **Share your entire screen** (choose "Entire screen"),
   4. **Start test in full screen**, 5. follow the yellow dot (calibration, ~20 s), 6. answer, 7. **Submit**.
7. Reviewer: open **http://localhost:8000/review.html** and pick a session. Only the flagged moments are shown as short clips.

Tuning view: add `?debug=1` to the candidate URL to see live zone and values under the small camera.
Production: set `REVIEWER_TOKEN`, serve over HTTPS, use `--workers N` (see docs sections 12 and 13).

Where things are saved: `data/sessions/<id>/` -> `report.json`, `camera.webm`, `screen.webm`, `clips/*.mp4`.
