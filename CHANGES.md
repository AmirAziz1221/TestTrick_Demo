# Changes: calibration dots removed, automatic screen fit added

## Behaviour
- No more 7-dot calibration. The test goes straight to question 1.
- The browser sends the screen size (width, height, pixel ratio); the server estimates the physical screen
  and the candidate's distance from the camera (from face width) and builds the "eyes on screen" area.
- First ~6 s (up to 12 s if frames are unclear): the screen position is learned, gaze is not judged,
  the small camera box shows "Fitting to your screen...". Off-screen frames are ignored by the fit.
- Eyes/head outside the screen area for 2 s -> red bar: "Your eyes are outside the screen area. Please look at the screen."
- Looking down while typing is still allowed; looking down without input is still flagged.

## Files changed
| File | Change |
|---|---|
| app/gaze.py | + screen_size_m(), estimate_distance_m(), estimate_box() |
| app/analysis.py | + set_screen(), auto-fit (_autofit_add/_autofit_finish), new warning text, up_warn_s default 12, status/summary include screen_fit |
| app/main.py | handles websocket message "screen_info" |
| app/config.py | + AUTOFIT_S, DEFAULT_DISTANCE_M, CAM_HFOV_DEG, EYE_PER_DEG (env-overridable) |
| frontend/index.html | removed calibration overlay/CSS/JS; sends screen_info; shows "Fitting to your screen..." |
| README.md | candidate steps updated |
| tests/test_gaze.py | + 3 tests for the screen fit |
| tests/helpers.py, tests/test_analysis.py | reset to the versions that match this code |

## Tuning (environment variables, no code change)
AWAY_WARN_S (2), AUTOFIT_S (6), EYE_PER_DEG (0.012), EYE_MARGIN (0.10), HEAD_TOL_DEG (10).
Open the test with ?debug=1 to see live eye/head values and the fitted screen.