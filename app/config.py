"""All settings. Every value can be overridden with an environment variable of the same name."""
import os
from pathlib import Path


def _f(name, default):
    return float(os.getenv(name, default))


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "face_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")
SESSIONS_DIR = ROOT / "data" / "sessions"
FRONTEND_DIR = ROOT / "frontend"

# ---- access / storage --------------------------------------------------------
REVIEWER_TOKEN = os.getenv("REVIEWER_TOKEN", "")     # empty = open (development only!)
RETENTION_DAYS = int(_f("RETENTION_DAYS", 0))        # 0 = never delete automatically
MAX_UPLOAD_MB = int(_f("MAX_UPLOAD_MB", 3000))       # per session, camera + screen together

# ---- warning timers: seconds an issue must last before a warning is raised -----
FACE_WARN_S = _f("FACE_WARN_S", 3)        # face absent / partly visible / second person / camera blocked
AWAY_WARN_S = _f("AWAY_WARN_S", 2)        # eyes or head turned away from the screen (requirement: 2 s)
DOWN_WARN_S = _f("DOWN_WARN_S", 2)        # looking down while NOT typing or using the mouse
UP_WARN_S = _f("UP_WARN_S", 12)           # looking up = "thinking"; only flagged when it lasts this long
ACTIVITY_WINDOW_S = _f("ACTIVITY_WINDOW_S", 3)  # a key/mouse action this recent means "typing"
CLEAR_FRAMES = int(_f("CLEAR_FRAMES", 2))       # on-screen frames needed to end an issue (ignores blips)

# ---- gaze thresholds (applied on top of each candidate's own calibration) -------
EYE_MARGIN = _f("EYE_MARGIN", 0.10)       # extra eye-direction room beyond the calibrated screen area
EYE_MARGIN_PCT = _f("EYE_MARGIN_PCT", 0.20)   # ...or this share of the calibrated range, whichever is larger
HEAD_TOL_DEG = _f("HEAD_TOL_DEG", 10)     # extra head-turn room (degrees) beyond the calibrated area
BLINK_T = _f("BLINK_T", 0.5)              # frames with eyes this closed are ignored
HEAD_PITCH_SIGN = _f("HEAD_PITCH_SIGN", 1)    # set to -1 if "looking down" is reported as "up" in ?debug=1
DARK_T = _f("DARK_T", 30)                 # mean brightness (0-255) below which a missing face = camera blocked

# ---- automatic screen fit (no calibration dots) ---------------------------------------------
AUTOFIT_S = _f("AUTOFIT_S", 6)                    # seconds at the start used to learn the candidate's screen position
DEFAULT_DISTANCE_M = _f("DEFAULT_DISTANCE_M", 0.55)   # assumed camera-to-face distance until it is measured
CAM_HFOV_DEG = _f("CAM_HFOV_DEG", 65)             # typical laptop/webcam horizontal field of view
EYE_PER_DEG = _f("EYE_PER_DEG", 0.012)            # eye-look blendshape change per degree of eye rotation (approximation)

# ---- evidence clips ---------------------------------------------------------------
CLIP_PAD_BEFORE_S = _f("CLIP_PAD_BEFORE_S", 3)
CLIP_PAD_AFTER_S = _f("CLIP_PAD_AFTER_S", 3)
CLIP_MAX_S = _f("CLIP_MAX_S", 120)


def settings():
    """Build the analysis Settings object from the values above."""
    from .analysis import Settings
    return Settings(face_warn_s=FACE_WARN_S, away_warn_s=AWAY_WARN_S, down_warn_s=DOWN_WARN_S,
                    up_warn_s=UP_WARN_S, activity_window_s=ACTIVITY_WINDOW_S, clear_frames=CLEAR_FRAMES,
                    eye_margin=EYE_MARGIN, eye_margin_pct=EYE_MARGIN_PCT, head_tol_deg=HEAD_TOL_DEG,
                    blink_t=BLINK_T, pitch_sign=HEAD_PITCH_SIGN, dark_t=DARK_T,
                    autofit_s=AUTOFIT_S, default_distance_m=DEFAULT_DISTANCE_M, cam_hfov_deg=CAM_HFOV_DEG,
                    eye_per_deg=EYE_PER_DEG)
