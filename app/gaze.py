"""Gaze logic: features, per-candidate screen calibration, and zone classification.
# This module handles gaze detection, calibration, and classification of where a person is looking.

Pure numpy/python (no MediaPipe import), so it is unit-testable without a camera.
# Only Python/NumPy are used here, so this logic can be tested without connecting to a camera.

Idea: instead of fixed guesses, every candidate first looks at 7 dots (centre, four corners,
top/bottom centre). We record where their eyes and head are for each dot and build a
"screen box" = everything that counts as looking at the screen for THIS person, THIS camera
and THIS monitor. Anything outside the box (plus a small margin) is 'down', 'up' or 'side'.
# The system calibrates individually for each person/camera/monitor instead of using fixed gaze thresholds.
"""

from __future__ import annotations  # Python: allows type annotations to be evaluated later
import math  # Python standard library: used for angle calculations and degree conversion
import statistics  # Python standard library: used to calculate median values
from dataclasses import dataclass, asdict  # Dataclasses: creates data containers; asdict converts them to dictionaries
import numpy as np  # NumPy: used for facial landmark arrays and numerical calculations
# MediaPipe facial landmark indexes used by this module.
# These numbers refer to specific points in the MediaPipe face landmark model.
FOREHEAD, CHIN, CHEEK_L, CHEEK_R = 10, 152, 234, 454
@dataclass  # Dataclass: creates a simple structured container for extracted face features
class Features:
    h: float       # Horizontal eye direction: 0 = straight; larger value = looking more sideways
    v: float       # Vertical eye direction: positive = looking down, negative = looking up
    yaw: float     # Head rotation from left to right, measured in degrees
    pitch: float   # Head rotation/tilt vertically, measured in degrees; positive = head down
    blink: float   # Eye-closure score: 0 = open, 1 = closed
def head_angles(pts: np.ndarray, pitch_sign: float = 1.0) -> tuple[float, float]:
    # Calculates the approximate head yaw and pitch from the 3D face landmarks.
    """Head yaw/pitch (degrees) from the 3D landmarks. Absolute values are not exact,
    but they are consistent for one person, which is all calibration needs."""
    # The exact angles are not important; consistency is important for calibration.
    yaw = math.degrees(  # math: converts the calculated angle from radians to degrees
        math.atan2(  # math: calculates the angle using the vertical and horizontal differences
            pts[CHEEK_R, 2] - pts[CHEEK_L, 2],  # NumPy: difference between right and left cheek depth (Z)
            pts[CHEEK_R, 0] - pts[CHEEK_L, 0]   # NumPy: difference between right and left cheek horizontal position (X)
        )
    )
    pitch = math.degrees(  # math: converts the pitch angle from radians to degrees
        math.atan2(  # math: calculates the vertical head angle
            pts[CHIN, 2] - pts[FOREHEAD, 2],  # NumPy: difference in depth between chin and forehead
            pts[CHIN, 1] - pts[FOREHEAD, 1]   # NumPy: difference in vertical position between chin and forehead
        )
    )
    return yaw, pitch_sign * pitch  # Returns yaw and adjusted pitch angle
def features(
    pts: np.ndarray,
    blend: dict | None,
    pitch_sign: float = 1.0
) -> Features:
    # Extracts eye direction, head angles, and blink information.
    """Eye direction comes from MediaPipe's own eye-look blendshapes (same model call, no extra cost)."""
    # Eye direction is obtained from the blendshape values already produced by MediaPipe.
    b = blend or {}  # Python: uses the provided blendshape dictionary or an empty dictionary if unavailable
    g = lambda k: float(b.get(k, 0.0))
    # Python: helper function that gets a blendshape score and returns 0 if it does not exist
    h = (
        g("eyeLookInLeft")       # MediaPipe blendshape: left eye looking inward
        + g("eyeLookOutRight")  # MediaPipe blendshape: right eye looking outward
        - g("eyeLookOutLeft")   # MediaPipe blendshape: left eye looking outward
        - g("eyeLookInRight")   # MediaPipe blendshape: right eye looking inward
    ) / 2
    # Calculates the combined horizontal eye direction.
    v = (
        g("eyeLookDownLeft")    # MediaPipe blendshape: left eye looking down
        + g("eyeLookDownRight") # MediaPipe blendshape: right eye looking down
        - g("eyeLookUpLeft")    # MediaPipe blendshape: left eye looking up
        - g("eyeLookUpRight")   # MediaPipe blendshape: right eye looking up
    ) / 2
    # Calculates the combined vertical eye direction.
    yaw, pitch = head_angles(pts, pitch_sign)
    # Calculates head rotation using the 3D facial landmarks.
    return Features(
        h,  # Horizontal eye direction
        v,  # Vertical eye direction
        yaw,  # Head yaw
        pitch,  # Head pitch
        max(
            g("eyeBlinkLeft"),   # MediaPipe: left eye blink/closure score
            g("eyeBlinkRight")   # MediaPipe: right eye blink/closure score
        )
    )
    # Returns all calculated information as one Features object.
@dataclass  # Dataclass: stores the calibrated boundaries for screen-looking
class ScreenBox:
    """Everything inside this box counts as 'looking at the screen'. Defaults = wide fallback."""
    # Defines the range of eye/head movements considered to be looking at the screen.
    h_lo: float = -0.35  # Minimum horizontal eye direction allowed
    h_hi: float = 0.35   # Maximum horizontal eye direction allowed
    v_lo: float = -0.30  # Minimum vertical eye direction allowed
    v_hi: float = 0.35   # Maximum vertical eye direction allowed
    yaw_lo: float = -25.0  # Minimum head yaw allowed in degrees
    yaw_hi: float = 25.0   # Maximum head yaw allowed in degrees
    pitch_lo: float = -30.0  # Minimum head pitch allowed in degrees
    pitch_hi: float = 30.0   # Maximum head pitch allowed in degrees
    def asdict(self) -> dict:
        # Converts the ScreenBox object into a dictionary.
        return {
            k: round(v, 3)  # Python: rounds each boundary value to 3 decimal places
            for k, v in asdict(self).items()  # Dataclasses: converts ScreenBox into dictionary items
        }
class Calibrator:
    # Handles the personalized gaze calibration process.
    MIN_POINTS = 5
    # At least 5 of the 7 calibration points must produce clean data.
    def __init__(self, blink_t: float = 0.5):
        self.blink_t = blink_t
        # Stores the maximum blink value allowed during calibration.
        self.points: dict[int, list[Features]] = {}
        # Stores collected Features for each calibration point.
        # Example: point 1 → [Features(...), Features(...), ...]
    def add(self, point: int, f: Features):
        # Adds one frame's Features to the selected calibration point.
        if f.blink <= self.blink_t:
            # Only accept the frame if the person is not blinking.
            self.points.setdefault(point, []).append(f)
            # Python dictionary: creates a list for the point if needed and adds the Features object.
    def build(self, cfg) -> tuple[ScreenBox | None, str]:
        # Builds the personalized ScreenBox from the collected calibration data.
        meds = []
        # Stores median values for each successful calibration point.
        for _, s in sorted(self.points.items()):
            # Loops through calibration points in numerical order.
            s = (
                s[3:] if len(s) >= 6
                else (s[1:] if len(s) >= 3 else [])
            )
            # Removes early frames while the eyes are moving toward the calibration dot.
            # This helps use more stable gaze measurements.
            if len(s) >= 2:
                # Requires at least two stable frames for this calibration point.
                meds.append([
                    statistics.median(
                        getattr(x, a) for x in s
                    )
                    for a in ("h", "v", "yaw", "pitch")
                ])
                # statistics.median: gets the middle/stable value for each feature.
                # getattr: accesses h, v, yaw, and pitch from each Features object.
        if len(meds) < self.MIN_POINTS:
            # Checks whether at least 5 calibration points were successfully collected.
            return None, (
                f"Only {len(meds)} of 7 calibration points were clear. "
                "Keep your face in the frame, in good light, and follow the dot."
            )
            # If calibration failed, return no ScreenBox and an explanation message.
        cols = list(zip(*meds))
        # Python zip: separates the collected values into columns:
        # h values, v values, yaw values, and pitch values.
        def span(col, floor, pct):
            # Calculates the allowed lower and upper range for one feature.
            lo, hi = min(col), max(col)
            # Finds the minimum and maximum calibration values.
            m = max(floor, pct * (hi - lo))
            # Calculates a margin around the observed range.
            # Uses either the minimum margin or a percentage of the range.
            return lo - m, hi + m
            # Returns the expanded lower and upper limits.
        h = span(cols[0], cfg.eye_margin, cfg.eye_margin_pct)
        # Creates the horizontal eye-direction range.
        v = span(cols[1], cfg.eye_margin, cfg.eye_margin_pct)
        # Creates the vertical eye-direction range.
        yaw = span(cols[2], cfg.head_tol_deg, 0.2)
        # Creates the allowed head-yaw range with a 20% margin.
        pitch = span(cols[3], cfg.head_tol_deg, 0.2)
        # Creates the allowed head-pitch range with a 20% margin.
        return ScreenBox(
            h[0], h[1],
            v[0], v[1],
            yaw[0], yaw[1],
            pitch[0], pitch[1]
        ), "ok"
        # Creates the personalized ScreenBox and returns "ok" to indicate successful calibration.
def classify_zone(f: Features, box: ScreenBox) -> str | None:
    # Determines where the person is looking based on Features and the calibrated ScreenBox.
    """'screen' | 'down' | 'up' | 'side' | None (blink / unusable frame)."""
    # Possible results:
    # screen = looking at screen
    # down   = looking below screen
    # up     = looking above screen
    # side   = looking left/right
    # None   = blinking or unusable frame
    if f.blink > 0.5:
        # If the person is blinking, gaze direction is considered unreliable.
        return None
        # Ignore this frame.
    excess: dict[str, float] = {}
    # Stores how strongly the current gaze exceeds each screen boundary.
    def add(zone, amount, scale):
        # Adds a possible zone classification based on how far the feature is outside the screen box.
        if amount > 0:
            # Only record the zone when the value has exceeded its boundary.
            excess[zone] = max(
                excess.get(zone, 0.0),
                amount / scale
            )
            # Normalizes the amount and keeps the strongest value for that zone.
    add("down", f.v - box.v_hi, 0.15)
    # Checks whether the eyes are looking below the calibrated screen range.
    add("up", box.v_lo - f.v, 0.15)
    # Checks whether the eyes are looking above the calibrated screen range.
    add(
        "side",
        max(box.h_lo - f.h, f.h - box.h_hi),
        0.15
    )
    # Checks whether the eyes are looking too far left or right.
    add("down", f.pitch - box.pitch_hi, 10.0)
    # Checks whether the head pitch indicates the head is tilted downward.
    add("up", box.pitch_lo - f.pitch, 10.0)
    # Checks whether the head pitch indicates the head is tilted upward.
    add(
        "side",
        max(box.yaw_lo - f.yaw, f.yaw - box.yaw_hi),
        10.0
    )
    # Checks whether the head is turned too far left or right.
    return max(excess, key=excess.get) if excess else "screen"
    # If one or more boundaries were exceeded, return the strongest zone.
    # If no boundary was exceeded, classify the person as looking at the screen.
    