"""Session monitor: turns per-frame data into debounced warnings and a timestamped event log.

Pure logic (no MediaPipe, no camera), so it is fully unit-tested.

Decision rules (all thresholds live in config.py):
  * face absent / partly visible / second person / camera blocked -> warning after FACE_WARN_S
  * eyes or head turned sideways                                  -> warning after AWAY_WARN_S (2 s)
  * looking DOWN + key/mouse activity in the last ACTIVITY_WINDOW_S -> typing, NOT cheating
  * looking DOWN + no key/mouse activity                          -> warning after DOWN_WARN_S
  * looking UP (thinking)                                         -> allowed, flagged only after UP_WARN_S
Times are seconds since the session started, so they line up with the recordings.
"""
from __future__ import annotations
from dataclasses import dataclass
from datetime import timedelta
import statistics
import numpy as np
from .gaze import Calibrator, ScreenBox, classify_zone, estimate_box, estimate_distance_m, features


@dataclass
class Settings:
    face_warn_s: float = 3.0
    away_warn_s: float = 2.0
    down_warn_s: float = 2.0
    up_warn_s: float = 12.0
    activity_window_s: float = 3.0
    clear_frames: int = 2
    eye_margin: float = 0.10
    eye_margin_pct: float = 0.20
    head_tol_deg: float = 10.0
    blink_t: float = 0.5
    pitch_sign: float = 1.0
    dark_t: float = 30.0
    autofit_s: float = 6.0
    default_distance_m: float = 0.55
    cam_hfov_deg: float = 65.0
    eye_per_deg: float = 0.012


# issue -> (message shown to the candidate, severity, category, Settings field with its timer)
ISSUES = {
    "absent": ("We can't see your face. Please stay in front of the camera.", "medium", "face", "face_warn_s"),
    "partial": ("Your face is only partly visible. Please centre it in the frame.", "medium", "face", "face_warn_s"),
    "multiple": ("More than one person is visible. Only the candidate may be on camera.", "high", "face", "face_warn_s"),
    "camera_blocked": ("Your camera looks blocked or too dark. Please fix it.", "high", "face", "face_warn_s"),
    "looking_away": ("Your eyes are outside the screen area. Please look at the screen.", "medium", "gaze", "away_warn_s"),
    "looking_down_no_input": ("Please look at the screen. Looking down without typing is flagged.", "high", "gaze", "down_warn_s"),
    "looking_up_long": ("Please keep your eyes on the screen.", "low", "gaze", "up_warn_s"),
}
# events reported by the browser: type -> (description, severity)
CLIENT_EVENTS = {
    "fullscreen_exit": ("Left full-screen mode", "medium"),
    "tab_hidden": ("Switched away from the test window", "medium"),
    "screen_share_stopped": ("Screen sharing was stopped", "high"),
    "multiple_monitors": ("More than one monitor detected", "high"),
    "paste": ("Paste attempted", "low"),
    "copy": ("Copy attempted", "low"),
    "right_click": ("Right-click used", "low"),
}
WEIGHT = {"low": 2, "medium": 5, "high": 10}


def classify_face(faces, margin=0.02, min_width=0.12) -> str:
    """absent | multiple | partial | full"""
    if not faces:
        return "absent"
    if len(faces) > 1:
        return "multiple"
    xs, ys = faces[0][:, 0], faces[0][:, 1]
    if xs.min() < margin or xs.max() > 1 - margin or ys.min() < margin or ys.max() > 1 - margin:
        return "partial"
    if xs.max() - xs.min() < min_width:
        return "partial"
    return "full"


class SessionMonitor:
    def __init__(self, cfg: Settings | None = None):
        self.cfg = cfg or Settings()
        self.box = ScreenBox()            # wide fallback until calibration succeeds
        self.calibrated = False
        self.calibrating = False
        self.calib: Calibrator | None = None
        self.calib_point = 0
        self.issue = None
        self.issue_since = None
        self.warning_active = False
        self.events: list[dict] = []
        self.frames = 0
        self.last_activity = -1e9
        self.activity = {"key": 0, "mouse": 0}
        self.typing_s = 0.0
        self.thinking_s = 0.0
        self._eid = 0
        self._issue_ev = None
        self._open_client: dict[str, dict] = {}
        self._clear = 0
        self._last_issue_t = 0.0
        self._last_t = None
        self._typing = False
        self.screen_info: dict | None = None
        self.screen_fit: dict | None = None
        self.autofit = False
        self._fit_t0 = None
        self._fit_samples: list[tuple] = []
        self._fit_dist: list[float] = []

    # ---------- automatic screen fit (replaces calibration dots) ----------
    def set_screen(self, info: dict):
        """Browser reports its screen size; build the screen area from it, refine it over the first seconds."""
        def num(k, lo, hi, default):
            try:
                return min(max(float(info.get(k, default)), lo), hi)
            except (TypeError, ValueError):
                return default
        self.screen_info = {"w": num("w", 320, 8192, 1920), "h": num("h", 240, 8192, 1080), "dpr": num("dpr", 0.5, 4, 1)}
        self.box = estimate_box(self.screen_info, self.cfg.default_distance_m, eye_per_deg=self.cfg.eye_per_deg,
                                eye_margin=self.cfg.eye_margin, head_tol=self.cfg.head_tol_deg)
        self.screen_fit = {**self.screen_info, "status": "estimated"}
        self.autofit, self._fit_t0, self._fit_samples, self._fit_dist = True, None, [], []

    def _autofit_add(self, now: float, face, f):
        if self._fit_t0 is None:
            self._fit_t0 = now
        # only frames that look like 'on screen' under the size-based estimate teach the fit (ignores glances away)
        if f.blink <= self.cfg.blink_t and classify_zone(f, self.box) == "screen":
            self._fit_samples.append((f.h, f.v, f.yaw, f.pitch))
            self._fit_dist.append(estimate_distance_m(float(face[:, 0].max() - face[:, 0].min()), self.cfg.cam_hfov_deg))
        elapsed = now - self._fit_t0
        if elapsed >= 2 * self.cfg.autofit_s or (elapsed >= self.cfg.autofit_s and len(self._fit_samples) >= 5):
            self._autofit_finish()            # enough clear frames, or give up waiting after twice the learning time

    def _autofit_finish(self):
        self.autofit = False
        if len(self._fit_samples) < 5:                 # not enough clear frames: keep the size-based estimate
            return
        center = tuple(statistics.median(c) for c in zip(*self._fit_samples))
        dist = statistics.median(self._fit_dist)
        self.box = estimate_box(self.screen_info, dist, center, eye_per_deg=self.cfg.eye_per_deg,
                                eye_margin=self.cfg.eye_margin, head_tol=self.cfg.head_tol_deg)
        self.screen_fit.update(status="fitted", distance_m=round(dist, 2), center=[round(c, 3) for c in center])
        self._fit_samples, self._fit_dist = [], []

    # ---------- calibration ----------
    def calibration_start(self):
        self.calibrating, self.calib, self.calib_point = True, Calibrator(self.cfg.blink_t), 0
        self.issue, self.issue_since, self.warning_active = None, None, False

    def calibration_point(self, i: int):
        self.calib_point = int(i)

    def calibration_finish(self):
        box, msg = self.calib.build(self.cfg) if self.calib else (None, "Calibration was not started.")
        self.calibrating = False
        if box:
            self.box, self.calibrated = box, True
        return box is not None, msg, (box.asdict() if box else None)

    # ---------- inputs from the browser ----------
    def add_activity(self, now: float, kind: str):
        kind = "key" if kind == "key" else "mouse"
        self.last_activity = now
        self.activity[kind] += 1

    def client_event(self, now: float, etype: str, state: str = "instant"):
        if etype not in CLIENT_EVENTS:
            return
        msg, sev = CLIENT_EVENTS[etype]
        if state == "start" and etype not in self._open_client:
            self._open_client[etype] = self._new_event(etype, now, sev, "client", msg)
        elif state == "end" and etype in self._open_client:
            self._close(self._open_client.pop(etype), now)
        elif state == "instant":
            self._close(self._new_event(etype, now, sev, "client", msg), now)

    # ---------- per-frame update ----------
    def update(self, now: float, faces, blend: dict | None = None, brightness: float | None = None) -> dict:
        self.frames += 1
        dt = min(max(now - self._last_t, 0.0), 2.0) if self._last_t is not None else 0.0
        self._last_t = now
        state = classify_face(faces)
        if state == "absent" and brightness is not None and brightness < self.cfg.dark_t:
            state = "camera_blocked"
        feats = zone = None
        if state == "full":
            feats = features(faces[0], blend, self.cfg.pitch_sign)
            if self.calibrating:
                self.calib.add(self.calib_point, feats)
            else:
                if self.autofit:
                    self._autofit_add(now, faces[0], feats)
                # while the screen position is being learned (first seconds) gaze is not judged
                zone = "screen" if self.autofit else classify_zone(feats, self.box)
        if self.calibrating:
            return self._status(state, None, feats)

        raw = self._raw_issue(state, zone, now, dt)
        if raw == self.issue:
            self._clear = 0
            if raw:
                self._last_issue_t = now
        elif raw is None and self.issue is not None and self._clear + 1 < self.cfg.clear_frames:
            self._clear += 1                      # a single on-screen frame does not end an issue
        else:
            self._end_issue(self._last_issue_t if self.issue else now)
            self.issue, self.issue_since = raw, (now if raw else None)
            self._last_issue_t, self._clear = now, 0

        if self.issue and not self.warning_active:
            msg, sev, cat, attr = ISSUES[self.issue]
            if now - self.issue_since >= getattr(self.cfg, attr):
                self.warning_active = True
                self._issue_ev = self._new_event(self.issue, self.issue_since, sev, cat, msg)
        return self._status(state, zone, feats)

    def _raw_issue(self, state, zone, now, dt):
        self._typing = False
        if state != "full":
            return state
        if zone is None:                          # blink / unusable frame: keep the current state
            return self.issue
        if zone == "screen":
            return None
        if zone == "side":
            return "looking_away"
        if zone == "down":
            if now - self.last_activity <= self.cfg.activity_window_s:
                self._typing = True
                self.typing_s += dt
                return None                       # typing or using the mouse: allowed
            return "looking_down_no_input"
        self.thinking_s += dt                     # zone == "up"
        return "looking_up_long"

    def _status(self, state, zone, feats) -> dict:
        out = {"type": "status", "face_state": state, "zone": zone, "issue": self.issue,
               "warning": self.warning_active, "typing": self._typing,
               "message": ISSUES[self.issue][0] if self.warning_active else None,
               "calibrating": self.calibrating, "calibrated": self.calibrated,
               "fitting": self.autofit, "screen_fit": self.screen_fit}
        if feats:
            out["debug"] = {k: round(v, 3) for k, v in vars(feats).items()}
        return out

    # ---------- events ----------
    def _new_event(self, etype, start, severity, category, message):
        self._eid += 1
        ev = {"id": self._eid, "type": etype, "category": category, "severity": severity,
              "message": message, "start": round(start, 2), "end": None, "duration": None}
        self.events.append(ev)
        return ev

    @staticmethod
    def _close(ev, end):
        ev["end"] = round(max(end, ev["start"]), 2)
        ev["duration"] = round(ev["end"] - ev["start"], 2)

    def _end_issue(self, end_t):
        if self._issue_ev is not None:
            self._close(self._issue_ev, end_t)
        self._issue_ev, self.warning_active = None, False

    def summary(self, now: float, started_at=None) -> dict:
        self._end_issue(now)
        self.issue = None
        for ev in list(self._open_client.values()):
            self._close(ev, now)
        self._open_client.clear()
        events = sorted(self.events, key=lambda e: e["start"])
        by_type: dict[str, float] = {}
        count: dict[str, int] = {}
        score = 0.0
        for e in events:
            by_type[e["type"]] = round(by_type.get(e["type"], 0) + e["duration"], 2)
            count[e["type"]] = count.get(e["type"], 0) + 1
            score += WEIGHT[e["severity"]] * (1 + min(e["duration"], 60) / 60)
            if started_at is not None:
                e["start_iso"] = (started_at + timedelta(seconds=e["start"])).isoformat(timespec="milliseconds")
                e["end_iso"] = (started_at + timedelta(seconds=e["end"])).isoformat(timespec="milliseconds")
        score = min(100, round(score))
        return {"duration_s": round(now, 1), "frames_analyzed": self.frames, "calibrated": self.calibrated,
                "screen_box": self.box.asdict() if (self.calibrated or self.screen_fit) else None,
                "screen_fit": self.screen_fit,
                "input_activity": self.activity, "typing_seconds": round(self.typing_s, 1),
                "thinking_seconds": round(self.thinking_s, 1),
                "warning_count": len(events), "count_by_type": count, "seconds_by_type": by_type,
                "risk_score": score, "risk_level": "high" if score >= 40 else "medium" if score >= 15 else "low",
                "events": events, "review_recommended": bool(events),
                "note": "Signals for human review only, not an automatic verdict."}