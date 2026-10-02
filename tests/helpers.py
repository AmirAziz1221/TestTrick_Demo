import numpy as np
from app.analysis import Settings, SessionMonitor


def make_face(cx=0.5, cy=0.5, w=0.3):
    """Synthetic 478-point face: only the landmarks our code reads are placed."""
    pts = np.tile([cx, cy, 0.0], (478, 1))
    pts[234] = [cx - w / 2, cy, 0]; pts[454] = [cx + w / 2, cy, 0]
    pts[10] = [cx, cy - 0.2, 0]; pts[152] = [cx, cy + 0.2, 0]
    return pts


def blend(h=0.0, v=0.0, blink=0.0):
    """h>0: look to the side. v>0: look down, v<0: look up."""
    b = {"eyeBlinkLeft": blink, "eyeBlinkRight": blink}
    b["eyeLookInLeft"] = b["eyeLookOutRight"] = max(h, 0)
    b["eyeLookOutLeft"] = b["eyeLookInRight"] = max(-h, 0)
    b["eyeLookDownLeft"] = b["eyeLookDownRight"] = max(v, 0)
    b["eyeLookUpLeft"] = b["eyeLookUpRight"] = max(-v, 0)
    return b


def run(m, seconds, start, faces=None, bl=None, step=0.7, bright=120):
    faces = [make_face()] if faces is None else faces
    t, out = start, None
    while t < start + seconds - 1e-9:
        out = m.update(t, faces, bl, bright)
        t += step
    return out, t


def calibrated_monitor(**kw):
    m = SessionMonitor(Settings(**kw))
    m.calibration_start()
    points = [(0, 0), (-.25, -.1), (.25, -.1), (.25, .2), (-.25, .2), (0, .2), (0, -.1)]
    t = 0.0
    for i, (h, v) in enumerate(points):
        m.calibration_point(i)
        _, t = run(m, 2.4, t, bl=blend(h, v), step=0.3)
    ok, msg, box = m.calibration_finish()
    assert ok, msg
    return m, t
