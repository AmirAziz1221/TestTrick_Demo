from app.analysis import Settings
from app.gaze import Calibrator, Features, ScreenBox, classify_zone
from .helpers import calibrated_monitor


def f(h=0, v=0, yaw=0, pitch=0, blink=0):
    return Features(h, v, yaw, pitch, blink)


def test_default_box_screen_and_directions():
    b = ScreenBox()
    assert classify_zone(f(), b) == "screen"
    assert classify_zone(f(v=0.7), b) == "down"
    assert classify_zone(f(v=-0.7), b) == "up"
    assert classify_zone(f(h=0.8), b) == "side"
    assert classify_zone(f(yaw=50), b) == "side"
    assert classify_zone(f(pitch=45), b) == "down"
    assert classify_zone(f(blink=0.9), b) is None


def test_calibration_builds_box_from_corners():
    m, _ = calibrated_monitor()
    b = m.box
    assert b.h_lo < -0.25 and b.h_hi > 0.25          # corners are inside the box
    assert b.v_hi > 0.2 and b.v_lo < -0.1
    assert classify_zone(f(h=0.2, v=0.15), b) == "screen"   # still reading the screen's corner
    assert classify_zone(f(v=0.6), b) == "down"               # keyboard-level
    assert classify_zone(f(h=0.7), b) == "side"


def test_calibration_rejects_poor_data():
    c = Calibrator()
    for _ in range(10):
        c.add(0, f())
    box, msg = c.build(Settings())
    assert box is None and "calibration points" in msg


# ---- automatic screen fit (replaces calibration dots) ----
from app.analysis import SessionMonitor
from app.gaze import estimate_box
from .helpers import blend, run

SCREEN = {"w": 1920, "h": 1080, "dpr": 1}


def test_box_grows_with_screen_size_and_shrinks_with_distance():
    big = estimate_box({"w": 2560, "h": 1440, "dpr": 1}, 0.55)
    small = estimate_box({"w": 1366, "h": 768, "dpr": 1}, 0.55)
    far = estimate_box({"w": 2560, "h": 1440, "dpr": 1}, 0.9)
    assert big.h_hi > small.h_hi and big.v_hi > small.v_hi
    assert far.h_hi < big.h_hi
    assert classify_zone(f(), big) == "screen"
    assert classify_zone(f(v=0.6), big) == "down"
    assert classify_zone(f(h=0.9), big) == "side"


def test_auto_fit_learns_screen_without_dots_then_warns_when_eyes_leave():
    m = SessionMonitor(Settings())
    m.set_screen(SCREEN)
    out, t = run(m, 3, 0.0, bl=blend(h=0.9))              # eyes off-screen while fitting: not judged yet
    assert out["fitting"] and not out["warning"]
    out, t = run(m, 6, t, bl=blend(h=0.05, v=0.02))       # reading the screen
    assert not out["fitting"] and m.screen_fit["status"] == "fitted"
    assert not out["warning"]
    out, t = run(m, 3, t, bl=blend(h=0.9))                # eyes go out of the screen
    assert out["warning"] and out["issue"] == "looking_away"
    assert "outside the screen" in out["message"]


def test_auto_fit_flags_looking_down_without_typing():
    m = SessionMonitor(Settings())
    m.set_screen(SCREEN)
    _, t = run(m, 7, 0.0, bl=blend(h=0.0, v=0.0))
    out, t = run(m, 3, t, bl=blend(v=0.7))
    assert out["warning"] and out["issue"] == "looking_down_no_input"
