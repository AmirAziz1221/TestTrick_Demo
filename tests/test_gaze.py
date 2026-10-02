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
