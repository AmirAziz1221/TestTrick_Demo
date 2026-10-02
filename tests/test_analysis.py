import pytest
from app.analysis import Settings, SessionMonitor, classify_face
from .helpers import blend, calibrated_monitor, make_face, run


def test_classify_face():
    assert classify_face([]) == "absent"
    assert classify_face([make_face(), make_face()]) == "multiple"
    assert classify_face([make_face(cx=0.05)]) == "partial"
    assert classify_face([0.5 + (make_face() - 0.5) * 0.3]) == "partial"
    assert classify_face([make_face()]) == "full"


def test_absent_face_waits_for_debounce_then_logs_event():
    m, t = calibrated_monitor()
    out, t = run(m, 2, t, faces=[])
    assert not out["warning"]
    out, t = run(m, 3, t, faces=[])
    assert out["warning"] and out["issue"] == "absent"
    out, t = run(m, 3, t, bl=blend())
    assert not out["warning"]
    s = m.summary(t)
    ev = s["events"][0]
    assert s["warning_count"] == 1 and ev["type"] == "absent" and ev["duration"] > 3


def test_dark_camera_is_reported_as_blocked():
    m, t = calibrated_monitor()
    out, t = run(m, 5, t, faces=[], bright=5)
    assert out["issue"] == "camera_blocked" and out["warning"]


def test_looking_away_warns_after_two_seconds():
    m, t = calibrated_monitor()
    out, t = run(m, 1.4, t, bl=blend(h=0.8))
    assert not out["warning"]
    out, t = run(m, 2.0, t, bl=blend(h=0.8))
    assert out["warning"] and out["issue"] == "looking_away"
    assert m.summary(t)["events"][0]["start"] < t - 2


def test_short_glance_is_ignored():
    m, t = calibrated_monitor()
    _, t = run(m, 1.4, t, bl=blend(h=0.8))
    out, t = run(m, 7, t, bl=blend())
    assert m.summary(t)["warning_count"] == 0


def test_blip_of_one_frame_does_not_reset_the_timer():
    m, t = calibrated_monitor()
    _, t = run(m, 1.4, t, bl=blend(h=0.8))
    _, t = run(m, 0.7, t, bl=blend())            # one on-screen frame
    out, t = run(m, 1.4, t, bl=blend(h=0.8))
    assert out["warning"]


def test_looking_down_while_typing_is_allowed():
    m, t = calibrated_monitor()
    for _ in range(10):                          # keys pressed throughout
        m.add_activity(t, "key")
        out, t = run(m, 0.7, t, bl=blend(v=0.7))
    assert not out["warning"] and out["typing"]
    s = m.summary(t)
    assert s["warning_count"] == 0 and s["typing_seconds"] > 0 and s["input_activity"]["key"] == 10


def test_looking_down_without_input_is_flagged():
    m, t = calibrated_monitor()
    m.add_activity(t, "key")
    out, t = run(m, 8, t, bl=blend(v=0.7))
    assert out["warning"] and out["issue"] == "looking_down_no_input"
    ev = m.summary(t)["events"][0]
    assert ev["severity"] == "high" and ev["type"] == "looking_down_no_input"


def test_mouse_movement_also_counts_as_activity():
    m, t = calibrated_monitor()
    for _ in range(6):
        m.add_activity(t, "mouse")
        out, t = run(m, 0.7, t, bl=blend(v=0.7))
    assert not out["warning"]


def test_looking_up_is_thinking_not_flagged_until_long():
    m, t = calibrated_monitor()
    out, t = run(m, 8, t, bl=blend(v=-0.7))
    assert not out["warning"]
    out, t = run(m, 6, t, bl=blend(v=-0.7))
    assert out["warning"] and out["issue"] == "looking_up_long"
    assert m.summary(t)["thinking_seconds"] > 10


def test_blinks_do_not_create_issues():
    m, t = calibrated_monitor()
    out, t = run(m, 6, t, bl=blend(h=0.8, blink=0.9))   # unusable frames keep state (none)
    assert not out["warning"]


def test_client_events_and_risk_score():
    m, t = calibrated_monitor()
    m.client_event(t, "fullscreen_exit", "start")
    m.client_event(t + 5, "fullscreen_exit", "end")
    m.client_event(t + 6, "paste")
    m.client_event(t + 7, "not_a_real_event")
    s = m.summary(t + 10)
    types = [e["type"] for e in s["events"]]
    assert types == ["fullscreen_exit", "paste"] and s["events"][0]["duration"] == 5
    assert s["risk_score"] > 0 and s["risk_level"] in ("low", "medium", "high")


def test_iso_timestamps_added():
    from datetime import datetime, timezone
    m, t = calibrated_monitor()
    m.client_event(t, "paste")
    s = m.summary(t + 1, datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc))
    assert s["events"][0]["start_iso"].startswith("2026-10-01T09:")
