from app import config, storage


def test_access_log_and_listing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SESSIONS_DIR", tmp_path)
    sid = "abcdef123456"
    storage.save_report(sid, {"session_id": sid, "candidate_id": "c1", "risk_level": "low", "full_status": "done"})
    storage.log_access(sid, "camera_full.webm", "10.0.0.5")
    line = (tmp_path / sid / "access.log").read_text().strip().split("\t")
    assert line[1:] == ["10.0.0.5", "camera_full.webm"]
    assert storage.list_reports()[0]["full_status"] == "done"
    assert not storage.valid_sid("../etc")