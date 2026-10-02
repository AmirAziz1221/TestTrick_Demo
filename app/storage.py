"""Disk layout: data/sessions/<id>/{camera.webm, screen.webm, report.json, clips/}."""
from __future__ import annotations
import json
from datetime import datetime, timezone
import re
import shutil
import time
from pathlib import Path
from . import config

SID_RE = re.compile(r"^[0-9a-f]{12}$")
KINDS = ("camera", "screen")


def valid_sid(sid: str) -> bool:
    return bool(SID_RE.match(sid or ""))


def session_dir(sid: str) -> Path:
    d = config.SESSIONS_DIR / sid
    d.mkdir(parents=True, exist_ok=True)
    return d


def append_chunk(sid: str, kind: str, data: bytes) -> int:
    with open(session_dir(sid) / f"{kind}.webm", "ab") as f:
        f.write(data)
    return len(data)


def size_of(sid: str, kind: str) -> int:
    p = config.SESSIONS_DIR / sid / f"{kind}.webm"
    return p.stat().st_size if p.exists() else 0


def save_report(sid: str, report: dict):
    p = session_dir(sid) / "report.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2))
    tmp.replace(p)


def load_report(sid: str) -> dict | None:
    p = config.SESSIONS_DIR / sid / "report.json"
    return json.loads(p.read_text()) if p.exists() else None


def list_reports() -> list[dict]:
    out = []
    for d in sorted(config.SESSIONS_DIR.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True):
        if d.is_dir() and valid_sid(d.name):
            r = load_report(d.name)
            if r:
                out.append({k: r.get(k) for k in ("session_id", "candidate_id", "test_id", "started_at",
                                                   "duration_s", "risk_score", "risk_level", "warning_count",
                                                   "clips_status", "full_status")})
    return out


def cleanup_old(days: int) -> int:
    """Delete session folders older than `days` (RETENTION_DAYS). Returns how many were removed."""
    if days <= 0:
        return 0
    cutoff, n = time.time() - days * 86400, 0
    for d in config.SESSIONS_DIR.glob("*"):
        if d.is_dir() and valid_sid(d.name) and d.stat().st_mtime < cutoff:
            shutil.rmtree(d, ignore_errors=True)
            n += 1
    return n


def log_access(sid: str, filename: str, who: str):
    """Audit trail: who opened a FULL recording of this session, and when (access.log in the session folder)."""
    with open(session_dir(sid) / "access.log", "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')}\t{who}\t{filename}\n")