"""Cut a short evidence clip (camera + screen) for every warning event, using ffmpeg."""
from __future__ import annotations
import shutil
import subprocess
from pathlib import Path
from . import config, storage


def ffmpeg_exe() -> str | None:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


def _run(cmd: list[str]) -> bool:
    try:
        return subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600).returncode == 0
    except Exception:
        return False


def prepare_recordings(sid: str, report: dict) -> dict:
    """Make both recordings seekable (browser files have no duration/index) so reviewers can scrub the
    FULL videos. The re-indexed copy <kind>_full.webm replaces the original (fast stream copy, no re-encode)."""
    ff = ffmpeg_exe()
    root = config.SESSIONS_DIR / sid
    for kind, rec in (report.get("recordings") or {}).items():
        src = root / f"{kind}.webm"
        if ff and src.exists() and src.stat().st_size > 0:
            dst = root / f"{kind}_full.webm"
            if _run([ff, "-y", "-i", str(src), "-c", "copy", str(dst)]) and dst.exists() and dst.stat().st_size > 0:
                src.unlink(missing_ok=True)
                rec["file"], rec["bytes"] = dst.name, dst.stat().st_size
    report["full_status"] = "done"
    storage.save_report(sid, report)
    return report


def make_clip(ff: str, src: Path, dst: Path, start_s: float, dur_s: float) -> bool:
    cmd = [ff, "-y", "-ss", f"{max(start_s, 0):.2f}", "-i", str(src), "-t", f"{max(dur_s, 1):.2f}",
           "-an", "-vf", "scale='min(1280,iw)':-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dst)]
    return _run(cmd) and dst.exists() and dst.stat().st_size > 0


def generate_clips(sid: str, report: dict) -> dict:
    """Adds ev['clips'] = {'camera': 'clips/...', 'screen': 'clips/...'} to every event and saves the report."""
    ff = ffmpeg_exe()
    root = config.SESSIONS_DIR / sid
    (root / "clips").mkdir(exist_ok=True)
    sources = {}
    for kind in storage.KINDS:
        rec = (report.get("recordings") or {}).get(kind)
        src = root / (rec or {}).get("file", f"{kind}.webm")
        if ff and rec and src.exists() and src.stat().st_size > 0:
            sources[kind] = (src, float(rec.get("offset_s", 0)))
    for ev in report["events"]:
        start = ev["start"] - config.CLIP_PAD_BEFORE_S
        length = min((ev["end"] or ev["start"]) - ev["start"] + config.CLIP_PAD_BEFORE_S + config.CLIP_PAD_AFTER_S,
                     config.CLIP_MAX_S)
        for kind, (src, offset) in sources.items():
            in_file = start - offset                       # position inside the recording
            dur = length + min(in_file, 0)                 # shorten if the event began before recording did
            name = f"event_{ev['id']:03d}_{kind}.mp4"
            if make_clip(ff, src, root / "clips" / name, in_file, dur):
                ev.setdefault("clips", {})[kind] = f"clips/{name}"
        storage.save_report(sid, report)                   # reviewers can already see finished clips
    report["clips_status"] = "done" if ff else "ffmpeg_missing"
    storage.save_report(sid, report)
    return report