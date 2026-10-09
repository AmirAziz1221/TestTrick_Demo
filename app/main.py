"""FastAPI server: candidate session + WebSocket monitoring, recording upload, reports, reviewer API."""
import asyncio, hmac, json, threading, time, uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from . import clips, config, storage
from .analysis import SessionMonitor
from .detector import FaceService

STATE: dict = {"service": None, "load_ms": None}
SESSIONS: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = time.perf_counter()
    STATE["service"] = FaceService()      # loaded once at startup, not per candidate or frame
    STATE["load_ms"] = round((time.perf_counter() - t) * 1000)
    config.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    removed = storage.cleanup_old(config.RETENTION_DAYS)
    print(f"MediaPipe model loaded in {STATE['load_ms']} ms | old sessions removed: {removed}")
    yield


app = FastAPI(title="TestTrick Proctor", lifespan=lifespan)


def reviewer(token: str | None = Query(None), x_token: str | None = Header(None)):
    """Reviewer endpoints are open only when REVIEWER_TOKEN is empty (development)."""
    if config.REVIEWER_TOKEN and not hmac.compare_digest(token or x_token or "", config.REVIEWER_TOKEN):
        raise HTTPException(401, "Reviewer token required")


class StartReq(BaseModel):
    candidate_id: str = "unknown"
    test_id: str = "unknown"
    consent: bool = False


@app.get("/api/health")
def health():
    return {"ok": STATE["service"] is not None, "model_load_ms": STATE["load_ms"],
            "active_sessions": len(SESSIONS), "ffmpeg": bool(clips.ffmpeg_exe())}


@app.post("/api/sessions")
def start_session(req: StartReq):
    if not req.consent:
        raise HTTPException(400, "Candidate consent is required")
    sid = uuid.uuid4().hex[:12]
    SESSIONS[sid] = {"meta": req.model_dump(), "monitor": SessionMonitor(config.settings()),
                     "t0": time.monotonic(), "started": datetime.now(timezone.utc),
                     "rec": {}, "bytes": 0}
    storage.session_dir(sid)
    return {"session_id": sid, "frame_interval_ms": 700}


async def _text_message(ws: WebSocket, mon: SessionMonitor, now: float, text: str):
    d = json.loads(text)
    t = d.get("t")
    if t == "act":
        mon.add_activity(now, d.get("k", "key"))
    elif t == "evt":
        mon.client_event(now, d.get("type", ""), d.get("state", "instant"))
    elif t == "screen_info":
        mon.set_screen(d)
    elif t == "calib_start":
        mon.calibration_start()
    elif t == "calib_point":
        mon.calibration_point(d.get("i", 0))
    elif t == "calib_end":
        ok, msg, box = mon.calibration_finish()
        await ws.send_json({"type": "calibration", "ok": ok, "message": msg, "box": box})


@app.websocket("/ws/{sid}")
async def ws_frames(ws: WebSocket, sid: str):
    s = SESSIONS.get(sid)
    if not s:
        await ws.close(code=4404)
        return
    await ws.accept()
    mon = s["monitor"]
    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break
            now = time.monotonic() - s["t0"]
            if msg.get("bytes"):
                t = time.perf_counter()
                fd = await asyncio.to_thread(STATE["service"].analyze, msg["bytes"])
                ms = round((time.perf_counter() - t) * 1000, 1)
                if fd is None:
                    await ws.send_json({"type": "error", "error": "bad_frame"})
                    continue
                out = mon.update(now, fd.faces, fd.blends[0] if fd.blends else None, fd.brightness)
                out["inference_ms"] = ms
                await ws.send_json(out)
            elif msg.get("text"):
                await _text_message(ws, mon, now, msg["text"])
    except WebSocketDisconnect:
        pass


@app.post("/api/sessions/{sid}/upload/{kind}")
async def upload(sid: str, kind: str, request: Request, seq: int = 0, x_rec_offset: float = Header(0)):
    s = SESSIONS.get(sid)
    if not s or kind not in storage.KINDS:
        raise HTTPException(404, "Unknown session or stream")
    rec = s["rec"].setdefault(kind, {"offset_s": max(x_rec_offset, 0) / 1000, "next": 0})
    if seq != rec["next"]:
        raise HTTPException(409, f"Expected chunk {rec['next']}")
    data = await request.body()
    if s["bytes"] + len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, "Upload limit reached")
    await asyncio.to_thread(storage.append_chunk, sid, kind, data)
    s["bytes"] += len(data)
    rec["next"] += 1
    return {"ok": True}


def _post_process(sid: str, report: dict):
    report = clips.prepare_recordings(sid, report)        # always: seekable full recordings
    if report["events"]:
        clips.generate_clips(sid, report)                 # only when there is something to review


@app.post("/api/sessions/{sid}/finish")
def finish(sid: str, reason: str = "submitted"):
    s = SESSIONS.pop(sid, None)
    if not s:
        raise HTTPException(404, "Unknown session")
    now = time.monotonic() - s["t0"]
    summary = s["monitor"].summary(now, s["started"])
    recordings = {k: {"file": f"{k}.webm", "offset_s": v["offset_s"], "bytes": storage.size_of(sid, k)}
                  for k, v in s["rec"].items()}
    report = {"session_id": sid, **{k: s["meta"][k] for k in ("candidate_id", "test_id", "consent")},
              "started_at": s["started"].isoformat(timespec="seconds"), "ended_by": reason,
              **summary, "recordings": recordings,
              "clips_status": "processing" if (summary["events"] and recordings) else "none",
              "full_status": "processing" if recordings else "none"}
    storage.save_report(sid, report)
    if recordings:
        threading.Thread(target=_post_process, args=(sid, report), daemon=True).start()
    return report


@app.get("/api/sessions", dependencies=[Depends(reviewer)])
def list_sessions():
    return storage.list_reports()


@app.get("/api/sessions/{sid}/report", dependencies=[Depends(reviewer)])
def get_report(sid: str):
    r = storage.load_report(sid) if storage.valid_sid(sid) else None
    if not r:
        raise HTTPException(404, "No report yet")
    return r


@app.get("/media/{sid}/{path:path}", dependencies=[Depends(reviewer)])
def media(sid: str, path: str, request: Request):
    root = (config.SESSIONS_DIR / sid).resolve()
    p = (root / path).resolve()
    if not storage.valid_sid(sid) or root not in p.parents or not p.is_file():
        raise HTTPException(404)
    rng = request.headers.get("range", "")
    if p.name.endswith(".webm") and (not rng or rng.startswith("bytes=0-")):   # log each full-recording view once
        storage.log_access(sid, p.name, request.client.host if request.client else "?")
    return FileResponse(p)


app.mount("/", StaticFiles(directory=config.FRONTEND_DIR, html=True), name="frontend")