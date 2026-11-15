"""sMeetSec local web server (FastAPI). Runs only on 127.0.0.1 - nothing leaves the PC."""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from . import config, diarize, hardware, llm, meetings, report
from . import models as model_store
from .governor import governor, lower_process_priority
from .jobs import jobs
from .transcriber import load_audio, models, transcribe_file

log = logging.getLogger("smeetsec")
app = FastAPI(title="sMeetSec")


def apply_guard_settings() -> None:
    s = config.load_settings()
    governor.configure(s.get("cpu_limit_percent", 80), s.get("cpu_limit_action", "throttle"))


@app.on_event("startup")
def _startup() -> None:
    config.ensure_dirs()
    config.add_windows_cuda_dll_dirs()
    s = config.load_settings()
    if s.get("low_priority", True):
        lower_process_priority()
    apply_guard_settings()
    governor.start()
    hardware.detect()
    # meetings left in a running state by a crash/close
    for m in meetings.list_all():
        if m.get("status") in ("transcribing", "reporting", "live"):
            has_t = bool(meetings.current_transcript(m["id"]))
            meetings.update(m["id"], status="transcribed" if has_t else "error",
                            error="Interrupted (app was closed)")


@app.on_event("shutdown")
def _shutdown() -> None:
    llm.shutdown()


# ---------------------------------------------------------------- pages
@app.get("/", response_class=HTMLResponse)
def index():
    return (config.STATIC_DIR / "index.html").read_text(encoding="utf-8")


# ---------------------------------------------------------------- system
@app.get("/api/system")
def system():
    s = config.load_settings()
    hw = hardware.detect()
    return {
        "app": {"name": config.APP_NAME, "full_name": config.APP_FULL_NAME, "version": config.APP_VERSION,
                "mock": config.MOCK},
        "hardware": hw,
        "effective": hardware.effective(s),
        "settings": s,
        "llm": llm.status(),
        "guard": governor.status(),
        "gpu_fallback": models.last_fallback,
        "speakers": diarize.status(),
        "models_ready": model_store.all_present(),
    }


@app.get("/api/models")
def models_status():
    return {"items": model_store.status(), "free_gb": model_store.disk_free_gb(), "folder": str(config.MODELS_DIR),
            "jobs": [j for j in jobs.list() if j["kind"] == "models" and j["status"] in ("queued", "waiting", "running")],
            "last": next((j for j in jobs.list() if j["kind"] == "models"), None)}


@app.post("/api/models/download")
def models_download(body: dict):
    ids = body.get("ids") or [i["id"] for i in model_store.status() if not i["present"]]
    running = [j for j in jobs.list() if j["kind"] == "models" and j["status"] in ("queued", "waiting", "running")]
    if running:
        return running[0]
    return jobs.submit("models", "_models", lambda job: model_store.run_download_job(job, ids), "Download models").to_dict()


@app.get("/api/legal/{name}")
def legal(name: str):
    if name not in ("PRIVACY", "THIRD-PARTY-NOTICES"):
        raise HTTPException(404, "Not found")
    p = config.LEGAL_DIR / f"{name}.txt"
    return {"text": p.read_text(encoding="utf-8") if p.exists() else "(file missing)"}


@app.get("/api/guard")
def guard():
    return governor.status()


@app.post("/api/settings")
def set_settings(body: dict):
    s = config.save_settings(body)
    apply_guard_settings()
    return s


# ---------------------------------------------------------------- meetings
@app.get("/api/meetings")
def list_meetings():
    return meetings.list_all()


@app.get("/api/meetings/{mid}")
def get_meeting(mid: str):
    try:
        meta = meetings.get(mid)
    except (KeyError, ValueError):
        raise HTTPException(404, "Meeting not found")
    reps = {}
    for lang in meta.get("reports", []):
        r = meetings.read_json(mid, f"report_{lang}.json")
        if r:
            reps[lang] = r
    trans = {}
    for lang in meta.get("translations", []):
        p = meetings.path(mid, f"transcript_{lang}.txt")
        if p.exists():
            trans[lang] = p.read_text(encoding="utf-8").splitlines()
    return {"meta": meta, "transcript": meetings.current_transcript(mid), "reports": reps,
            "translations": trans, "files": meetings.files(mid), "jobs": jobs.active_for(mid)}


@app.delete("/api/meetings/{mid}")
def delete_meeting(mid: str):
    if jobs.active_for(mid):
        raise HTTPException(409, "A job is running for this meeting")
    meetings.delete(mid)
    return {"ok": True}


@app.patch("/api/meetings/{mid}")
def rename_meeting(mid: str, body: dict):
    return meetings.update(mid, title=str(body.get("title", ""))[:200] or "Meeting")


@app.put("/api/meetings/{mid}/transcript")
def save_transcript(mid: str, body: dict):
    """Save user edits (text corrections, speaker names)."""
    t = meetings.current_transcript(mid)
    if not t:
        raise HTTPException(404, "No transcript")
    segs = body.get("segments")
    if not isinstance(segs, list) or len(segs) != len(t["segments"]):
        raise HTTPException(400, "Segment list does not match")
    for old, new in zip(t["segments"], segs):
        old["text"] = str(new.get("text", old["text"]))
        old["speaker"] = str(new.get("speaker", old.get("speaker", "")))[:60]
    name = meetings.get(mid).get("transcript") or "transcript.json"
    meetings.write_json(mid, name, t)
    report.export_transcript(mid, t)
    return {"ok": True}


@app.get("/api/meetings/{mid}/files/{name}")
def download(mid: str, name: str):
    try:
        p = meetings.path(mid, name)
    except ValueError:
        raise HTTPException(400, "Bad name")
    if not p.exists():
        raise HTTPException(404, "File not found")
    return FileResponse(p, filename=name)


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), title: str = Form(""), language: str = Form("auto"),
                 auto_report: bool = Form(False), speakers: int = Form(0)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in meetings.AUDIO_EXT | meetings.VIDEO_EXT:
        raise HTTPException(400, f"Unsupported file type: {ext or 'unknown'}")
    meta = meetings.create(title, file.filename or "", "upload", language)
    dest = meetings.path(meta["id"], "source" + ext)
    with open(dest, "wb") as f:
        while chunk := await file.read(4 * 1024 * 1024):
            f.write(chunk)
    meetings.update(meta["id"], source=dest.name, num_speakers=speakers)
    job = _queue_transcribe(meta["id"], language, "best")
    if auto_report:
        _queue_report(meta["id"], config.load_settings()["report_languages"], True)
    return {"meeting": meetings.get(meta["id"]), "job": job.to_dict()}


@app.post("/api/meetings/{mid}/transcribe")
def retranscribe(mid: str, body: dict):
    meta = meetings.get(mid)
    if not meta.get("source"):
        raise HTTPException(400, "No recording for this meeting")
    if "speakers" in body:
        meetings.update(mid, num_speakers=int(body.get("speakers") or 0))
    return _queue_transcribe(mid, body.get("language") or meta.get("language", "auto"),
                             body.get("quality", "best")).to_dict()


@app.post("/api/meetings/{mid}/speakers")
def identify_speakers(mid: str, body: dict):
    """Run speaker identification on the existing transcript (keeps text edits)."""
    meta = meetings.get(mid)
    if not meta.get("source"):
        raise HTTPException(400, "No recording for this meeting")
    if not meetings.current_transcript(mid):
        raise HTTPException(400, "Transcribe first")
    if not diarize.available():
        raise HTTPException(400, "Speaker models are not installed. Run tools\\download_models.py")
    n = int(body.get("speakers") or 0)
    meetings.update(mid, num_speakers=n)

    def run(job):
        audio = load_audio(meetings.path(mid, meta["source"]))
        job.update(0.05, "Identifying speakers")
        turns = diarize.diarize(audio, n, progress=lambda f, m: job.update(0.05 + 0.9 * f, m), cancel=job.cancel)
        t = meetings.current_transcript(mid)
        for s in t["segments"]:  # fresh labels; keep names the user typed
            if s.get("speaker", "").startswith("Speaker "):
                s["speaker"] = ""
        t["speakers"] = diarize.assign(t["segments"], turns)
        name = meetings.get(mid).get("transcript") or "transcript.json"
        meetings.write_json(mid, name, t)
        report.export_transcript(mid, t)
    return jobs.submit("speakers", mid, run, f"Identify speakers: {meta['title']}").to_dict()


@app.post("/api/meetings/{mid}/rename_speaker")
def rename_speaker(mid: str, body: dict):
    old, new = str(body.get("old", "")), str(body.get("new", "")).strip()[:60]
    t = meetings.current_transcript(mid)
    if not t or not old:
        raise HTTPException(400, "Nothing to rename")
    for s in t["segments"]:
        if s.get("speaker") == old:
            s["speaker"] = new
    name = meetings.get(mid).get("transcript") or "transcript.json"
    meetings.write_json(mid, name, t)
    report.export_transcript(mid, t)
    return {"ok": True}


@app.post("/api/meetings/{mid}/report")
def make_report(mid: str, body: dict):
    langs = body.get("languages") or config.load_settings()["report_languages"]
    return _queue_report(mid, langs, bool(body.get("translate_transcript", True))).to_dict()


def _queue_transcribe(mid: str, language: str, quality: str):
    def run(job):
        meta = meetings.update(mid, status="transcribing", error="")
        src = meetings.path(mid, meta["source"])
        audio = load_audio(src)
        s = config.load_settings()
        n = int(meta.get("num_speakers") or 0)
        do_spk = s.get("speakers_auto", True) and n >= 0 and diarize.available()
        span = 0.85 if do_spk else 1.0
        t = transcribe_file(src, language, quality, progress=lambda f, m: job.update(f * span, m),
                            cancel=job.cancel, audio=audio)
        if do_spk and t["segments"]:
            turns = diarize.diarize(audio, n, cancel=job.cancel,
                                    progress=lambda f, m: job.update(0.85 + 0.14 * f, m))
            t["speakers"] = diarize.assign(t["segments"], turns)
        meetings.write_json(mid, "transcript.json", t)
        report.export_transcript(mid, t)
        meetings.update(mid, status="transcribed", transcript="transcript.json", duration=t["duration"],
                        detected_language=t["language"], language=language)
    title = meetings.get(mid)["title"]
    return jobs.submit("transcribe", mid, run, f"Transcribe: {title}")


def _queue_report(mid: str, languages: list[str], translate_transcript: bool):
    def run(job):
        report.run_report_job(job, mid, languages, translate_transcript)
    title = meetings.get(mid)["title"]
    return jobs.submit("report", mid, run, f"Translate & report: {title}")


# ---------------------------------------------------------------- jobs
@app.get("/api/jobs")
def list_jobs():
    return jobs.list()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    return {"ok": jobs.cancel(job_id)}


# ---------------------------------------------------------------- live
_live = {"session": None}


@app.websocket("/ws/live")
async def live_ws(ws: WebSocket):
    from .live import LiveSession

    await ws.accept()
    loop = asyncio.get_running_loop()
    outq: asyncio.Queue = asyncio.Queue()

    def emit(msg: dict):
        loop.call_soon_threadsafe(outq.put_nowait, msg)

    async def sender():
        while True:
            msg = await outq.get()
            if msg is None:
                break
            await ws.send_text(json.dumps(msg, ensure_ascii=False))

    sess = None
    send_task = asyncio.create_task(sender())
    try:
        first = json.loads(await ws.receive_text())
        eff = hardware.effective(config.load_settings())
        if not eff["live_allowed"]:
            emit({"type": "error", "msg": "Live mode is not supported on this hardware (no NVIDIA GPU). "
                                          "Use batch mode, or enable experimental live mode in Settings."})
            emit(None)
            await send_task
            await ws.close()
            return
        if _live["session"] is not None:
            emit({"type": "error", "msg": "A live session is already running."})
            emit(None)
            await send_task
            await ws.close()
            return
        sess = await asyncio.to_thread(LiveSession, first.get("title", ""), first.get("language", "auto"), emit)
        _live["session"] = sess
        emit({"type": "started", "meeting_id": sess.mid})

        async def cpu_ticker():
            while True:
                await asyncio.sleep(1)
                emit({"type": "cpu", "value": round(governor.usage(), 1), "limit": governor.limit,
                      "seconds": round(sess.total_samples / 16000, 1)})
        ticker = asyncio.create_task(cpu_ticker())
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if msg.get("bytes"):
                    sess.feed(msg["bytes"])
                elif msg.get("text"):
                    cmd = json.loads(msg["text"])
                    if cmd.get("cmd") == "stop":
                        break
        finally:
            ticker.cancel()
        meta = await asyncio.to_thread(sess.stop)
        emit({"type": "stopped", "meeting": meta})
    except WebSocketDisconnect:
        pass
    finally:
        if sess is not None:
            await asyncio.to_thread(sess.stop)
            _live["session"] = None
        emit(None)
        try:
            await asyncio.wait_for(send_task, 5)
        except Exception:
            pass
        try:
            await ws.close()
        except Exception:
            pass
