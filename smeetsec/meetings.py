"""Meeting storage: one folder per meeting under data/meetings/<id>/."""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from pathlib import Path

from . import config

_lock = threading.Lock()

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma", ".webm"}
VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".wmv", ".m4v", ".mpg", ".mpeg", ".flv", ".ts"}


def _dir(mid: str) -> Path:
    if not re.fullmatch(r"[0-9a-zA-Z_-]+", mid or ""):
        raise ValueError("bad meeting id")
    return config.MEETINGS_DIR / mid


def path(mid: str, name: str = "") -> Path:
    d = _dir(mid)
    if not name:
        return d
    p = (d / name).resolve()
    if d.resolve() not in p.parents:
        raise ValueError("bad file name")
    return p


def create(title: str, source_name: str = "", kind: str = "upload", language: str = "auto") -> dict:
    mid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    d = _dir(mid)
    d.mkdir(parents=True, exist_ok=True)
    meta = {
        "id": mid,
        "title": title or Path(source_name).stem or "Meeting",
        "created": time.time(),
        "kind": kind,  # upload | live
        "source": "",
        "source_name": source_name,
        "language": language,
        "detected_language": "",
        "duration": 0,
        "status": "new",  # new | transcribing | transcribed | live | reporting | done | error
        "error": "",
        "transcript": "",  # transcript.json | transcript_live.json
        "translations": [],
        "reports": [],
    }
    save_meta(meta)
    return meta


def save_meta(meta: dict) -> None:
    with _lock:
        p = _dir(meta["id"]) / "meta.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)


def get(mid: str) -> dict:
    p = _dir(mid) / "meta.json"
    if not p.exists():
        raise KeyError(mid)
    return json.loads(p.read_text(encoding="utf-8"))


def update(mid: str, **changes) -> dict:
    with _lock:
        meta = get(mid)
        meta.update(changes)
    save_meta(meta)
    return meta


def list_all() -> list[dict]:
    out = []
    if not config.MEETINGS_DIR.exists():
        return out
    for d in config.MEETINGS_DIR.iterdir():
        f = d / "meta.json"
        if f.exists():
            try:
                out.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass
    out.sort(key=lambda m: m.get("created", 0), reverse=True)
    return out


def delete(mid: str) -> None:
    shutil.rmtree(_dir(mid), ignore_errors=True)


def read_json(mid: str, name: str):
    p = path(mid, name)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(mid: str, name: str, data) -> None:
    p = path(mid, name)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def files(mid: str) -> list[dict]:
    d = _dir(mid)
    out = []
    for f in sorted(d.iterdir()):
        if f.is_file() and f.name != "meta.json" and not f.name.endswith(".tmp"):
            out.append({"name": f.name, "size": f.stat().st_size})
    return out


def current_transcript(mid: str) -> dict | None:
    meta = get(mid)
    for name in (meta.get("transcript"), "transcript.json", "transcript_live.json"):
        if name:
            t = read_json(mid, name)
            if t:
                return t
    return None
