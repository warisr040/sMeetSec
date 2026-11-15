"""Model manager: what is installed, and one-click download (needs internet once).

Everything goes into the models\\ folder next to sMeetSec.exe, so a models folder can also be
copied from another PC (offline install).
  models\\<whisper-name>\\        speech recognition (CTranslate2 Whisper)
  models\\diarization\\           speaker identification (sherpa-onnx)
  models\\llm\\*.gguf             translation / report model (llama.cpp)
"""
from __future__ import annotations

import json
import shutil
import tarfile
import threading
import urllib.request
from pathlib import Path

from . import config, diarize, hardware, llm

HF = "https://huggingface.co"
WHISPER_REPOS = {
    "large-v3": "Systran/faster-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "medium": "Systran/faster-whisper-medium",
    "small": "Systran/faster-whisper-small",
    "base": "Systran/faster-whisper-base",
}
WHISPER_FILES = ("config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.json",
                 "vocabulary.txt")
LLM_REPOS = {
    "9B": "bartowski/Qwen_Qwen3.5-9B-GGUF",
    "4B": "bartowski/Qwen_Qwen3.5-4B-GGUF",
}
LLM_QUANT = "Q4_K_M"


class Cancelled(Exception):
    pass


def _whisper_needed() -> list[str]:
    e = hardware.effective(config.load_settings())
    names = [e["batch_model"]]
    if e["live_allowed"] and e["live_model"] not in names:
        names.append(e["live_model"])
    return [n for n in names if n in WHISPER_REPOS]


def _whisper_present(name: str) -> bool:
    d = config.MODELS_DIR / name
    return (d / "model.bin").exists() and (d / "config.json").exists()


def status() -> list[dict]:
    if config.MOCK:
        return [{"id": "whisper", "label": "Speech recognition (Whisper large-v3-turbo)", "present": True, "size": "1.6 GB"},
                {"id": "speakers", "label": "Speaker identification", "present": True, "size": "46 MB"},
                {"id": "llm", "label": "Translation & reports (Qwen3.5 4B)", "present": True, "size": "3.0 GB"}]
    tier = hardware.detect()["tier"]
    need = _whisper_needed()
    size = {"large-v3": "3.1 GB", "large-v3-turbo": "1.6 GB", "medium": "1.5 GB", "small": "0.5 GB", "base": "0.15 GB"}
    m = llm.pick_gguf()
    return [
        {"id": "whisper", "label": "Speech recognition (Whisper " + ", ".join(need) + ")",
         "present": all(_whisper_present(n) for n in need), "size": " + ".join(size.get(n, "?") for n in need)},
        {"id": "speakers", "label": "Speaker identification", "present": diarize.SEG_MODEL.exists() and diarize.EMB_MODEL.exists(),
         "size": "46 MB"},
        {"id": "llm", "label": f"Translation & reports ({m.name if m else 'Qwen3.5 ' + tier['llm_size']})",
         "present": m is not None, "size": "6 GB" if tier["llm_size"] == "9B" else "3 GB",
         "engine": llm.engine()},
    ]


def all_present() -> bool:
    return all(x["present"] for x in status() if not (x["id"] == "llm" and llm.engine() == "ollama"))


# ---------------------------------------------------------------- download helpers
def _get_json(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "sMeetSec"}), timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def _download(url: str, dest: Path, progress, cancel: threading.Event, label: str) -> None:
    """Stream to dest.part, then rename (so an interrupted download is never mistaken for a complete file)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "sMeetSec"})
    with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done = 0
        while True:
            if cancel.is_set():
                f.close()
                tmp.unlink(missing_ok=True)
                raise Cancelled()
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if progress:
                mb = done / 1048576
                progress(done / total if total else 0.0,
                         f"{label}: {mb:,.0f} / {total / 1048576:,.0f} MB" if total else f"{label}: {mb:,.0f} MB")
    tmp.replace(dest)


def download_whisper(name: str, progress, cancel) -> None:
    repo = WHISPER_REPOS[name]
    files = [s["rfilename"] for s in _get_json(f"{HF}/api/models/{repo}").get("siblings", [])]
    files = [f for f in files if f in WHISPER_FILES]
    dest = config.MODELS_DIR / name
    for i, fn in enumerate(sorted(files, key=lambda x: x == "model.bin")):  # model.bin last
        if (dest / fn).exists():
            continue
        _download(f"{HF}/{repo}/resolve/main/{fn}", dest / fn, progress, cancel, f"Whisper {name} - {fn}")


def download_speakers(progress, cancel) -> None:
    d = diarize.DIAR_DIR
    if not diarize.SEG_MODEL.exists():
        tmp = d / "seg.tar.bz2"
        _download(diarize.URLS["seg"], tmp, progress, cancel, "Speaker segmentation")
        with tarfile.open(tmp) as tf:
            tf.extractall(d)
        tmp.unlink(missing_ok=True)
    if not diarize.EMB_MODEL.exists():
        _download(diarize.URLS["emb"], diarize.EMB_MODEL, progress, cancel, "Speaker embedding")


def download_llm(progress, cancel, size: str | None = None) -> None:
    size = size or hardware.detect()["tier"]["llm_size"]
    repo = LLM_REPOS[size]
    files = [s["rfilename"] for s in _get_json(f"{HF}/api/models/{repo}").get("siblings", [])]
    cands = [f for f in files if f.endswith(".gguf") and LLM_QUANT in f and "/" not in f and "mmproj" not in f.lower()]
    if not cands:
        raise RuntimeError(f"No {LLM_QUANT} GGUF found in {repo}")
    fn = sorted(cands, key=len)[0]
    dest = llm.LLM_DIR / fn
    if not dest.exists():
        _download(f"{HF}/{repo}/resolve/main/{fn}", dest, progress, cancel, f"Qwen3.5 {size}")


def run_download_job(job, ids: list[str]) -> None:
    """Background job: download the selected model groups."""
    if config.MOCK:
        import time
        for i in range(10):
            time.sleep(0.1)
            job.update(i / 10, "Downloading (mock)")
        return
    ids = [i for i in ("speakers", "whisper", "llm") if i in ids]  # small first
    errors = []
    for k, mid in enumerate(ids):
        a, b = k / len(ids), (k + 1) / len(ids)

        def prog(f, m, a=a, b=b):
            job.update(a + (b - a) * min(1.0, f), m)
        try:
            if mid == "whisper":
                for n in _whisper_needed():
                    download_whisper(n, prog, job.cancel)
            elif mid == "speakers":
                download_speakers(prog, job.cancel)
            elif mid == "llm":
                download_llm(prog, job.cancel)
        except Cancelled:
            from .governor import JobCancelled
            raise JobCancelled()
        except Exception as e:  # keep going with the other models
            errors.append(f"{mid}: {e}")
    if errors:
        raise RuntimeError("Some downloads failed (check the internet connection, then press Download again): "
                           + "; ".join(errors))


def disk_free_gb() -> float:
    try:
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        return round(shutil.disk_usage(config.MODELS_DIR).free / 1024**3, 1)
    except Exception:
        return 0.0
