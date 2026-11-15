"""Offline speaker identification ("who spoke when") with sherpa-onnx.

Models (downloaded once by tools/download_models.py into models/diarization/):
  * pyannote segmentation 3.0 (ONNX, MIT)            - finds speech turns
  * 3D-Speaker ERes2Net speaker embedding (Apache-2.0) - tells voices apart
No PyTorch needed; runs on CPU with the same thread cap as the rest of the app.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np

from . import config, hardware
from .governor import CpuLimitExceeded, JobCancelled, governor

log = logging.getLogger("smeetsec.diarize")

DIAR_DIR = config.MODELS_DIR / "diarization"
SEG_MODEL = DIAR_DIR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"
EMB_MODEL = DIAR_DIR / "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"

URLS = {
    "seg": "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/"
           "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
    "emb": "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
           "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
}


def available() -> bool:
    if config.MOCK:
        return True
    try:
        import sherpa_onnx  # noqa: F401
    except Exception:
        return False
    return SEG_MODEL.exists() and EMB_MODEL.exists()


def status() -> dict:
    try:
        import sherpa_onnx  # noqa: F401
        lib = True
    except Exception:
        lib = False
    return {"available": available(), "library": lib, "models": SEG_MODEL.exists() and EMB_MODEL.exists()}


def diarize(audio: np.ndarray, num_speakers: int = 0, progress=None,
            cancel: threading.Event | None = None) -> list[dict]:
    """Return [{start, end, speaker:int}] sorted by time. num_speakers 0 = detect automatically."""
    if config.MOCK:
        return _mock(len(audio) / 16000, num_speakers or 2)
    import sherpa_onnx

    eff = hardware.effective(config.load_settings())
    threads = max(1, eff["cpu_threads"])
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(SEG_MODEL)),
            num_threads=threads,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(EMB_MODEL), num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=num_speakers if num_speakers > 0 else -1,
            threshold=float(config.load_settings().get("speaker_threshold", 0.5)),
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not cfg.validate():
        raise RuntimeError("Speaker models are missing or invalid. Run tools\\download_models.py")
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)

    state = {"err": None}

    def cb(done: int, total: int) -> int:
        # CPU guard + cancel inside the native loop: wait (throttle) or abort (stop/cancel).
        try:
            governor.checkpoint(cancel)
        except (CpuLimitExceeded, JobCancelled) as e:
            state["err"] = e
            return 1
        if progress:
            progress(done / max(1, total), f"Identifying speakers ({done}/{total})")
        return 0

    result = sd.process(np.ascontiguousarray(audio, dtype=np.float32), callback=cb).sort_by_start_time()
    if state["err"] is not None:
        raise state["err"]
    return [{"start": float(r.start), "end": float(r.end), "speaker": int(r.speaker)} for r in result]


def assign(segments: list[dict], turns: list[dict]) -> int:
    """Give each transcript segment the speaker that overlaps it most. Returns number of speakers.

    Speakers are numbered in order of first appearance: "Speaker 1", "Speaker 2", ...
    Names the user already typed are kept.
    """
    if not turns:
        return 0
    order: dict[int, int] = {}
    for t in turns:
        if t["speaker"] not in order:
            order[t["speaker"]] = len(order) + 1
    for s in segments:
        best, best_ov = None, 0.0
        for t in turns:
            if t["end"] <= s["start"]:
                continue
            if t["start"] >= s["end"]:
                break
            ov = min(s["end"], t["end"]) - max(s["start"], t["start"])
            if ov > best_ov:
                best, best_ov = t["speaker"], ov
        if best is None:  # no overlap: nearest turn
            mid = (s["start"] + s["end"]) / 2
            best = min(turns, key=lambda t: min(abs(t["start"] - mid), abs(t["end"] - mid)))["speaker"]
        s["speaker_id"] = order[best]
        if not s.get("speaker") or s["speaker"].startswith("Speaker "):
            s["speaker"] = f"Speaker {order[best]}"
    return len(order)


def download_models(log_fn=print) -> None:
    """Download the two small ONNX models (about 46 MB) - needs internet once."""
    import tarfile
    import urllib.request

    DIAR_DIR.mkdir(parents=True, exist_ok=True)
    if not SEG_MODEL.exists():
        tmp = DIAR_DIR / "seg.tar.bz2"
        log_fn("Downloading speaker segmentation model ...")
        urllib.request.urlretrieve(URLS["seg"], tmp)
        with tarfile.open(tmp) as tf:
            tf.extractall(DIAR_DIR)
        tmp.unlink(missing_ok=True)
    if not EMB_MODEL.exists():
        log_fn("Downloading speaker embedding model ...")
        urllib.request.urlretrieve(URLS["emb"], EMB_MODEL)
    log_fn("Speaker models ready in " + str(DIAR_DIR))


def _mock(duration: float, n: int) -> list[dict]:
    time.sleep(0.2)
    out, t, k = [], 0.0, 0
    while t < duration:
        out.append({"start": t, "end": min(duration, t + 6.5), "speaker": k % n})
        t += 6.5
        k += 1
    return out
