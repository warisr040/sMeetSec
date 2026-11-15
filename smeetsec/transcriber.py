"""Speech-to-text with faster-whisper (offline). Handles audio and video files."""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np

from . import config, hardware
from .governor import governor

log = logging.getLogger("smeetsec.transcriber")

SAMPLE_RATE = 16000

# Phrases Whisper tends to invent on silence/noise.
HALLUCINATIONS = {
    "thank you for watching", "thanks for watching", "please subscribe",
    "ご視聴ありがとうございました", "ご視聴ありがとうございました。", "チャンネル登録お願いします",
    "ขอบคุณที่รับชม", "ขอบคุณที่รับชมครับ", "ขอบคุณที่รับชมค่ะ", "ซับไตเติ้ลโดย",
    "谢谢观看", "謝謝觀看", "谢谢大家观看", "字幕由amara.org社区提供", "字幕由amara.org社區提供",
    "请不吝点赞 订阅 转发 打赏支持明镜与点点栏目", "点赞订阅转发", "明镜与点点栏目",
}


def is_hallucination(text: str, no_speech_prob: float = 0.0, avg_logprob: float = 0.0) -> bool:
    t = text.strip().lower().strip(".!。！ ")
    if not t:
        return True
    if t in HALLUCINATIONS:
        return True
    if no_speech_prob > 0.6 and avg_logprob < -1.0:
        return True
    return False


class ModelCache:
    """Keeps loaded models so the second job is fast. Falls back to CPU if CUDA libs are missing."""

    def __init__(self) -> None:
        self._models: dict[tuple, object] = {}
        self._lock = threading.Lock()
        self.last_fallback: str | None = None

    def get(self, name: str, device: str, compute: str, threads: int):
        if config.MOCK:
            return MockModel(name)
        key = (name, device, compute, threads)
        from .models import WHISPER_REPOS, _whisper_present
        if name in WHISPER_REPOS and not _whisper_present(name) and not Path(config.resolve_model_ref(name)).exists():
            raise RuntimeError(f"The speech model '{name}' is not installed. Open Models (top bar) and press Download.")
        with self._lock:
            if key in self._models:
                return self._models[key]
            from faster_whisper import WhisperModel
            ref = config.resolve_model_ref(name)
            try:
                m = WhisperModel(ref, device=device, compute_type=compute, cpu_threads=threads,
                                 num_workers=1, download_root=str(config.MODELS_DIR))
            except Exception as e:  # typically missing cuBLAS/cuDNN on Windows
                if device != "cuda":
                    raise
                self.last_fallback = f"GPU load failed ({e}); using CPU instead."
                log.warning(self.last_fallback)
                key = (name, "cpu", "int8", threads)
                if key in self._models:
                    return self._models[key]
                m = WhisperModel(ref, device="cpu", compute_type="int8", cpu_threads=threads,
                                 num_workers=1, download_root=str(config.MODELS_DIR))
            self._models[key] = m
            return m

    def unload_all(self) -> None:
        with self._lock:
            self._models.clear()


models = ModelCache()


def load_audio(path: str | Path) -> np.ndarray:
    """Decode any audio/video file to 16 kHz mono float32 (uses PyAV, no ffmpeg install needed)."""
    if config.MOCK:
        import wave
        try:
            with wave.open(str(path), "rb") as w:
                n = w.getnframes()
                return np.zeros(int(n * SAMPLE_RATE / w.getframerate()), dtype=np.float32)
        except Exception:
            return np.zeros(SAMPLE_RATE * 95, dtype=np.float32)
    return _decode(path)


def _decode(path: str | Path) -> np.ndarray:
    """Own decoder (PyAV) - works across PyAV versions (faster-whisper's helper breaks on PyAV >= 16)."""
    import av

    try:
        container = av.open(str(path), mode="r", metadata_errors="ignore")
    except TypeError:
        container = av.open(str(path), mode="r")
    chunks = []
    with container:
        if not container.streams.audio:
            raise ValueError("The file has no audio track.")
        stream = container.streams.audio[0]
        resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)
        for frame in container.decode(stream):
            for f in resampler.resample(frame):
                chunks.append(f.to_ndarray().reshape(-1))
        for f in resampler.resample(None):
            chunks.append(f.to_ndarray().reshape(-1))
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks).astype(np.float32) / 32768.0


def transcribe_file(path: str | Path, language: str = "auto", quality: str = "best",
                    progress=None, cancel: threading.Event | None = None,
                    audio: np.ndarray | None = None) -> dict:
    """
    language: auto | mixed | ja | en | th
    quality:  best (tier batch model) | fast (live model)
    progress: callback(fraction 0..1, message)
    """
    s = config.load_settings()
    eff = hardware.effective(s)
    name = eff["batch_model"] if quality == "best" else eff["live_model"]
    compute = eff["batch_compute"] if quality == "best" else eff["live_compute"]

    def report(f, msg):
        if progress:
            progress(f, msg)

    report(0.01, "Decoding audio")
    if audio is None:
        audio = load_audio(path)
    duration = len(audio) / SAMPLE_RATE
    if duration < 0.5:
        raise ValueError("The file has no audio (or it could not be decoded).")

    report(0.03, f"Loading model {name} ({eff['device']}, {eff['cpu_threads']} threads)")
    model = models.get(name, eff["device"], compute, eff["cpu_threads"])

    lang = None if language in ("auto", "mixed", "", None) else language
    kwargs = dict(
        language=lang,
        multilingual=(language == "mixed"),
        beam_size=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,  # avoids repetition loops in long meetings
        temperature=[0.0, 0.2, 0.4, 0.6],
        initial_prompt=config.zh_prompt() if lang == "zh" else None,
        hotwords=" ".join(t for t, _ in config.glossary_terms())[:400] or None,
    )

    segments_gen, info = model.transcribe(audio, **kwargs)
    detected = info.language
    used_model = name

    # Switch to the Thai fine-tuned model when the meeting is Thai and one is installed.
    thai = (s.get("thai_model") or "").strip()
    if detected == "th" and thai and language != "mixed" and not config.MOCK:
        ref = config.resolve_model_ref(thai)
        if Path(ref).exists():
            report(0.05, "Thai detected - switching to Thai model")
            model = models.get(thai, eff["device"], compute, eff["cpu_threads"])
            kwargs["language"] = "th"
            segments_gen, info = model.transcribe(audio, **kwargs)
            used_model = thai

    # Chinese detected automatically: re-run with the script prompt so output is all Simplified or all Traditional.
    if detected == "zh" and lang is None and language != "mixed":
        kwargs["language"] = "zh"
        kwargs["initial_prompt"] = config.zh_prompt()
        segments_gen, info = model.transcribe(audio, **kwargs)

    report(0.06, f"Transcribing ({detected}, {duration/60:.1f} min)")
    out = []
    for seg in segments_gen:
        # CPU guard between ~30 s windows: throttle (wait) or stop.
        governor.checkpoint(cancel, on_wait=lambda u: report(
            0.06 + 0.92 * min(1.0, seg.end / duration), f"Paused: CPU {u:.0f}% over limit"))
        if is_hallucination(seg.text, seg.no_speech_prob, seg.avg_logprob):
            continue
        out.append({
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
            "speaker": "",
        })
        report(0.06 + 0.92 * min(1.0, seg.end / duration),
               f"Transcribing {fmt_time(seg.end)} / {fmt_time(duration)}")

    report(1.0, "Done")
    return {
        "language": detected,
        "language_probability": round(float(info.language_probability or 0), 3),
        "duration": round(duration, 2),
        "model": used_model,
        "device": eff["device"],
        "segments": out,
    }


def fmt_time(sec: float, srt: bool = False) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int(sec % 3600 // 60)
    s = sec % 60
    if srt:
        return f"{h:02d}:{m:02d}:{int(s):02d},{int((s - int(s)) * 1000):03d}"
    if h:
        return f"{h}:{m:02d}:{int(s):02d}"
    return f"{m:02d}:{int(s):02d}"


# ---------------------------------------------------------------- mock (for UI testing)
class _Seg:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text
        self.no_speech_prob, self.avg_logprob = 0.0, -0.2


class _Info:
    language = "en"
    language_probability = 0.99


class MockModel:
    LINES = [
        "Good morning everyone, let's start the weekly project meeting.",
        "First item is the ECU flashing schedule for next month.",
        "We agreed the UDS test bench must be ready by the 20th.",
        "Somchai will prepare the CAN log analysis before Friday.",
        "Tanaka-san asked whether the Japanese supplier can deliver earlier.",
        "We still need to confirm the budget for the extra test units.",
        "Okay, any other topics? If not, let's close the meeting.",
    ]

    def __init__(self, name):
        self.name = name

    def transcribe(self, audio, **kw):
        import time
        dur = max(1.0, len(audio) / SAMPLE_RATE)
        n = max(1, min(len(self.LINES), int(dur // 3) or 1))
        step = dur / n

        def gen():
            for i in range(n):
                time.sleep(0.15)
                yield _Seg(i * step, (i + 1) * step, self.LINES[i % len(self.LINES)])
        return gen(), _Info()
