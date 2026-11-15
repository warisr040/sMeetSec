"""Live transcription.

The browser captures microphone / system audio, converts it to 16 kHz mono int16 and
streams it over a WebSocket. This module:
  * records everything to audio.wav (so a full-quality batch pass can run afterwards),
  * cuts speech into utterances with a simple energy detector,
  * shows 'partial' text for the utterance in progress and 'commit' text when it ends.

Load control:
  * one worker thread; a partial is only computed from the newest audio, so it never builds a backlog,
  * if the worker falls behind or CPU is over the limit, partials are skipped first,
  * if CPU stays over the limit for 15 s, live transcription stops but recording continues.
"""
from __future__ import annotations

import collections
import logging
import threading
import time
import wave

import numpy as np

from . import config, hardware, meetings
from .governor import governor
from .jobs import jobs
from .transcriber import SAMPLE_RATE, is_hallucination, models

log = logging.getLogger("smeetsec.live")

FRAME = int(0.1 * SAMPLE_RATE)  # 100 ms analysis frames
MIN_UTT = 1.2  # s, shortest utterance worth committing
TAIL_SILENCE = 0.7  # s of silence that ends an utterance
MAX_UTT = 15.0  # s, force a commit for long monologues
CPU_STOP_AFTER = 15.0  # s over the limit before live transcription is stopped


class LiveSession:
    def __init__(self, title: str, language: str, emit):
        self.emit = emit
        self.language = language
        s = config.load_settings()
        self.eff = hardware.effective(s)
        self.meta = meetings.create(title or "Live meeting", kind="live", language=language)
        self.mid = self.meta["id"]
        meetings.update(self.mid, status="live", source="audio.wav", source_name="Live recording")
        self._wav = wave.open(str(meetings.path(self.mid, "audio.wav")), "wb")
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(SAMPLE_RATE)
        self.total_samples = 0

        self._lock = threading.Lock()
        self.pending = np.zeros(0, dtype=np.float32)
        self.pending_start = 0  # absolute sample index of pending[0]
        self.segments: list[dict] = []
        self.rms_hist: collections.deque[float] = collections.deque(maxlen=150)  # last 15 s of 100 ms frames
        self._carry = np.zeros(0, dtype=np.float32)
        self.transcribing = True
        self.partials_on = True
        self._high_since: float | None = None
        self._last_partial = 0.0
        self._last_partial_len = 0
        self._stop = threading.Event()
        self.model = None
        self.hotwords = " ".join(t for t, _ in config.glossary_terms())[:400] or None
        jobs.live_active.set()
        self._t = threading.Thread(target=self._run, name="live-worker", daemon=True)
        self._t.start()

    # ---------------------------------------------------------- input
    def feed(self, pcm: bytes) -> None:
        if self._stop.is_set() or not pcm:
            return
        self._wav.writeframes(pcm)
        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        with self._lock:
            self.total_samples += len(x)
            buf = np.concatenate([self._carry, x])
            n = len(buf) // FRAME
            if n:
                self.rms_hist.extend(np.sqrt(np.mean(buf[: n * FRAME].reshape(n, FRAME) ** 2, axis=1)).tolist())
            self._carry = buf[n * FRAME:]
            if self.transcribing:
                self.pending = np.concatenate([self.pending, x])
            else:
                self.pending_start = self.total_samples

    # ---------------------------------------------------------- worker
    def _run(self) -> None:
        try:
            self.emit({"type": "status", "state": "loading",
                       "msg": f"Loading {self.eff['live_model']} ({self.eff['device']})"})
            self.model = models.get(self.eff["live_model"], self.eff["device"], self.eff["live_compute"],
                                    self.eff["cpu_threads"])
            note = " (experimental on CPU)" if self.eff["live_experimental"] else ""
            self.emit({"type": "status", "state": "listening", "msg": "Listening" + note})
        except Exception as e:
            log.exception("live model load failed")
            self.transcribing = False
            self.emit({"type": "status", "state": "error",
                       "msg": f"Could not load the live model ({e}). Recording continues."})
        while not self._stop.is_set():
            time.sleep(0.25)
            if self.model is None or not self.transcribing:
                continue
            try:
                self._cpu_guard()
                self._step()
            except Exception as e:
                log.exception("live step failed")
                self.emit({"type": "status", "state": "warn", "msg": f"Live error: {e}"})
                time.sleep(1)

    def _cpu_guard(self) -> None:
        over = governor.over()
        now = time.time()
        if over:
            if self._high_since is None:
                self._high_since = now
                self.partials_on = False
                self.emit({"type": "status", "state": "throttled",
                           "msg": f"CPU {governor.usage():.0f}% - preview text paused"})
            elif now - self._high_since > CPU_STOP_AFTER:
                self.transcribing = False
                with self._lock:
                    self.pending = np.zeros(0, dtype=np.float32)
                    self.pending_start = self.total_samples
                governor.log("CPU high during live - live transcription stopped, recording continues")
                self.emit({"type": "status", "state": "stopped_cpu",
                           "msg": "CPU stayed over the limit: live transcription stopped. Recording continues - "
                                  "run 'Full-quality transcription' after the meeting."})
        else:
            if self._high_since is not None:
                self._high_since = None
                self.partials_on = True
                self.emit({"type": "status", "state": "listening", "msg": "Listening"})

    def _frames_rms(self, x: np.ndarray) -> np.ndarray:
        n = len(x) // FRAME
        if n == 0:
            return np.zeros(0)
        return np.sqrt(np.mean(x[: n * FRAME].reshape(n, FRAME) ** 2, axis=1))

    def _step(self, final: bool = False) -> None:
        with self._lock:
            win = self.pending.copy()
            start = self.pending_start
        dur = len(win) / SAMPLE_RATE
        if dur < 0.5 and not final:
            return
        rms = self._frames_rms(win)
        with self._lock:
            hist = np.array(self.rms_hist) if self.rms_hist else np.array([0.003])
        noise_floor = float(np.percentile(hist, 10))  # pauses between words set the floor
        thr = float(np.clip(noise_floor * 3.0, 0.008, 0.05))
        voiced = rms > thr
        if not voiced.any():
            if dur > 2.0 or final:  # silence only: drop it, keep 0.3 s of lead-in
                self._advance(len(win) - int(0.3 * SAMPLE_RATE) if not final else len(win))
            return

        tail_frames = int(TAIL_SILENCE * 10)
        tail_silent = len(voiced) > tail_frames and not voiced[-tail_frames:].any()
        if final or (tail_silent and dur >= MIN_UTT) or dur >= MAX_UTT:
            cut = len(win)
            if dur >= MAX_UTT and not tail_silent and not final:
                # cut at the quietest frame in the last 3 s, so words are not split
                look = rms[-30:]
                cut = (len(rms) - len(look) + int(np.argmin(look))) * FRAME
            self._commit(win[:cut], start)
            self._advance(cut)
            return

        # partial preview for the utterance in progress
        now = time.time()
        if (self.partials_on and now - self._last_partial >= self.eff["partial_interval"]
                and len(win) - self._last_partial_len >= int(0.5 * SAMPLE_RATE)):
            t0 = time.time()
            text = self._transcribe(win, beam=1)
            took = time.time() - t0
            self._last_partial = time.time()
            self._last_partial_len = len(win)
            # if a partial took longer than the audio it covers, we're too slow: skip partials for a while
            if took > self.eff["partial_interval"] * 2:
                self._last_partial = time.time() + took
            self.emit({"type": "partial", "text": text})

    def _advance(self, n: int) -> None:
        n = max(0, n)
        with self._lock:
            self.pending = self.pending[n:]
            self.pending_start += n
        self._last_partial_len = 0

    def _prompt(self) -> str | None:
        if not self.segments:
            return config.zh_prompt() if self.language == "zh" else None
        return " ".join(s["text"] for s in self.segments[-3:])[-200:]

    def _transcribe(self, x: np.ndarray, beam: int) -> str:
        segs, _info = self.model.transcribe(
            x, language=None if self.language in ("auto", "mixed") else self.language,
            beam_size=beam, vad_filter=False, condition_on_previous_text=False,
            initial_prompt=self._prompt(), hotwords=self.hotwords, temperature=0.0)
        return " ".join(s.text.strip() for s in segs
                        if not is_hallucination(s.text, s.no_speech_prob, s.avg_logprob)).strip()

    def _commit(self, x: np.ndarray, start: int) -> None:
        if len(x) < int(0.3 * SAMPLE_RATE):
            return
        segs, info = self.model.transcribe(
            x, language=None if self.language in ("auto", "mixed") else self.language,
            beam_size=3, vad_filter=False, condition_on_previous_text=False,
            initial_prompt=self._prompt(), hotwords=self.hotwords, temperature=[0.0, 0.2])
        t0 = start / SAMPLE_RATE
        for s in segs:
            if is_hallucination(s.text, s.no_speech_prob, s.avg_logprob):
                continue
            seg = {"start": round(t0 + s.start, 2), "end": round(t0 + s.end, 2), "text": s.text.strip(),
                   "speaker": "", "lang": getattr(info, "language", "")}
            self.segments.append(seg)
            self.emit({"type": "commit", "segment": seg})
        self.emit({"type": "partial", "text": ""})

    # ---------------------------------------------------------- stop
    def stop(self) -> dict:
        if self._stop.is_set():
            return meetings.get(self.mid)
        self._stop.set()
        self._t.join(timeout=30)
        try:
            if self.model is not None and self.transcribing:
                self._step(final=True)
        except Exception:
            log.exception("final live commit failed")
        self._wav.close()
        jobs.live_active.clear()
        dur = self.total_samples / SAMPLE_RATE
        langs = [s.get("lang") for s in self.segments if s.get("lang")]
        lang = max(set(langs), key=langs.count) if langs else ""
        t = {"language": lang, "duration": round(dur, 2), "model": self.eff["live_model"],
             "device": self.eff["device"], "live": True, "segments": self.segments}
        meetings.write_json(self.mid, "transcript_live.json", t)
        from .report import export_transcript
        export_transcript(self.mid, t)
        return meetings.update(self.mid, status="transcribed", transcript="transcript_live.json",
                               duration=round(dur, 2), detected_language=lang)
