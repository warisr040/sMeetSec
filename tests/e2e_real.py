"""Real end-to-end test with small real models (used by GitHub Actions on Windows).

Downloads Whisper 'base', the speaker models and a small Qwen3.5 GGUF into a temp models folder,
then: transcribes a 4-speaker test recording, identifies speakers, and translates/summarises
with the built-in llama.cpp engine (SMEETSEC_LLAMA_SERVER must point to llama-server.exe).
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
tmp = Path(tempfile.mkdtemp(prefix="smeetsec_e2e_"))
os.environ.setdefault("SMEETSEC_MODELS", str(tmp / "models"))
os.environ.setdefault("SMEETSEC_DATA", str(tmp / "data"))
sys.path.insert(0, str(ROOT))

from smeetsec import config, diarize, llm, models, report  # noqa: E402
from smeetsec.governor import governor  # noqa: E402
from smeetsec.transcriber import load_audio, transcribe_file  # noqa: E402

WAV = "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav"


def step(name):
    print(f"\n=== {name}", flush=True)
    return time.time()


def main() -> int:
    config.ensure_dirs()
    config.save_settings({"batch_model": "base", "live_model": "base", "cpu_limit_percent": 100})
    governor.configure(100, "throttle")
    governor.start()
    cancel = threading.Event()
    prog = lambda f, m: None  # noqa: E731

    t = step("download models")
    models.download_whisper("base", prog, cancel)
    models.download_speakers(prog, cancel)
    llm_ok = False
    for size, repo in (("0.8B", "bartowski/Qwen_Qwen3.5-0.8B-GGUF"), ("2B", "bartowski/Qwen_Qwen3.5-2B-GGUF"),
                       ("4B", "bartowski/Qwen_Qwen3.5-4B-GGUF")):
        try:
            models.LLM_REPOS[size] = repo
            models.download_llm(prog, cancel, size=size)
            llm_ok = True
            print("LLM:", size, repo)
            break
        except Exception as e:
            print("LLM", size, "not available:", e)
    print(f"models ready in {time.time() - t:.0f}s:", [(i["id"], i["present"]) for i in models.status()])

    wav = tmp / "test.wav"
    urllib.request.urlretrieve(WAV, wav)

    t = step("transcribe (Whisper base, CPU)")
    tr = transcribe_file(wav, "auto", "best", progress=prog)
    print(f"{time.time() - t:.1f}s, language={tr['language']}, segments={len(tr['segments'])}")
    for s in tr["segments"][:6]:
        print(f"  [{s['start']:.1f}] {s['text']}")
    assert tr["segments"], "no transcript"
    assert tr["language"] == "zh", "expected Chinese"

    t = step("identify speakers")
    turns = diarize.diarize(load_audio(wav), 0)
    n = diarize.assign(tr["segments"], turns)
    print(f"{time.time() - t:.1f}s, speakers found: {n}")
    assert n >= 2, "expected several speakers"

    if not llm_ok:
        print("no LLM could be downloaded - skipping translation")
        return 1
    t = step("translate with built-in llama.cpp")
    for tgt in ("en", "th", "ja"):
        out = report.translate_text("截止日期从3月15日推迟到了4月底，预算为240万泰铢。", "zh", tgt)
        print(f"  zh->{tgt}: {out}")
        assert out.strip(), "empty translation"
    print(f"{time.time() - t:.1f}s")

    t = step("report (JSON) with built-in llama.cpp")
    rep = report.build_report(tr, "zh", prog, cancel)
    print("  title:", rep["title"])
    print("  summary:", rep["summary"][:200])
    assert rep["summary"], "empty report"
    print(f"{time.time() - t:.1f}s")
    llm.shutdown()
    print("\nE2E PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
