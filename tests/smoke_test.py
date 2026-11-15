"""Automated end-to-end test in mock mode (no models needed).

    python tests/smoke_test.py
Starts the engine with SMEETSEC_MOCK=1 on a temporary data folder and checks: upload -> transcription
-> speakers -> rename -> translation & report (4 languages) -> files, CPU guard, settings, legal,
live WebSocket session. Prints PASS / FAIL per step.
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, info: str = "") -> None:
    results.append((name, ok, info))
    print(("PASS " if ok else "FAIL ") + name + (f"  ({info})" if info else ""))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    import httpx

    tmp = Path(tempfile.mkdtemp(prefix="smeetsec_test_"))
    wav = tmp / "test.wav"
    x = (np.sin(np.arange(16000 * 20) / 16000 * 2 * np.pi * 220) * 8000).astype(np.int16)
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(16000), w.writeframes(x.tobytes())

    port = free_port()
    env = dict(os.environ, SMEETSEC_MOCK="1", SMEETSEC_DATA=str(tmp / "data"))
    logf = open(tmp / "server.log", "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen([sys.executable, str(ROOT / "run.py"), "--no-browser", "--port", str(port)],
                            cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    c = httpx.Client(base_url=base, timeout=30)
    try:
        for _ in range(60):
            try:
                if c.get("/api/guard").status_code == 200:
                    break
            except Exception:
                time.sleep(0.25)
        sysinfo = c.get("/api/system").json()
        check("engine starts", sysinfo["app"]["mock"] is True, sysinfo["app"]["version"])
        check("CPU guard defaults", sysinfo["guard"]["limit"] == 80 and sysinfo["guard"]["action"] == "throttle")
        check("UI page served", "sMeetSec" in c.get("/").text)
        check("legal texts", "third-party" in c.get("/api/legal/THIRD-PARTY-NOTICES").json()["text"].lower())

        c.post("/api/settings", json={"report_languages": ["en", "th", "ja", "zh"], "glossary": "Somchai\nUDS"})

        def wait_jobs():
            for _ in range(120):
                js = c.get("/api/jobs").json()
                if js and all(j["status"] in ("done", "error", "stopped", "cancelled") for j in js):
                    return js
                time.sleep(0.25)
            return c.get("/api/jobs").json()

        resp = c.post("/api/upload", files={"file": ("test.wav", wav.read_bytes())},
                      data={"title": "Smoke test", "language": "en", "auto_report": "true", "speakers": "2"})
        if resp.status_code != 200:
            print("upload failed:", resp.status_code, resp.text[:500])
            logf.flush()
            print((tmp / "server.log").read_text(encoding="utf-8", errors="replace")[-4000:])
        r = resp.json()
        mid = r["meeting"]["id"]
        js = wait_jobs()
        check("transcribe + report jobs", all(j["status"] == "done" for j in js), str([(j["kind"], j["status"], j["error"]) for j in js]))
        m = c.get(f"/api/meetings/{mid}").json()
        segs = m["transcript"]["segments"]
        check("transcript segments", len(segs) > 0, f"{len(segs)} segments")
        spk = sorted({s["speaker"] for s in segs})
        check("speakers assigned", spk == ["Speaker 1", "Speaker 2"], str(spk))
        check("reports in 4 languages", sorted(m["reports"]) == ["en", "ja", "th", "zh"], str(sorted(m["reports"])))
        names = {f["name"] for f in m["files"]}
        need = {"transcript.txt", "transcript.srt", "report.html", "report_zh.docx", "report_th.md", "transcript_ja.txt"}
        check("output files", need <= names, ", ".join(sorted(need - names)))

        c.post(f"/api/meetings/{mid}/rename_speaker", json={"old": "Speaker 1", "new": "Pack"})
        segs = c.get(f"/api/meetings/{mid}").json()["transcript"]["segments"]
        check("rename speaker everywhere", "Pack" in {s["speaker"] for s in segs} and "Speaker 1" not in {s["speaker"] for s in segs})
        txt = c.get(f"/api/meetings/{mid}/files/transcript.txt").text
        check("rename reaches exports", "Pack:" in txt)

        c.post(f"/api/meetings/{mid}/speakers", json={"speakers": 2})
        js = wait_jobs()
        check("identify speakers job", js[0]["status"] == "done", js[0]["error"])

        c.post(f"/api/meetings/{mid}/report", json={"languages": ["en"], "translate_transcript": False})
        js = wait_jobs()
        rep = c.get(f"/api/meetings/{mid}").json()["reports"]["en"]
        check("participants in report", "Pack" in rep["meta"].get("participants", ""), rep["meta"].get("participants", ""))

        # live session (needs experimental mode on CPU)
        c.post("/api/settings", json={"live_force_experimental": True})

        async def live():
            import websockets
            got = []
            async with websockets.connect(f"ws://127.0.0.1:{port}/ws/live") as ws:
                await ws.send(json.dumps({"title": "Live smoke", "language": "en"}))

                async def rx():
                    async for msg in ws:
                        d = json.loads(msg)
                        got.append(d)
                        if d["type"] == "stopped":
                            return
                t = asyncio.create_task(rx())
                rng = np.random.default_rng(1)
                for _ in range(2):
                    for _ in range(30):
                        await ws.send((rng.normal(0, 0.1, 1600) * 32767).astype(np.int16).tobytes())
                        await asyncio.sleep(0.03)
                    for _ in range(15):
                        await ws.send((rng.normal(0, 0.001, 1600) * 32767).astype(np.int16).tobytes())
                        await asyncio.sleep(0.08)
                await ws.send(json.dumps({"cmd": "stop"}))
                await asyncio.wait_for(t, 20)
            return got
        got = asyncio.run(live())
        commits = [g for g in got if g["type"] == "commit"]
        stopped = [g for g in got if g["type"] == "stopped"]
        check("live commits text", len(commits) >= 2, f"{len(commits)} commits")
        check("live saves meeting + audio", bool(stopped) and stopped[0]["meeting"]["status"] == "transcribed")
        if stopped:
            files = {f["name"] for f in c.get(f"/api/meetings/{stopped[0]['meeting']['id']}").json()["files"]}
            check("live audio.wav kept", "audio.wav" in files)

        c.post("/api/settings", json={"live_force_experimental": False})
        check("delete meeting", c.delete(f"/api/meetings/{mid}").json().get("ok") is True)
    finally:
        proc.terminate()
        try:
            proc.wait(5)
        except Exception:
            proc.kill()

    bad = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(bad)}/{len(results)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
