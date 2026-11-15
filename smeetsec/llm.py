"""Local LLM for translation and reports.

Two engines:
  * builtin (default) - llama.cpp's llama-server.exe shipped with sMeetSec + a GGUF model in models\\llm\\.
    No Ollama, no Python install. sMeetSec starts it on demand and stops it after 10 idle minutes.
  * ollama (optional)  - an Ollama server the user installed separately.
Both run only on 127.0.0.1.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import psutil

from . import config, hardware


class LLMError(Exception):
    pass


LLM_DIR = config.MODELS_DIR / "llm"


# ================================================================ builtin engine (llama.cpp)
def llama_server_path() -> Path | None:
    env = os.environ.get("SMEETSEC_LLAMA_SERVER")
    cands = [Path(env)] if env else []
    names = ["llama-server.exe", "llama-server"]
    for d in (config.APP_DIR / "llama", config.APP_DIR / "llama" / "bin", config.APP_DIR):
        cands += [d / n for n in names]
    for c in cands:
        if c.exists():
            return c
    w = shutil.which("llama-server")
    return Path(w) if w else None


def gguf_files() -> list[Path]:
    if not LLM_DIR.exists():
        return []
    return sorted([p for p in LLM_DIR.glob("*.gguf") if "mmproj" not in p.name.lower()],
                  key=lambda p: p.stat().st_size)


def pick_gguf() -> Path | None:
    s = config.load_settings()
    files = gguf_files()
    want = (s.get("llm_gguf") or "auto").strip()
    if want and want != "auto":
        for f in files:
            if f.name == want:
                return f
    if not files:
        return None
    pref = hardware.detect()["tier"]["llm_size"]  # "9B" or "4B"
    for f in files:
        if pref.lower() in f.name.lower():
            return f
    return files[-1] if pref == "9B" else files[0]


class _Server:
    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self.port = 0
        self.model: Path | None = None
        self.threads = 0
        self.last_used = 0.0
        self.lock = threading.Lock()
        threading.Thread(target=self._idle_reaper, daemon=True).start()

    def _free_port(self) -> int:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def ensure(self, model: Path, threads: int, ctx: int) -> int:
        with self.lock:
            if self.proc and self.proc.poll() is None and self.model == model and self.threads == threads:
                self.last_used = time.time()
                return self.port
            self.stop_locked()
            exe = llama_server_path()
            if not exe:
                raise LLMError("The built-in translation engine (llama-server) is missing. Reinstall sMeetSec.")
            self.port = self._free_port()
            log = open(config.DATA_DIR / "llm.log", "w", encoding="utf-8", errors="replace")
            args = [str(exe), "-m", str(model), "--host", "127.0.0.1", "--port", str(self.port),
                    "-t", str(threads), "-c", str(ctx), "--jinja", "-np", "1"]
            gl = int(config.load_settings().get("llm_gpu_layers") or 0)
            if gl:
                args += ["-ngl", str(gl)]
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
            self.proc = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, cwd=str(exe.parent),
                                         creationflags=flags)
            if os.name != "nt":
                try:
                    psutil.Process(self.proc.pid).nice(10)
                except Exception:
                    pass
            self.model, self.threads = model, threads
            deadline = time.time() + 300
            while time.time() < deadline:
                if self.proc.poll() is not None:
                    raise LLMError("The translation engine stopped while loading the model (see data\\llm.log). "
                                   "The model file may be incomplete or need a newer engine.")
                try:
                    with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=2) as r:
                        if r.status == 200:
                            self.last_used = time.time()
                            return self.port
                except Exception:
                    pass
                time.sleep(0.5)
            self.stop_locked()
            raise LLMError("The translation engine did not start in time (see data\\llm.log).")

    def stop_locked(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except Exception:
                self.proc.kill()
        self.proc = None

    def stop(self) -> None:
        with self.lock:
            self.stop_locked()

    def _idle_reaper(self) -> None:  # free the RAM after 10 idle minutes
        while True:
            time.sleep(30)
            if self.proc and time.time() - self.last_used > 600:
                self.stop()


_server: _Server | None = None


def _builtin_chat(system: str, user: str, json_mode: bool, num_ctx: int, temperature: float, timeout: int) -> str:
    global _server
    model = pick_gguf()
    if not model:
        raise LLMError("No translation model installed. Open Models (top bar) and download it.")
    eff = hardware.effective(config.load_settings())
    if _server is None:
        _server = _Server()
    port = _server.ensure(model, eff["cpu_threads"], max(num_ctx, 8192))
    body = {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": 4096,
        "stream": False,
        # Qwen3.x: answer directly, no "thinking" (much faster on CPU)
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise LLMError("Translation engine error: " + e.read().decode("utf-8", "replace")[:300])
    except urllib.error.URLError as e:
        raise LLMError(f"Translation engine not reachable: {e}")
    finally:
        _server.last_used = time.time()
    return (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""


def shutdown() -> None:
    if _server:
        _server.stop()


# ================================================================ Ollama engine (optional)
def _url(path: str) -> str:
    return config.load_settings()["ollama_url"].rstrip("/") + path


def _ollama_status(eff: dict) -> dict:
    try:
        with urllib.request.urlopen(_url("/api/tags"), timeout=3) as r:
            data = json.loads(r.read().decode("utf-8"))
        names = [m["name"] for m in data.get("models", [])]
        sel = eff["llm_model"]
        installed = any(n == sel or n.split(":")[0] == sel or n == sel + ":latest" for n in names)
        return {"engine": "ollama", "running": True, "models": names, "selected": sel, "installed": installed}
    except Exception as e:
        return {"engine": "ollama", "running": False, "models": [], "selected": eff["llm_model"],
                "installed": False, "error": str(e)}


def lower_ollama_priority() -> None:
    for p in psutil.process_iter(["name"]):
        try:
            if (p.info["name"] or "").lower().startswith("ollama"):
                p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
        except Exception:
            pass


def _ollama_chat(system, user, json_mode, num_ctx, temperature, timeout) -> str:
    eff = hardware.effective(config.load_settings())
    body = {
        "model": eff["llm_model"],
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "stream": False,
        "options": {"temperature": temperature, "num_ctx": num_ctx, "num_thread": eff["cpu_threads"]},
        "keep_alive": "10m",
        "think": False,
    }
    if json_mode:
        body["format"] = "json"
    req = urllib.request.Request(_url("/api/chat"), data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    lower_ollama_priority()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")
        if "not found" in detail:
            raise LLMError(f"Model '{eff['llm_model']}' is not installed. Run: ollama pull {eff['llm_model']}")
        raise LLMError(f"Ollama error: {detail}")
    except urllib.error.URLError:
        raise LLMError("Ollama is not running. Start Ollama, or switch Settings > Translation engine to Built-in.")
    return data.get("message", {}).get("content", "") or ""


# ================================================================ public API
def engine() -> str:
    return "ollama" if config.load_settings().get("llm_engine") == "ollama" else "builtin"


def status() -> dict:
    if config.MOCK:
        return {"engine": "builtin", "running": True, "models": ["mock"], "selected": "mock", "installed": True}
    s = config.load_settings()
    eff = hardware.effective(s)
    if engine() == "ollama":
        return _ollama_status(eff)
    m = pick_gguf()
    return {"engine": "builtin", "running": llama_server_path() is not None,
            "models": [f.name for f in gguf_files()], "selected": m.name if m else "",
            "installed": m is not None}


def chat(system: str, user: str, json_mode: bool = False, num_ctx: int = 8192,
         temperature: float = 0.2, timeout: int = 1800) -> str:
    if config.MOCK:
        return _mock(system, user, json_mode)
    fn = _ollama_chat if engine() == "ollama" else _builtin_chat
    content = fn(system, user, json_mode, num_ctx, temperature, timeout)
    return re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()


def chat_json(system: str, user: str, **kw):
    """Ask for JSON; if the model ignores JSON mode, retry once in plain mode."""
    try:
        return parse_json(chat(system, user, json_mode=True, **kw))
    except LLMError as e:
        msg = str(e)
        if any(k in msg for k in ("not running", "not installed", "No translation model", "missing", "not reachable",
                                  "did not start", "stopped")):
            raise
        return parse_json(chat(system + " Return only valid JSON, no other text.", user, json_mode=False, **kw))


def parse_json(text: str):
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise LLMError("The LLM did not return valid JSON.")


# ---------------------------------------------------------------- mock
def _mock(system: str, user: str, json_mode: bool) -> str:
    time.sleep(0.3)
    if "Translate" in system and not json_mode:
        tag = "TH" if "into Thai" in system else "JA" if "into Japanese" in system else "ZH" if "into Chinese" in system else "EN"
        return "\n".join(f"{line} [{tag}]" for line in user.split("\n---\n", 1)[-1].splitlines() if line.strip())
    if json_mode and "extract notes" in system.lower():
        return json.dumps({"points": ["UDS bench ready by the 20th"], "decisions": ["Bench deadline 20th"],
                           "actions": [{"task": "CAN log analysis", "owner": "Somchai", "due": "Friday"}],
                           "questions": ["Budget for extra test units"]})
    if json_mode:
        return json.dumps({
            "title": "Weekly project meeting",
            "summary": "The team reviewed the ECU flashing schedule and the UDS test bench timeline.",
            "key_points": ["ECU flashing schedule for next month", "UDS test bench deadline"],
            "decisions": ["The UDS test bench must be ready by the 20th"],
            "action_items": [{"task": "Prepare CAN log analysis", "owner": "Somchai", "due": "Friday"}],
            "open_questions": ["Can the Japanese supplier deliver earlier?", "Budget for extra test units"],
        }, ensure_ascii=False)
    return "ok"
