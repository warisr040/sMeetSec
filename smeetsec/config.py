"""Paths and user settings for sMeetSec."""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

APP_NAME = "sMeetSec"
APP_FULL_NAME = "simply handle IT Meeting Secretary"
APP_VERSION = "0.3.0"


def _app_dir() -> Path:
    # When frozen by PyInstaller, keep data next to the .exe, not in the temp unpack dir.
    if getattr(sys, "frozen", False):
        d = Path(sys.executable).resolve().parent
        # packaged backend lives in <sMeetSec folder>\backend\ - keep data and models in the main folder
        return d.parent if d.name.lower() == "backend" else d
    return Path(__file__).resolve().parent.parent


APP_DIR = _app_dir()
DATA_DIR = Path(os.environ.get("SMEETSEC_DATA", APP_DIR / "data"))
MEETINGS_DIR = DATA_DIR / "meetings"
MODELS_DIR = Path(os.environ.get("SMEETSEC_MODELS", APP_DIR / "models"))
SETTINGS_FILE = DATA_DIR / "settings.json"

if getattr(sys, "frozen", False):
    STATIC_DIR = Path(getattr(sys, "_MEIPASS", APP_DIR)) / "smeetsec" / "static"
else:
    STATIC_DIR = Path(__file__).resolve().parent / "static"

MOCK = os.environ.get("SMEETSEC_MOCK") == "1"

LEGAL_DIR = (Path(getattr(sys, "_MEIPASS", APP_DIR)) / "legal") if getattr(sys, "frozen", False) else APP_DIR / "legal"

LANG_NAMES = {"ja": "Japanese", "en": "English", "th": "Thai", "zh": "Chinese"}
LANG_CODES = ["en", "th", "ja", "zh"]


def lang_label(code: str) -> str:
    """Language name for LLM prompts; Chinese includes the chosen script."""
    if code == "zh":
        trad = load_settings().get("chinese_script") == "traditional"
        return "Chinese (Traditional Chinese characters, Taiwan usage)" if trad else "Chinese (Simplified Chinese characters, Mainland usage)"
    return LANG_NAMES.get(code, code)


def glossary_terms() -> list[tuple[str, str]]:
    out = []
    for line in (load_settings().get("glossary") or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            a, b = line.split("=", 1)
            out.append((a.strip(), b.strip()))
        else:
            out.append((line, ""))
    return [t for t in out if t[0]][:200]


def zh_prompt() -> str:
    """Initial prompt that steers Whisper to the chosen Chinese script (it otherwise mixes them)."""
    trad = load_settings().get("chinese_script") == "traditional"
    return "以下是普通話的會議記錄。" if trad else "以下是普通话的会议记录。"

DEFAULT_SETTINGS = {
    # "auto" picks a model from the hardware tier; or a name like "large-v3", "large-v3-turbo", "small",
    # or a folder name inside models/ containing a CTranslate2 Whisper model.
    "batch_model": "auto",
    "live_model": "auto",
    # Optional CTranslate2 Thai fine-tuned Whisper model folder (inside models/ or absolute path).
    # Used automatically when the recording is detected as Thai.
    "thai_model": "",
    "device": "auto",  # auto | cpu | cuda
    "cpu_threads": 0,  # 0 = let CTranslate2 decide
    "llm_engine": "builtin",  # builtin (llama.cpp, shipped with sMeetSec) | ollama (installed separately)
    "llm_gguf": "auto",  # GGUF file in models\llm\ (auto = by RAM: 9B with 16 GB+, else 4B)
    "llm_gpu_layers": 0,  # >0 only with a GPU build of llama.cpp
    "ollama_url": "http://127.0.0.1:11434",
    "llm_model": "auto",  # auto = picked from RAM; or any model name pulled in Ollama
    "report_languages": ["en", "th", "ja"],
    "live_force_experimental": False,
    # CPU guard
    "cpu_limit_percent": 80,
    "cpu_limit_action": "throttle",  # throttle (pause until below limit) | stop (cancel the job)
    "cpu_thread_fraction": 0.65,  # share of CPU threads given to speech recognition / LLM
    "low_priority": True,
    # Glossary: one term per line, optionally "term = preferred translation".
    # Helps Whisper spell names / part numbers and keeps translations consistent.
    "glossary": "",
    "speakers_auto": True,  # identify speakers after transcription
    "speaker_threshold": 0.9,  # clustering distance when the number of speakers is not given (higher = fewer speakers)
    "chinese_script": "simplified",  # simplified | traditional
    "theme": "simple",  # simple | dark | starry | forest | burning | sakura | snowy
    "port": 8765,
}

_lock = threading.Lock()


def ensure_dirs() -> None:
    MEETINGS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_settings() -> dict:
    ensure_dirs()
    s = dict(DEFAULT_SETTINGS)
    if SETTINGS_FILE.exists():
        try:
            s.update(json.loads(SETTINGS_FILE.read_text(encoding="utf-8")))
        except Exception:
            pass
    return s


def save_settings(new: dict) -> dict:
    with _lock:
        s = load_settings()
        for k, v in new.items():
            if k in DEFAULT_SETTINGS:
                s[k] = v
        SETTINGS_FILE.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
        return s


def resolve_model_ref(name: str) -> str:
    """Return a local model folder if one exists in models/, else the name (downloaded on first use)."""
    if not name:
        return name
    p = Path(name)
    if p.is_absolute() and p.exists():
        return str(p)
    local = MODELS_DIR / name
    if local.exists():
        return str(local)
    local2 = MODELS_DIR / f"faster-whisper-{name}"
    if local2.exists():
        return str(local2)
    return name


def add_windows_cuda_dll_dirs() -> None:
    """On Windows, make pip-installed NVIDIA cuBLAS/cuDNN DLLs visible to CTranslate2."""
    if os.name != "nt":
        return
    try:
        import site
        roots = list(site.getsitepackages()) + [site.getusersitepackages()]
        if getattr(sys, "frozen", False):
            roots.append(str(APP_DIR))
            roots.append(getattr(sys, "_MEIPASS", ""))
        for root in roots:
            nv = Path(root) / "nvidia"
            if not nv.exists():
                continue
            for bin_dir in nv.glob("*/bin"):
                try:
                    os.add_dll_directory(str(bin_dir))
                    os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")
                except Exception:
                    pass
    except Exception:
        pass
