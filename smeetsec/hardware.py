"""Hardware detection and the 'tier' that decides which models and modes are allowed."""
from __future__ import annotations

import platform
import shutil
import subprocess

import psutil

from . import config


def _cpu_name() -> str:
    name = platform.processor() or ""
    if platform.system() == "Windows":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=8,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if out.stdout.strip():
                name = out.stdout.strip().splitlines()[0]
        except Exception:
            pass
    elif platform.system() == "Linux":
        try:
            for line in open("/proc/cpuinfo", encoding="utf-8"):
                if line.startswith("model name"):
                    name = line.split(":", 1)[1].strip()
                    break
        except Exception:
            pass
    return name or "Unknown CPU"


def _gpus() -> list[dict]:
    """NVIDIA GPUs usable by CTranslate2 (CUDA)."""
    gpus: list[dict] = []
    try:
        import ctranslate2
        count = ctranslate2.get_cuda_device_count()
    except Exception:
        count = 0
    if count <= 0:
        return gpus
    smi = shutil.which("nvidia-smi")
    if smi:
        try:
            out = subprocess.run(
                [smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=8,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            for line in out.stdout.strip().splitlines():
                n, mem = [x.strip() for x in line.rsplit(",", 1)]
                gpus.append({"name": n, "vram_gb": round(float(mem) / 1024, 1)})
        except Exception:
            pass
    if not gpus:
        gpus = [{"name": "NVIDIA GPU", "vram_gb": None} for _ in range(count)]
    return gpus


_cache: dict | None = None


def detect(refresh: bool = False) -> dict:
    global _cache
    if _cache is not None and not refresh:
        return _cache
    ram_gb = round(psutil.virtual_memory().total / 1024**3, 1)
    gpus = _gpus()
    info = {
        "os": f"{platform.system()} {platform.release()}",
        "cpu": _cpu_name(),
        "cpu_cores": psutil.cpu_count(logical=False) or 1,
        "cpu_threads": psutil.cpu_count(logical=True) or 1,
        "ram_gb": ram_gb,
        "gpus": gpus,
    }
    info["tier"] = _tier(info)
    _cache = info
    return info


def _tier(info: dict) -> dict:
    """Pick models and decide whether live mode is supported."""
    gpus = info["gpus"]
    ram = info["ram_gb"]
    vram = max((g["vram_gb"] or 0) for g in gpus) if gpus else 0
    unknown_vram = bool(gpus) and all(g["vram_gb"] is None for g in gpus)

    if gpus and (vram >= 6 or unknown_vram):
        t = {
            "id": "gpu_high", "label": "NVIDIA GPU (6 GB+)", "device": "cuda",
            "batch_model": "large-v3", "batch_compute": "float16",
            "live_model": "large-v3-turbo", "live_compute": "float16",
            "live_supported": True, "partial_interval": 1.0,
        }
    elif gpus:
        t = {
            "id": "gpu_mid", "label": "NVIDIA GPU (<6 GB)", "device": "cuda",
            "batch_model": "large-v3-turbo", "batch_compute": "int8_float16",
            "live_model": "small", "live_compute": "int8_float16",
            "live_supported": True, "partial_interval": 1.5,
        }
    else:
        t = {
            "id": "cpu", "label": "CPU only", "device": "cpu",
            "batch_model": "large-v3-turbo" if ram >= 8 else "small", "batch_compute": "int8",
            "live_model": "small" if info["cpu_cores"] >= 8 else "base", "live_compute": "int8",
            "live_supported": False, "partial_interval": 3.0,
        }
    # Qwen3.5 small models: Apache-2.0 (commercial use OK), 201 languages incl. Thai/Japanese/Chinese.
    t["llm_model"] = "qwen3.5:9b" if ram >= 15 else "qwen3.5:4b"
    t["llm_size"] = "9B" if ram >= 15 else "4B"
    t["notes"] = _notes(t, info)
    return t


def _notes(t: dict, info: dict) -> list[str]:
    n = []
    if t["id"] == "cpu":
        n.append("No NVIDIA GPU found: batch mode recommended. A 1-hour recording takes roughly 30-60 min to transcribe.")
        n.append("Live mode is off by default on CPU. It can be forced in Settings (experimental, small model, high CPU load).")
    if info["ram_gb"] < 15:
        n.append("Less than 16 GB RAM: the smaller translation model (Qwen3.5 4B) is used.")
    return n


def effective(settings: dict) -> dict:
    """Merge user settings over tier defaults."""
    info = detect()
    t = info["tier"]
    device = settings.get("device", "auto")
    if device == "auto":
        device = t["device"]
    if device == "cuda" and not info["gpus"]:
        device = "cpu"
    cpu_compute = "int8"
    batch_model = settings.get("batch_model") or "auto"
    live_model = settings.get("live_model") or "auto"
    llm = settings.get("llm_model") or "auto"
    return {
        "device": device,
        "batch_model": t["batch_model"] if batch_model == "auto" else batch_model,
        "batch_compute": t["batch_compute"] if device == "cuda" and t["device"] == "cuda" else cpu_compute,
        "live_model": t["live_model"] if live_model == "auto" else live_model,
        "live_compute": t["live_compute"] if device == "cuda" and t["device"] == "cuda" else cpu_compute,
        "live_allowed": t["live_supported"] or bool(settings.get("live_force_experimental")),
        "live_experimental": not t["live_supported"],
        "partial_interval": t["partial_interval"],
        "llm_model": t["llm_model"] if llm == "auto" else llm,
        "cpu_threads": int(settings.get("cpu_threads") or 0) or _thread_cap(settings),
    }


def _thread_cap(settings: dict) -> int:
    from .governor import default_thread_cap
    return default_thread_cap(float(settings.get("cpu_thread_fraction") or 0.65))
