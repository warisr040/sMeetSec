"""Download all models this PC needs into models\\ (needs internet once).

    python tools/download_models.py
The same downloads are available in the app: Models (top bar) > Download.
A finished models\\ folder can be copied to other PCs (offline install).
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from smeetsec import config, models  # noqa: E402


def main() -> None:
    config.ensure_dirs()
    cancel = threading.Event()
    last = {"m": ""}

    def prog(f, m):
        if m != last["m"]:
            print(f"\r{m} ({f * 100:.0f}%)        ", end="", flush=True)
            last["m"] = m

    for item in models.status():
        if item["present"]:
            print(f"{item['label']}: already installed")
            continue
        print(f"{item['label']}: downloading ({item['size']})")
        if item["id"] == "whisper":
            for n in models._whisper_needed():
                models.download_whisper(n, prog, cancel)
        elif item["id"] == "speakers":
            models.download_speakers(prog, cancel)
        elif item["id"] == "llm" and "--no-llm" not in sys.argv:
            models.download_llm(prog, cancel)
        print()
    print("Models ready in", config.MODELS_DIR)


if __name__ == "__main__":
    main()
