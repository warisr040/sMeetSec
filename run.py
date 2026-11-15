"""Start sMeetSec: local server on 127.0.0.1 and open the browser."""
from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import time
import webbrowser

from smeetsec import config


def free_port(preferred: int) -> int:
    for p in [preferred] + list(range(preferred + 1, preferred + 20)):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    return preferred


def _arg(name: str, default=None):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def main() -> None:
    # pythonw / windowed builds have no console: send output to the log instead of crashing.
    if sys.stdout is None:
        sys.stdout = open(config.DATA_DIR / "server.log", "a", encoding="utf-8", buffering=1) if config.DATA_DIR.exists() else open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = sys.stdout
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    config.ensure_dirs()
    fixed = _arg("--port")
    port = int(fixed) if fixed else free_port(int(config.load_settings().get("port", 8765)))
    url = f"http://127.0.0.1:{port}/"
    print(f"\n  {config.APP_NAME} {config.APP_VERSION} - {config.APP_FULL_NAME}\n  Open: {url}\n  Close this window to quit.\n")
    if "--no-browser" not in sys.argv:
        threading.Thread(target=lambda: (time.sleep(1.5), webbrowser.open(url)), daemon=True).start()
    import uvicorn
    from smeetsec.app import app
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning", ws_max_size=16 * 1024 * 1024)


if __name__ == "__main__":
    main()
