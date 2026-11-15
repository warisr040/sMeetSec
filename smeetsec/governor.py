"""CPU guard: thread cap, low process priority, and a watchdog that throttles or stops work.

The watchdog samples total system CPU every second and keeps a rolling average
(default 3 s) so short spikes do not trigger it. Heavy work calls `checkpoint()`
between chunks; when usage is over the limit the checkpoint either waits until
usage drops (throttle) or raises `CpuLimitExceeded` (stop).
"""
from __future__ import annotations

import collections
import os
import threading
import time

import psutil


class CpuLimitExceeded(Exception):
    pass


class JobCancelled(Exception):
    pass


class Governor:
    def __init__(self) -> None:
        self.limit = 80.0
        self.action = "throttle"  # throttle | stop
        self.window = 3
        self.max_wait = 120.0  # throttle gives up and stops after waiting this long
        self._samples: collections.deque[float] = collections.deque(maxlen=self.window)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.state = "ok"  # ok | throttling | stopped
        self.events: collections.deque[dict] = collections.deque(maxlen=50)

    # ---------- setup ----------
    def configure(self, limit: float, action: str, window: int = 3) -> None:
        self.limit = max(30.0, min(100.0, float(limit)))
        self.action = action if action in ("throttle", "stop") else "throttle"
        if window != self.window:
            self.window = max(1, int(window))
            with self._lock:
                self._samples = collections.deque(self._samples, maxlen=self.window)

    def start(self) -> None:
        if self._thread:
            return
        psutil.cpu_percent(None)  # prime
        self._thread = threading.Thread(target=self._run, name="cpu-watchdog", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            v = psutil.cpu_percent(interval=1.0)
            with self._lock:
                self._samples.append(v)

    # ---------- readings ----------
    def usage(self) -> float:
        with self._lock:
            if not self._samples:
                return 0.0
            return sum(self._samples) / len(self._samples)

    def over(self) -> bool:
        return self.usage() > self.limit

    def log(self, msg: str) -> None:
        self.events.append({"t": time.time(), "msg": msg, "cpu": round(self.usage(), 1)})

    # ---------- used by workers ----------
    def checkpoint(self, cancel_event: threading.Event | None = None, on_wait=None) -> None:
        """Call between chunks of heavy work."""
        if cancel_event is not None and cancel_event.is_set():
            raise JobCancelled()
        if not self.over():
            self.state = "ok"
            return
        if self.action == "stop":
            self.state = "stopped"
            self.log(f"CPU over {self.limit:.0f}% - job stopped")
            raise CpuLimitExceeded(f"CPU usage {self.usage():.0f}% exceeded the {self.limit:.0f}% limit")
        # throttle: wait until usage drops
        self.state = "throttling"
        self.log(f"CPU over {self.limit:.0f}% - pausing")
        t0 = time.time()
        while self.over():
            if cancel_event is not None and cancel_event.is_set():
                raise JobCancelled()
            if on_wait:
                on_wait(self.usage())
            if time.time() - t0 > self.max_wait:
                self.state = "stopped"
                self.log("CPU stayed high too long - job stopped")
                raise CpuLimitExceeded(
                    f"CPU stayed above {self.limit:.0f}% for {self.max_wait:.0f}s; job stopped"
                )
            time.sleep(0.5)
        self.state = "ok"
        self.log("CPU back under limit - resuming")

    def status(self) -> dict:
        return {
            "cpu": round(self.usage(), 1),
            "limit": self.limit,
            "action": self.action,
            "state": self.state,
            "events": list(self.events)[-10:],
        }


governor = Governor()


def default_thread_cap(fraction: float = 0.65) -> int:
    logical = psutil.cpu_count(logical=True) or 2
    return max(1, int(logical * fraction))


def lower_process_priority() -> None:
    """Run at below-normal priority so the rest of the PC stays responsive."""
    try:
        p = psutil.Process(os.getpid())
        if os.name == "nt":
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            p.nice(10)
    except Exception:
        pass
