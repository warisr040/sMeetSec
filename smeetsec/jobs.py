"""Background job queue. One worker = one heavy job at a time, which keeps CPU load predictable."""
from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
import uuid

from .governor import CpuLimitExceeded, JobCancelled

log = logging.getLogger("smeetsec.jobs")


class Job:
    def __init__(self, kind: str, meeting_id: str, fn, title: str):
        self.id = uuid.uuid4().hex[:8]
        self.kind = kind
        self.meeting_id = meeting_id
        self.title = title
        self.fn = fn
        self.status = "queued"  # queued | waiting | running | done | error | cancelled | stopped
        self.progress = 0.0
        self.message = "Waiting in queue"
        self.error = ""
        self.created = time.time()
        self.finished = 0.0
        self.cancel = threading.Event()

    def update(self, fraction: float, message: str) -> None:
        self.progress = max(self.progress, min(1.0, fraction))
        self.message = message

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "meeting_id": self.meeting_id, "title": self.title,
            "status": self.status, "progress": round(self.progress, 3), "message": self.message,
            "error": self.error, "created": self.created, "finished": self.finished,
        }


class JobQueue:
    def __init__(self) -> None:
        self._q: queue.Queue[Job] = queue.Queue()
        self.jobs: dict[str, Job] = {}
        self.live_active = threading.Event()  # batch jobs wait while live transcription runs
        self._t = threading.Thread(target=self._worker, name="job-worker", daemon=True)
        self._t.start()

    def submit(self, kind: str, meeting_id: str, fn, title: str) -> Job:
        j = Job(kind, meeting_id, fn, title)
        self.jobs[j.id] = j
        self._q.put(j)
        return j

    def cancel(self, job_id: str) -> bool:
        j = self.jobs.get(job_id)
        if not j or j.status in ("done", "error", "cancelled", "stopped"):
            return False
        j.cancel.set()
        if j.status in ("queued", "waiting"):
            j.status = "cancelled"
            j.message = "Cancelled"
        return True

    def list(self) -> list[dict]:
        cutoff = time.time() - 6 * 3600
        return [j.to_dict() for j in sorted(self.jobs.values(), key=lambda x: x.created, reverse=True)
                if not j.finished or j.finished > cutoff]

    def active_for(self, meeting_id: str) -> list[dict]:
        return [j.to_dict() for j in self.jobs.values()
                if j.meeting_id == meeting_id and j.status in ("queued", "waiting", "running")]

    def _worker(self) -> None:
        while True:
            j = self._q.get()
            if j.cancel.is_set():
                continue
            while self.live_active.is_set() and not j.cancel.is_set():
                j.status = "waiting"
                j.message = "Waiting: live transcription is running"
                time.sleep(1)
            if j.cancel.is_set():
                j.status, j.message, j.finished = "cancelled", "Cancelled", time.time()
                continue
            j.status = "running"
            j.message = "Starting"
            try:
                j.fn(j)
                j.status = "done"
                j.progress = 1.0
                j.message = "Done"
            except JobCancelled:
                j.status, j.message = "cancelled", "Cancelled"
            except CpuLimitExceeded as e:
                j.status, j.error, j.message = "stopped", str(e), "Stopped by CPU guard"
            except Exception as e:
                log.error("job %s failed: %s", j.id, traceback.format_exc())
                j.status, j.error, j.message = "error", str(e), "Failed"
            finally:
                j.finished = time.time()
                self._after(j)

    def _after(self, j: Job) -> None:
        # Reflect failure in the meeting status so the UI shows it.
        if j.status in ("error", "stopped", "cancelled"):
            try:
                from . import meetings
                m = meetings.get(j.meeting_id)
                if m.get("status") in ("transcribing", "reporting"):
                    has_t = bool(meetings.current_transcript(j.meeting_id))
                    meetings.update(j.meeting_id,
                                    status="transcribed" if has_t else "error",
                                    error=j.error or j.message)
            except Exception:
                pass


jobs = JobQueue()
