"""Structured JSON-lines output for the shared VCM benchmark."""

import json
import os
import threading
from datetime import datetime


class BenchmarkLogger:
    """Write exactly one parseable record for each wake and VCM decision."""

    def __init__(self, path=None):
        if path is None:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.expanduser(
                "~/vcm_benchmark/vcm_%s.log" % stamp)
        else:
            path = os.path.abspath(os.path.expanduser(path))
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.path = path
        self._file = open(path, "a", buffering=1, encoding="utf-8")
        self._lock = threading.Lock()

    def _write(self, payload):
        with self._lock:
            self._file.write(json.dumps(payload, separators=(",", ":")) + "\n")
            self._file.flush()

    def wake(self):
        self._write({"event": "wake", "wake": True})

    def decision(self, result):
        accepted = bool(result.get("ok") and result.get("intent"))
        latency = result.get("latency") or {}
        slot = result.get("slot") if accepted else ""
        self._write({
            "intent": result.get("intent") if accepted else "OUT_OF_SCOPE",
            "slot": "" if slot is None else str(slot),
            "infer_ms": round(float(latency.get("classify_ms", 0.0)), 3),
            "audio_ms": round(float(result.get("audio_ms", 0.0)), 3),
        })

    def close(self):
        with self._lock:
            if not self._file.closed:
                self._file.close()

