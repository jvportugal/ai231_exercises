"""Opt-in exact waveform and decision capture, separate from benchmark JSONL."""
import hashlib
import json
import os
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


class DiagnosticRecorder:
    def __init__(self, directory, model, labels, policy, runtime_root):
        parent = Path(directory).expanduser().resolve()
        parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.directory = parent / (stamp + "-" + str(os.getpid()))
        self.directory.mkdir(exist_ok=False)
        self.index = 0
        model = Path(model).resolve()
        assets = [Path(labels), Path(runtime_root) / "vcm_infer.py",
                  Path(runtime_root) / "vcm_long_window.py"]
        assets.extend(Path(runtime_root) / "vcm_action_runner" / name
                      for name in ("voice.py", "wakeword.py", "runner.py"))
        if isinstance(policy, (str, os.PathLike)):
            assets.append(Path(policy))
        manifest = model / "ensemble.json" if model.is_dir() else model
        assets.append(manifest)
        if manifest.suffix == ".json":
            configuration = json.loads(manifest.read_text())
            if configuration.get("type") == "long_window_ensemble_onnx_v1":
                assets.extend(manifest.parent / configuration[k]
                              for k in ("long_model", "short_model"))
            else:
                assets.extend(manifest.parent / m["model"]
                              for m in configuration.get("members", []))
        assets.append(manifest.parent / "melbank.npy")
        hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in assets if p.is_file()}
        self._write_json(self.directory / "session.json", {
            "model": str(model), "sha256": hashes, "sample_rate": 16000,
            "format": "float32 NPY is exact model input; PCM16 WAV is for listening",
            "started_utc": stamp,
            "argv": sys.argv,
        })
        print("  [diagnostics] exact command audio and decisions: %s" % self.directory,
              flush=True)

    @staticmethod
    def _write_json(path, value):
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def record(self, audio, result, info, *, legacy_e8=False):
        self.index += 1
        stem = self.directory / ("command_%04d" % self.index)
        x = np.asarray(audio, dtype=np.float32).reshape(-1)
        np.save(str(stem) + ".npy", x, allow_pickle=False)
        pcm = np.rint(np.clip(x, -1, 1) * 32767).astype("<i2")
        with wave.open(str(stem) + ".wav", "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(16000)
            stream.writeframes(pcm.tobytes())
        detail = dict(info)
        self._write_json(Path(str(stem) + ".json"), {
            "decision_time_unix": time.time(), "legacy_e8": bool(legacy_e8),
            "result": result, "frontend_and_decoder": detail,
        })
