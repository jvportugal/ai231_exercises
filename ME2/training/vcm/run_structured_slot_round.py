#!/usr/bin/env python3
"""Train the two predeclared structured-slot candidates on idle GPUs.

Selection uses TEST/tune-OOS only.  HOLDOUT is never loaded by the trainer.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
TRAINER = HERE / "train_temporal_attention.py"
TEACHER = Path("/tmp/rv90_export/vcm.pt")

EXPERIMENTS = (
    {
        "gpu": "0", "tag": "structured_rh_s11251", "seed": "11251",
        "output": "/tmp/vcm_structured_rh_s11251",
        "curriculum": "real-heavy", "kd_weight": "0.0",
    },
    {
        "gpu": "1", "tag": "structured_kd_s11252", "seed": "11252",
        "output": "/tmp/vcm_structured_kd_s11252",
        "curriculum": "balanced", "kd_weight": "0.25",
    },
)


def command(experiment):
    result = [
        sys.executable, "-u", str(TRAINER),
        "--tag", experiment["tag"],
        "--epochs", "30", "--samples-per-epoch", "36000",
        "--batch", "256", "--workers", "12", "--threads", "8",
        "--lr", "3e-4", "--width", "142", "--gru-layers", "2",
        "--aux-words", "96", "--aux-weight", "0.20",
        "--keyword-weight", "2.0", "--structured-slots",
        "--local-slot-weight", "0.75", "--pause-p", "0.25",
        "--pause-min", "0.20", "--pause-max", "0.80",
        "--short-form-boost", "2.0",
        "--curriculum", experiment["curriculum"],
        "--seed", experiment["seed"],
    ]
    if float(experiment["kd_weight"]) > 0.0:
        result.extend([
            "--teacher", str(TEACHER),
            "--kd-weight", experiment["kd_weight"],
            "--temperature", "2.0",
        ])
    return result


def progress(path):
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return "starting"
    for line in reversed(lines):
        if line.startswith(("VALID ", "BEST ", "parameters=", "teacher=")):
            return line[:500]
    return lines[-1][:500] if lines else "starting"


def main():
    if not TEACHER.exists():
        raise FileNotFoundError(TEACHER)
    children = []
    for experiment in EXPERIMENTS:
        output = Path(experiment["output"])
        output.mkdir(parents=True, exist_ok=True)
        log_path = output / "train.log"
        log_handle = log_path.open("w")
        environment = dict(os.environ)
        environment.update({
            "CUDA_VISIBLE_DEVICES": experiment["gpu"],
            "AI231_VCM_OUTPUT": str(output),
            "AI231_VCM_FEATURE_CACHE": "/tmp/vcm_feature_cache",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        })
        process = subprocess.Popen(
            command(experiment), cwd=HERE, env=environment,
            stdout=log_handle, stderr=subprocess.STDOUT,
            start_new_session=True)
        children.append((experiment, process, log_handle, log_path))
        print(f"LAUNCH {experiment['tag']} GPU{experiment['gpu']} "
              f"pid={process.pid} log={log_path}", flush=True)
    try:
        while any(process.poll() is None for _, process, _, _ in children):
            time.sleep(30)
            for experiment, process, _, log_path in children:
                state = "RUN" if process.poll() is None else f"EXIT{process.returncode}"
                print(f"{state} {experiment['tag']}: {progress(log_path)}", flush=True)
    except BaseException:
        for _, process, _, _ in children:
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for _, _, log_handle, _ in children:
            log_handle.close()
    failed = [experiment["tag"] for experiment, process, _, _ in children
              if process.returncode != 0]
    if failed:
        raise RuntimeError(f"failed experiments: {failed}")
    print("ALL STRUCTURED-SLOT RUNS FINISHED", flush=True)


if __name__ == "__main__":
    main()
