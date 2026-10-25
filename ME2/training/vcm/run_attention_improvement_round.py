"""Run the fixed attention improvement round on four otherwise idle GPUs.

The configurations are chosen before HOLDOUT evaluation.  Each child writes
its own checkpoints/report under /tmp and never loads HOLDOUT.
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

EXPERIMENTS = [
    {
        "gpu": "1", "tag": "attention_kd128_s6101",
        "output": "/tmp/vcm_attention_kd128_s6101",
        "width": "128", "layers": "2", "seed": "6101",
        "aux_words": "0", "aux_weight": "0",
    },
    {
        "gpu": "3", "tag": "attention_kd128_s6102",
        "output": "/tmp/vcm_attention_kd128_s6102",
        "width": "128", "layers": "2", "seed": "6102",
        "aux_words": "0", "aux_weight": "0",
    },
    {
        "gpu": "6", "tag": "attention_kd144_s6201",
        "output": "/tmp/vcm_attention_kd144_s6201",
        "width": "144", "layers": "2", "seed": "6201",
        "aux_words": "0", "aux_weight": "0",
    },
    {
        "gpu": "7", "tag": "attention_kd128_words_s6103",
        "output": "/tmp/vcm_attention_kd128_words_s6103",
        "width": "128", "layers": "2", "seed": "6103",
        "aux_words": "96", "aux_weight": "0.15",
    },
]


def command(experiment):
    return [
        sys.executable, "-u", str(TRAINER),
        "--tag", experiment["tag"],
        "--epochs", "30",
        "--samples-per-epoch", "36000",
        "--batch", "256",
        "--workers", "12",
        "--threads", "8",
        "--lr", "3e-4",
        "--width", experiment["width"],
        "--gru-layers", experiment["layers"],
        "--aux-words", experiment["aux_words"],
        "--aux-weight", experiment["aux_weight"],
        "--teacher", str(TEACHER),
        "--kd-weight", "0.35",
        "--temperature", "2.0",
        "--seed", experiment["seed"],
    ]


def last_progress(log_path):
    try:
        lines = log_path.read_text().splitlines()
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
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
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
                status = "RUN" if process.poll() is None else f"EXIT{process.returncode}"
                print(f"{status} {experiment['tag']}: {last_progress(log_path)}",
                      flush=True)
    except BaseException:
        for _, process, _, _ in children:
            if process.poll() is None:
                process.terminate()
        raise
    finally:
        for _, _, log_handle, _ in children:
            log_handle.close()

    failures = [experiment["tag"] for experiment, process, _, _ in children
                if process.returncode != 0]
    if failures:
        raise RuntimeError(f"failed experiments: {failures}")
    print("ALL ATTENTION IMPROVEMENT RUNS FINISHED", flush=True)


if __name__ == "__main__":
    main()
