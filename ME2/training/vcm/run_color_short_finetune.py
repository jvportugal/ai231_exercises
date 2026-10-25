#!/usr/bin/env python3
"""Low-LR deployment-aligned fine-tune for short COLOR commands."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path


HERE = Path(__file__).resolve().parent
TRAINER = HERE / "train_temporal_attention.py"
EXPERIMENTS = (
    {
        "gpu": "0", "tag": "frontend_color_rh_s13251", "seed": "13251",
        "output": "/tmp/vcm_frontend_color_rh_s13251",
        "curriculum": "real-heavy",
        "base": "/tmp/vcm_color_short_rh_s12251/vcm_attention_color_short_rh_s12251.pt",
    },
    {
        "gpu": "1", "tag": "frontend_color_kd_s13252", "seed": "13252",
        "output": "/tmp/vcm_frontend_color_kd_s13252",
        "curriculum": "balanced",
        "base": "/tmp/vcm_color_short_kd_s12252/vcm_attention_color_short_kd_s12252.pt",
    },
)


def command(experiment):
    return [
        sys.executable, "-u", str(TRAINER),
        "--tag", experiment["tag"], "--base-checkpoint", experiment["base"],
        "--epochs", "8", "--samples-per-epoch", "24000",
        "--batch", "256", "--workers", "12", "--threads", "8",
        "--lr", "2e-5", "--width", "142", "--gru-layers", "2",
        "--aux-words", "96", "--aux-weight", "0.25",
        "--keyword-weight", "4.0", "--structured-slots",
        "--local-slot-weight", "0.75", "--pause-p", "0.25",
        "--pause-min", "0.20", "--pause-max", "0.80",
        "--deployment-frontend-p", "0.75",
        "--short-form-boost", "1.0", "--color-short-boost", "40.0",
        "--curriculum", experiment["curriculum"], "--seed", experiment["seed"],
    ]


def progress(path):
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return "starting"
    for line in reversed(lines):
        if line.startswith(("VALID ", "BEST ", "parameters=", "base_checkpoint=")):
            return line[:500]
    return lines[-1][:500] if lines else "starting"


def main():
    children = []
    for experiment in EXPERIMENTS:
        if not Path(experiment["base"]).exists():
            raise FileNotFoundError(experiment["base"])
        output = Path(experiment["output"])
        output.mkdir(parents=True, exist_ok=True)
        log_path = output / "train.log"
        handle = log_path.open("w")
        environment = dict(os.environ)
        environment.update({
            "CUDA_VISIBLE_DEVICES": experiment["gpu"],
            "AI231_VCM_OUTPUT": str(output),
            "AI231_VCM_FEATURE_CACHE": "/tmp/vcm_feature_cache",
            "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        })
        process = subprocess.Popen(
            command(experiment), cwd=HERE, env=environment,
            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        children.append((experiment, process, handle, log_path))
        print(f"LAUNCH {experiment['tag']} GPU{experiment['gpu']} pid={process.pid}",
              flush=True)
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
        for _, _, handle, _ in children:
            handle.close()
    failures = [experiment["tag"] for experiment, process, _, _ in children
                if process.returncode != 0]
    if failures:
        raise RuntimeError(f"failed experiments: {failures}")
    print("ALL COLOR-SHORT FINE-TUNES FINISHED", flush=True)


if __name__ == "__main__":
    main()
