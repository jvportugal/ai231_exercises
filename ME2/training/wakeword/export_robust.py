#!/usr/bin/env python3
"""Export RobustWakeCNN-v2 and its exact mel bank for the Pi package."""
import argparse
import json
import os

import librosa
import numpy as np
import onnx
import onnxruntime as ort
import torch

from train_robust import (HOP, N_FFT, N_FRAMES, N_MELS, SR, WINDOW_SECONDS,
                          RobustWakeCNN)

HERE = os.path.dirname(os.path.abspath(__file__))
ME2_ROOT = os.path.dirname(os.path.dirname(HERE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(HERE, "checkpoints", "destiny_robust_v2.pt"),
    )
    parser.add_argument("--deployment-report", default="deployment_eval_robust_v2.json")
    parser.add_argument("--threshold", type=float, default=0.875)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--require-shipped-equivalence", action="store_true",
                        help="fail unless export is numerically identical to the current asset")
    args = parser.parse_args()

    checkpoint_path = (args.checkpoint if os.path.isabs(args.checkpoint) else
                       os.path.join(HERE, args.checkpoint))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = RobustWakeCNN().eval()
    model.load_state_dict(checkpoint["model_state"])

    os.makedirs(args.destination, exist_ok=True)
    model_path = os.path.join(args.destination, "destiny.onnx")
    dummy = torch.zeros(1, 2, N_MELS, N_FRAMES, dtype=torch.float32)
    torch.onnx.export(
        model, dummy, model_path, input_names=["features"], output_names=["logit"],
        dynamic_axes={"features": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17, do_constant_folding=True, dynamo=False,
        external_data=False)
    onnx.checker.check_model(onnx.load(model_path))

    melbank = librosa.filters.mel(
        sr=SR, n_fft=N_FFT, n_mels=N_MELS, fmin=35.0, fmax=7600.0)
    np.save(os.path.join(args.destination, "melbank.npy"), melbank.astype(np.float32))

    probe = np.random.default_rng(231023).normal(
        size=(16, 2, N_MELS, N_FRAMES)).astype(np.float32)
    with torch.inference_mode():
        torch_logits = model(torch.from_numpy(probe)).numpy()
    runtime = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    exported_logits = runtime.run(
        None, {runtime.get_inputs()[0].name: probe})[0]
    verification = {
        "pytorch_onnx_max_abs_error": float(
            np.max(np.abs(torch_logits - exported_logits))),
        "pytorch_onnx_decisions_match": bool(np.array_equal(
            torch_logits >= 0.0, exported_logits >= 0.0)),
    }
    shipped_path = os.path.join(ME2_ROOT, "wakeword_model", "destiny.onnx")
    if os.path.exists(shipped_path):
        shipped = ort.InferenceSession(
            shipped_path, providers=["CPUExecutionProvider"])
        shipped_logits = shipped.run(
            None, {shipped.get_inputs()[0].name: probe})[0]
        verification.update({
            "exported_shipped_max_abs_error": float(
                np.max(np.abs(exported_logits - shipped_logits))),
            "exported_shipped_decisions_match": bool(np.array_equal(
                exported_logits >= 0.0, shipped_logits >= 0.0)),
        })
    if (verification["pytorch_onnx_max_abs_error"] > 1e-4
            or not verification["pytorch_onnx_decisions_match"]
            or (args.require_shipped_equivalence and
                verification.get("exported_shipped_max_abs_error", 0.0) > 1e-4)
            or (args.require_shipped_equivalence and
                not verification.get("exported_shipped_decisions_match", True))):
        raise RuntimeError("wake export parity check failed: %r" % verification)

    result = checkpoint["metrics"]
    deployment_report_path = (args.deployment_report if os.path.isabs(args.deployment_report)
                              else os.path.join(HERE, args.deployment_report))
    with open(deployment_report_path, encoding="utf-8") as f:
        deployment = json.load(f)
    deployment["checkpoint"] = os.path.basename(str(deployment.get(
        "checkpoint", checkpoint_path)))
    metadata = {
        "wake_word": "Destiny",
        "model": checkpoint["architecture"],
        "parameters": checkpoint["parameters"],
        "sample_rate": SR,
        "window_seconds": WINDOW_SECONDS,
        "frames": N_FRAMES,
        "input_channels": checkpoint["input_channels"],
        "deployment_threshold": args.threshold,
        "threshold_policy": "balanced point from deployment-matched start/center/end scoring",
        "held_out": deployment,
        "training_validation": result,
        "export_verification": verification,
        "training": {
            "source": "MLCommons Multilingual Spoken Words, speaker-disjoint split",
            "user_recordings_required": False,
            "augmentations": [
                "whole-word pitch +/-3.5 semitones",
                "speed 0.78x-1.24x",
                "independent half-word timing warp",
                "gain -22dB to +16dB",
                "close-mic proximity, plosive, saturation and clipping",
                "far-field room impulse responses",
                "procedural fan/air-conditioner hum at 2-32dB SNR",
                "real DCASE fan recordings with machine-ID-disjoint validation",
            ],
            "external_noise": checkpoint.get("external_noise"),
        },
    }
    with open(os.path.join(args.destination, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(json.dumps(metadata, indent=2))
    print("exported", model_path)


if __name__ == "__main__":
    main()
