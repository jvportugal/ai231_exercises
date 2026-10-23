#!/usr/bin/env python3
"""Rebuild the shipped RV90 PyTorch/ONNX bundle from retained checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

import config as C
from battery import CRNN_N


HERE = Path(__file__).resolve().parent
ME2_ROOT = HERE.parents[1]
EXPECTED_PARAMS = 3_192_807


class Exporter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mel):
        return self.model(mel)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path,
                        default=HERE / "checkpoints" / "vcm_crnn_X6_babble.pt")
    parser.add_argument("--refined", type=Path,
                        default=HERE / "checkpoints" / "vcm_crnn_rv_kd_ep03.pt")
    parser.add_argument("--refined-weight", type=float, default=0.20)
    parser.add_argument("--destination", type=Path,
                        default=HERE / "artifacts" / "rv90")
    args = parser.parse_args()

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    refined = torch.load(args.refined, map_location="cpu", weights_only=False)
    mixed = {}
    for key, base_value in base["model"].items():
        refined_value = refined["model"][key]
        if base_value.is_floating_point():
            mixed[key] = base_value.mul(1.0 - args.refined_weight).add(
                refined_value, alpha=args.refined_weight)
        else:
            if not torch.equal(base_value, refined_value):
                raise RuntimeError(f"non-floating state differs: {key}")
            mixed[key] = base_value.clone()

    cfg = base["config"]
    model = CRNN_N(n_a=C.N_A, n_b=C.N_B, ch=tuple(cfg["ch"]),
                   gru=cfg["gru"], gru_layers=cfg["gru_layers"])
    model.load_state_dict(mixed)
    model.eval()
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != EXPECTED_PARAMS:
        raise RuntimeError(f"unexpected parameter count: {count}")

    args.destination.mkdir(parents=True, exist_ok=True)
    checkpoint = args.destination / "vcm.pt"
    torch.save({
        "model": mixed,
        "config": cfg,
        "n_params": count,
        "n_frames": 250,
        "source": {
            "base": args.base.name,
            "refined": args.refined.name,
            "refined_weight": args.refined_weight,
        },
    }, checkpoint)

    onnx_path = args.destination / "vcm.onnx"
    sample = torch.randn(1, 1, 40, 250)
    torch.onnx.export(
        Exporter(model), (sample,), onnx_path,
        input_names=["mel"], output_names=["logits_a", "logits_b"],
        dynamic_axes={"mel": {0: "batch"}, "logits_a": {0: "batch"},
                      "logits_b": {0: "batch"}},
        opset_version=13, do_constant_folding=True, dynamo=False,
        external_data=False,
    )

    for name in ("melbank.npy", "labels.json", "policy.json"):
        shutil.copy2(ME2_ROOT / "vcm_model" / name, args.destination / name)

    # Legacy GRU export can mutate internal module state. Reload a fresh model
    # before comparing PyTorch and ONNX outputs.
    parity_model = CRNN_N(n_a=C.N_A, n_b=C.N_B, ch=tuple(cfg["ch"]),
                          gru=cfg["gru"], gru_layers=cfg["gru_layers"])
    parity_model.load_state_dict(mixed)
    parity_model.eval()
    runtime = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    probe = np.random.default_rng(231024).random(
        (1, 1, 40, 250), dtype=np.float32)
    with torch.inference_mode():
        torch_a, torch_b = parity_model(torch.from_numpy(probe))
    onnx_a, onnx_b = runtime.run(None, {runtime.get_inputs()[0].name: probe})
    parity = {
        "intent_max_abs_error": float(np.max(np.abs(torch_a.numpy() - onnx_a))),
        "slot_max_abs_error": float(np.max(np.abs(torch_b.numpy() - onnx_b))),
        "intent_argmax_match": bool(torch_a.argmax(1).item() == onnx_a.argmax(1).item()),
        "slot_argmax_match": bool(torch_b.argmax(1).item() == onnx_b.argmax(1).item()),
    }
    if (parity["intent_max_abs_error"] > 1e-4
            or parity["slot_max_abs_error"] > 1e-4
            or not parity["intent_argmax_match"]
            or not parity["slot_argmax_match"]):
        raise RuntimeError(f"PyTorch/ONNX parity check failed: {parity}")
    manifest = {
        "parameters": count,
        "refined_weight": args.refined_weight,
        "parity": parity,
        "files": {path.name: sha256(path) for path in sorted(args.destination.iterdir())
                  if path.is_file()},
        "shipped_vcm_sha256": sha256(ME2_ROOT / "vcm_model" / "vcm.onnx"),
    }
    with (args.destination / "export_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
