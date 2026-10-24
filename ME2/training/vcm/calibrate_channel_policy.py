"""Fit and export a threshold-free calibrated VCM deployment policy.

The neural checkpoint is frozen. A scalar temperature and one bias per intent
logit are learned on TEST-as-validation plus tune-OOS, using both clean and one
fixed procedural playback-channel view. Deployment then uses calibrated argmax
over the 19 commands plus the learned REJECT class. Slots are decoded only
among the three values legal for the selected intent.

HOLDOUT is evaluated only after all fitted parameters are fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
from battery import CRNN_N
from eval_ens5c import load_split
from experiment_weight_soup import build_features, labels, load_oos
from finetune_realvoice_safe import build_channel_features


HERE = Path(__file__).resolve().parent
ME2_ROOT = HERE.parents[1]


class Exporter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mel):
        return self.model(mel)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def infer_logits(model, features, device, batch=256):
    logits_a, logits_b = [], []
    with torch.inference_mode():
        for start in range(0, len(features), batch):
            inputs = torch.from_numpy(features[start:start + batch]).to(device)
            out_a, out_b = model(inputs)
            logits_a.append(out_a.cpu().numpy())
            logits_b.append(out_b.cpu().numpy())
    return np.concatenate(logits_a), np.concatenate(logits_b)


def fit_calibration(logits, targets, device, reject_weight=1.0):
    x = torch.from_numpy(logits).float().to(device)
    y = torch.from_numpy(targets).long().to(device)
    log_temperature = nn.Parameter(torch.zeros((), device=device))
    bias = nn.Parameter(torch.zeros(C.N_A, device=device))
    optimizer = torch.optim.LBFGS(
        [log_temperature, bias], lr=0.25, max_iter=250,
        tolerance_grad=1e-9, tolerance_change=1e-10,
        line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad(set_to_none=True)
        temperature = log_temperature.exp().clamp(0.25, 8.0)
        centered_bias = bias - bias.mean()
        calibrated = x / temperature + centered_bias
        class_weights = torch.ones(C.N_A, device=device)
        class_weights[C.N_INTENT] = reject_weight
        loss = (F.cross_entropy(calibrated, y, weight=class_weights)
                + 1e-4 * centered_bias.square().mean())
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        temperature = float(log_temperature.exp().clamp(0.25, 8.0).cpu())
        centered_bias = (bias - bias.mean()).cpu().numpy().astype(np.float64)
    return temperature, centered_bias


def slot_groups():
    result = {}
    for intent, values in C.SLOTTED.items():
        result[C.INTENT2IDX[intent]] = np.asarray([
            C.VALUE2IDX[C.PAIR2TOKEN[(intent, value)]] for value in values
        ], dtype=np.int64)
    return result


def decode(logits_a, logits_b, temperature, bias):
    calibrated = logits_a / temperature + bias[None, :]
    pred_a = calibrated.argmax(axis=1)
    pred_b = np.full(len(pred_a), C.N_VALUE, dtype=np.int64)
    groups = slot_groups()
    for intent_id, valid in groups.items():
        rows = np.flatnonzero(pred_a == intent_id)
        if len(rows):
            local = logits_b[np.ix_(rows, valid)].argmax(axis=1)
            pred_b[rows] = valid[local]
    return pred_a, pred_b


def metrics(frame, target_a, target_b, pred_a, pred_b):
    intent = pred_a == target_a
    slotted = target_b != C.N_VALUE
    command = intent & (~slotted | (pred_b == target_b))
    real = frame["is_synthetic"].fillna(1).astype(int).to_numpy() == 0
    synthetic = ~real
    return {
        "intent": round(float(intent.mean() * 100), 3),
        "e2e": round(float(command.mean() * 100), 3),
        "real_e2e": round(float(command[real].mean() * 100), 3),
        "synthetic_e2e": round(float(command[synthetic].mean() * 100), 3),
    }


def evaluate_view(frame, target_a, target_b, logits_a, logits_b,
                  temperature, bias):
    pred_a, pred_b = decode(logits_a, logits_b, temperature, bias)
    return metrics(frame, target_a, target_b, pred_a, pred_b)


def reject_rate(logits, temperature, bias):
    predictions = (logits / temperature + bias[None, :]).argmax(1)
    return round(float((predictions == C.N_INTENT).mean() * 100), 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1024)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    model = CRNN_N(n_a=C.N_A, n_b=C.N_B, ch=tuple(cfg["ch"]),
                   gru=cfg["gru"], gru_layers=cfg["gru_layers"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    test = load_split("test")
    tune = load_oos(str(Path(C.HF_DIR) / "synthetic_negatives" / "tune_oos-*.parquet"))
    test_a, test_b = labels(test)
    tune_a = np.full(len(tune), C.N_INTENT, dtype=np.int64)

    print("building clean calibration views", flush=True)
    test_clean = build_features(test, 250)
    tune_clean = build_features(tune, 250)
    print("building acoustic calibration views", flush=True)
    test_channel = build_channel_features(test, args.seed + 1000)
    tune_channel = build_channel_features(tune, args.seed + 2000)

    test_clean_a, test_clean_b = infer_logits(model, test_clean, device)
    tune_clean_a, _ = infer_logits(model, tune_clean, device)
    test_channel_a, test_channel_b = infer_logits(model, test_channel, device)
    tune_channel_a, _ = infer_logits(model, tune_channel, device)
    calibration_logits = np.concatenate([
        test_clean_a, test_channel_a, tune_clean_a, tune_channel_a])
    calibration_targets = np.concatenate([test_a, test_a, tune_a, tune_a])
    zero_bias = np.zeros(C.N_A, dtype=np.float64)
    baseline_clean_oos = reject_rate(tune_clean_a, 1.0, zero_bias)
    baseline_channel_oos = reject_rate(tune_channel_a, 1.0, zero_bias)
    candidates = []
    for reject_weight in (1.0, 1.25, 1.5, 2.0, 3.0, 4.0):
        candidate_temperature, candidate_bias = fit_calibration(
            calibration_logits, calibration_targets, device, reject_weight)
        clean_metrics = evaluate_view(
            test, test_a, test_b, test_clean_a, test_clean_b,
            candidate_temperature, candidate_bias)
        channel_metrics = evaluate_view(
            test, test_a, test_b, test_channel_a, test_channel_b,
            candidate_temperature, candidate_bias)
        candidate = {
            "reject_weight": reject_weight,
            "temperature": candidate_temperature,
            "bias": candidate_bias,
            "clean": clean_metrics,
            "channel": channel_metrics,
            "clean_oos_reject": reject_rate(
                tune_clean_a, candidate_temperature, candidate_bias),
            "channel_oos_reject": reject_rate(
                tune_channel_a, candidate_temperature, candidate_bias),
        }
        candidate["selection_score"] = (
            0.4 * clean_metrics["e2e"] + 0.6 * channel_metrics["e2e"])
        candidates.append(candidate)
        printable = {key: value for key, value in candidate.items()
                     if key != "bias"}
        print("CALIBRATION " + json.dumps(printable, sort_keys=True), flush=True)

    # Preserve the raw model's OOS rejection on both clean and channel views;
    # among those safe candidates, maximize the deployment-weighted E2E score.
    feasible = [candidate for candidate in candidates
                if candidate["clean_oos_reject"] >= baseline_clean_oos
                and candidate["channel_oos_reject"] >= baseline_channel_oos
                and candidate["clean"]["e2e"] >= 90.0]
    if not feasible:
        raise RuntimeError("no calibrated candidate preserved baseline OOS rejection")
    selected = max(feasible, key=lambda candidate: candidate["selection_score"])
    temperature = selected["temperature"]
    bias = selected["bias"]
    report = {
        "protocol": ("temperature and class biases fitted on TEST + tune-OOS, "
                     "clean and procedural acoustic views; HOLDOUT evaluation only"),
        "temperature": temperature,
        "selected_reject_weight": selected["reject_weight"],
        "intent_bias": {name: float(value) for name, value in zip(
            C.INTENTS + [C.REJECT], bias)},
        "before": {
            "clean": evaluate_view(test, test_a, test_b, test_clean_a,
                                   test_clean_b, 1.0, zero_bias),
            "channel": evaluate_view(test, test_a, test_b, test_channel_a,
                                     test_channel_b, 1.0, zero_bias),
            "clean_oos_reject": baseline_clean_oos,
            "channel_oos_reject": baseline_channel_oos,
        },
        "after": {
            "clean": evaluate_view(test, test_a, test_b, test_clean_a,
                                   test_clean_b, temperature, bias),
            "channel": evaluate_view(test, test_a, test_b, test_channel_a,
                                     test_channel_b, temperature, bias),
            "clean_oos_reject": selected["clean_oos_reject"],
            "channel_oos_reject": selected["channel_oos_reject"],
        },
        "calibration_candidates": [
            {key: value for key, value in candidate.items() if key != "bias"}
            for candidate in candidates
        ],
    }

    # Sealed holdout is touched only after calibration parameters are final.
    holdout = load_split("holdout")
    holdout_a, holdout_b = labels(holdout)
    holdout_x = build_features(holdout, 250)
    holdout_logits_a, holdout_logits_b = infer_logits(model, holdout_x, device)
    report["holdout_evaluation"] = evaluate_view(
        holdout, holdout_a, holdout_b, holdout_logits_a, holdout_logits_b,
        temperature, bias)
    holdout_pred_a, holdout_pred_b = decode(
        holdout_logits_a, holdout_logits_b, temperature, bias)
    holdout_oos = holdout_a == C.N_INTENT
    holdout_command = ~holdout_oos
    holdout_command_ok = ((holdout_pred_a == holdout_a)
                          & ((holdout_b == C.N_VALUE)
                             | (holdout_pred_b == holdout_b)))
    report["holdout_evaluation"].update({
        "oos_reject": round(float(
            (holdout_pred_a[holdout_oos] == C.N_INTENT).mean() * 100), 3),
        "in_scope_e2e": round(float(
            holdout_command_ok[holdout_command].mean() * 100), 3),
        "n_oos": int(holdout_oos.sum()),
        "n_commands": int(holdout_command.sum()),
    })

    args.destination.mkdir(parents=True, exist_ok=True)
    policy = {
        "model": "Channel-calibrated RV",
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "decoder": {
            "type": "temperature_bias_argmax",
            "temperature": temperature,
            "selected_reject_weight": selected["reject_weight"],
            "intent_bias": report["intent_bias"],
            "reject_rule": "calibrated argmax over 19 commands plus learned REJECT",
            "slot_rule": "conditional argmax over the three values valid for selected intent",
        },
        "calibration_scope": report["protocol"],
    }
    with open(args.destination / "policy.json", "w") as handle:
        json.dump(policy, handle, indent=2)
    with open(args.destination / "calibration_report.json", "w") as handle:
        json.dump(report, handle, indent=2)

    onnx_path = args.destination / "vcm.onnx"
    sample = torch.randn(1, 1, 40, 250, device=device)
    torch.onnx.export(
        Exporter(model), (sample,), onnx_path,
        input_names=["mel"], output_names=["logits_a", "logits_b"],
        dynamic_axes={"mel": {0: "batch"}, "logits_a": {0: "batch"},
                      "logits_b": {0: "batch"}},
        opset_version=13, do_constant_folding=True, dynamo=False,
        external_data=False)
    for name in ("melbank.npy", "labels.json"):
        shutil.copy2(ME2_ROOT / "vcm_model" / name, args.destination / name)

    manifest = {
        "parameters": policy["n_params"],
        "checkpoint": str(args.checkpoint),
        "files": {path.name: sha256(path) for path in sorted(args.destination.iterdir())
                  if path.is_file()},
    }
    with open(args.destination / "export_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
