#!/usr/bin/env python3
"""Select, calibrate, and export structured-slot models without HOLDOUT.

The two checkpoints are mixed using TEST clean/channel and tune-OOS only.
Selection covers ensemble weight, local/global slot-head weight, REJECT bias,
slot safety gates, and a guarded COLOR rescue based on global slot and word
evidence.  The sealed HOLDOUT is intentionally never loaded here.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import torch

import config as C
from eval_ens5c import load_split
from experiment_weight_soup import build_features, labels, load_oos
from finetune_source_curriculum import build_channel_features, cached_features
from battery import build_deployment_features
from train_temporal_attention import TemporalAttentionVCM


HERE = Path(__file__).resolve().parent
ME2_ROOT = HERE.parents[1]
PARAMETER_CAP = 3_192_807


def softmax(values, axis=1):
    values = np.asarray(values, dtype=np.float64)
    shifted = values - values.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def load_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    if not cfg.get("structured_slots", False):
        raise ValueError(f"{path} is not a structured-slot checkpoint")
    vocabulary = checkpoint.get("vocabulary", [])
    model = TemporalAttentionVCM(
        width=cfg["width"], gru_layers=cfg["gru_layers"],
        auxiliary_words=len(vocabulary), structured_slots=True).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, vocabulary, checkpoint


def infer(model, features, device, batch=256):
    collected = [[], [], [], []]
    with torch.inference_mode():
        for start in range(0, len(features), batch):
            inputs = torch.from_numpy(
                np.asarray(features[start:start + batch])).to(device)
            outputs = model(inputs)
            for destination, value in zip(collected, outputs):
                destination.append(value.cpu().numpy())
    logits = [np.concatenate(values) for values in collected]
    local = logits[2].reshape(len(logits[2]), len(C.SLOTTED), 3)
    local_probs = softmax(local, axis=2).reshape(logits[2].shape)
    return [softmax(logits[0]), softmax(logits[1]),
            local_probs, sigmoid(logits[3])]


def mix(first, second, weight):
    return [weight * a + (1.0 - weight) * b
            for a, b in zip(first, second)]


def apply_reject_bias(pa, bias):
    adjusted = np.asarray(pa, dtype=np.float64).copy()
    adjusted[:, C.N_INTENT] *= np.exp(bias)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def decode(pa, global_slots, local_slots, local_weight):
    pred_a = pa.argmax(1)
    pred_b = np.full(len(pa), C.N_VALUE, dtype=np.int64)
    slot_conf = np.zeros(len(pa), dtype=np.float64)
    slot_margin = np.zeros(len(pa), dtype=np.float64)
    global_conf = np.zeros(len(pa), dtype=np.float64)
    global_margin = np.zeros(len(pa), dtype=np.float64)
    for group, (intent, values) in enumerate(C.SLOTTED.items()):
        intent_id = C.INTENT2IDX[intent]
        rows = np.flatnonzero(pred_a == intent_id)
        if not len(rows):
            continue
        valid = np.arange(group * 3, group * 3 + 3)
        global_local = global_slots[np.ix_(rows, valid)]
        global_local /= global_local.sum(axis=1, keepdims=True)
        local = local_slots[np.ix_(rows, valid)]
        probabilities = ((1.0 - local_weight) * global_local
                         + local_weight * local)
        choice = probabilities.argmax(1)
        pred_b[rows] = valid[choice]
        ordered = np.sort(probabilities, axis=1)
        slot_conf[rows] = probabilities[np.arange(len(rows)), choice]
        slot_margin[rows] = ordered[:, -1] - ordered[:, -2]
        chosen_global = global_slots[rows, pred_b[rows]]
        competitors = global_slots[rows].copy()
        competitors[np.arange(len(rows)), pred_b[rows]] = -1.0
        global_conf[rows] = chosen_global
        global_margin[rows] = chosen_global - competitors.max(axis=1)
    return (pred_a, pred_b, slot_conf, slot_margin,
            global_conf, global_margin)


def score_view(frame, target_a, target_b, decoded):
    pred_a, pred_b = decoded[:2]
    intent_ok = pred_a == target_a
    command_ok = intent_ok & ((target_b == C.N_VALUE) | (pred_b == target_b))
    real = frame["is_synthetic"].fillna(1).astype(int).to_numpy() == 0
    slotted_and_intent_right = (target_b != C.N_VALUE) & intent_ok
    result = {
        "intent": float(intent_ok.mean() * 100.0),
        "e2e": float(command_ok.mean() * 100.0),
        "real_e2e": float(command_ok[real].mean() * 100.0),
        "synthetic_e2e": float(command_ok[~real].mean() * 100.0),
        "slot_exact_when_intent_right": float(
            (pred_b[slotted_and_intent_right]
             == target_b[slotted_and_intent_right]).mean() * 100.0),
    }
    transcripts = frame["transcript"].fillna("").astype(str)
    short_color = ((target_a == C.INTENT2IDX["COLOR"])
                   & (transcripts.str.findall(r"[A-Za-z]+").str.len().to_numpy() <= 3))
    if short_color.any():
        result["short_color_e2e"] = float(
            command_ok[short_color].mean() * 100.0)
        result["short_color_n"] = int(short_color.sum())
    return result


def oos_reject(pa):
    return float((pa.argmax(1) == C.N_INTENT).mean() * 100.0)


def calibrate_slot_gates(target_a, target_b, views, local_weight):
    result = {}
    for intent, values in C.SLOTTED.items():
        intent_id = C.INTENT2IDX[intent]
        correct_conf, correct_margin = [], []
        for pa, global_slots, local_slots in views:
            decoded = decode(pa, global_slots, local_slots, local_weight)
            pred_a, pred_b, confidence, margin = decoded[:4]
            rows = ((target_a == intent_id) & (pred_a == target_a)
                    & (pred_b == target_b))
            correct_conf.extend(confidence[rows].tolist())
            correct_margin.extend(margin[rows].tolist())
        if correct_conf:
            # Each independent one-percent lower-tail gate is deliberately
            # conservative; it redirects ambiguity to a repeat rather than
            # optimizing against the sealed physical benchmark.
            result[intent] = {
                "min_confidence": round(float(np.quantile(correct_conf, 0.01)), 6),
                "min_margin": round(float(np.quantile(correct_margin, 0.01)), 6),
            }
    return result


def color_joint_rescue(pa, global_slots, words, vocabulary, config):
    pred = pa.argmax(1)
    rescued = pred.copy()
    word_index = {word: index for index, word in enumerate(vocabulary)}
    color_id = C.INTENT2IDX["COLOR"]
    color_values = np.arange(C.VALUE2IDX["COLOR_RED"],
                             C.VALUE2IDX["COLOR_GREEN"] + 1)
    family_words = ("color", "colour", "light", "lights")
    for row in range(len(pa)):
        if pred[row] in (color_id, C.N_INTENT):
            continue
        local = global_slots[row, color_values]
        choice = int(local.argmax())
        slot_index = int(color_values[choice])
        confidence = float(global_slots[row, slot_index])
        competitors = global_slots[row].copy()
        competitors[slot_index] = -1.0
        margin = confidence - float(competitors.max())
        top_conf = float(pa[row, pred[row]])
        intent_conf = float(pa[row, color_id])
        rank = int(np.sum(pa[row] > intent_conf)) + 1
        word_ok = any(name in word_index and
                      words[row, word_index[name]] >= config["min_word_conf"]
                      for name in family_words)
        if (confidence >= config["min_global_slot_conf"]
                and margin >= config["min_global_slot_margin"]
                and intent_conf >= config["min_intent_conf"]
                and intent_conf >= config["min_intent_ratio"] * top_conf
                and rank <= config["max_intent_rank"]
                and pa[row, C.N_INTENT] <= config["max_reject_conf"]
                and word_ok):
            rescued[row] = color_id
    return rescued


def select_joint_config(target_a, target_b, views, vocabulary, local_weight):
    baseline = []
    for pa, global_slots, local_slots, _ in views:
        pred_a, pred_b = decode(
            pa, global_slots, local_slots, local_weight)[:2]
        baseline.append({
            "intent": float((pred_a == target_a).mean()),
            "e2e": float(((pred_a == target_a)
                          & ((target_b == C.N_VALUE) | (pred_b == target_b))).mean()),
        })
    best = None
    for slot_conf, slot_margin, ratio, word_conf in itertools.product(
            (0.30, 0.40, 0.50, 0.60), (0.05, 0.10, 0.20, 0.30),
            (0.50, 0.65, 0.80), (0.50, 0.60, 0.70, 0.80)):
        config = {
            "enabled": True, "allowed_intents": ["COLOR"],
            "allow_from_reject": False,
            "min_global_slot_conf": slot_conf,
            "min_global_slot_margin": slot_margin,
            "min_intent_conf": 0.05, "min_intent_ratio": ratio,
            "max_intent_rank": 3, "max_reject_conf": 0.35,
            "min_word_conf": word_conf,
            "require_word_support": True,
            "word_requirement_any": {
                "COLOR_RED": ["color", "colour", "light", "lights"],
                "COLOR_BLUE": ["color", "colour", "light", "lights"],
                "COLOR_GREEN": ["color", "colour", "light", "lights"],
            },
        }
        results, changes = [], 0
        for pa, global_slots, local_slots, words in views:
            predictions = color_joint_rescue(
                pa, global_slots, words, vocabulary, config)
            pred_a, pred_b = decode(
                pa, global_slots, local_slots, local_weight)[:2]
            changed = predictions != pred_a
            if changed.any():
                group = list(C.SLOTTED).index("COLOR")
                valid = np.arange(group * 3, group * 3 + 3)
                rows = np.flatnonzero(changed)
                global_local = global_slots[np.ix_(rows, valid)]
                global_local /= global_local.sum(axis=1, keepdims=True)
                local = local_slots[np.ix_(rows, valid)]
                probabilities = ((1.0 - local_weight) * global_local
                                 + local_weight * local)
                pred_b[rows] = valid[probabilities.argmax(1)]
            intent_accuracy = float((predictions == target_a).mean())
            e2e = float(((predictions == target_a)
                        & ((target_b == C.N_VALUE) | (pred_b == target_b))).mean())
            results.append({"intent": intent_accuracy, "e2e": e2e})
            changes += int(changed.sum())
        # Never sacrifice intent or end-to-end accuracy in either view.
        if any(result[metric] + 1e-12 < before[metric]
               for result, before in zip(results, baseline)
               for metric in ("intent", "e2e")):
            continue
        gain = 0.4 * (results[0]["e2e"] - baseline[0]["e2e"]) + 0.6 * (
            results[1]["e2e"] - baseline[1]["e2e"])
        intent_gain = 0.4 * (
            results[0]["intent"] - baseline[0]["intent"]) + 0.6 * (
                results[1]["intent"] - baseline[1]["intent"])
        candidate = (gain, intent_gain, -changes, config, results, changes)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    if best is None or (best[0] <= 0.0 and best[1] <= 0.0):
        return {"enabled": False}, {"gain": 0.0, "changes": 0}
    return best[3], {
        "e2e_gain": best[0], "intent_gain": best[1], "changes": best[5],
        "before": baseline, "after": best[4],
    }


def export_model(model, destination):
    sample = torch.randn(1, 1, 40, 250, device=next(model.parameters()).device)
    torch.onnx.export(
        model, (sample,), destination,
        input_names=["mel"],
        output_names=["logits_a", "logits_b", "local_slot_logits", "word_logits"],
        dynamic_axes={name: {0: "batch"} for name in
                      ("mel", "logits_a", "logits_b",
                       "local_slot_logits", "word_logits")},
        opset_version=17, do_constant_folding=True, dynamo=False,
        external_data=False)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_a, vocabulary_a, checkpoint_a = load_model(args.checkpoint_a, device)
    model_b, vocabulary_b, checkpoint_b = load_model(args.checkpoint_b, device)
    if vocabulary_a != vocabulary_b:
        raise ValueError("ensemble members use different word vocabularies")
    vocabulary = vocabulary_a

    test = load_split("test")
    target_a, target_b = labels(test)
    tune = load_oos(str(Path(C.HF_DIR) / "synthetic_negatives" /
                        "tune_oos-*.parquet"))
    cache = Path("/tmp/vcm_feature_cache")
    test_clean = cached_features(cache / "test_clean_250.npy",
                                 lambda: build_features(test, 250))
    test_deployment = cached_features(
        cache / "test_deployment_250.npy",
        lambda: build_deployment_features(test, 250))
    tune_clean = cached_features(cache / "tune_oos_clean_250.npy",
                                 lambda: build_features(tune, 250))
    test_channel = cached_features(
        cache / f"test_channel_seed{args.seed}_250.npy",
        lambda: build_channel_features(test, 250, args.seed))
    tune_channel = cached_features(
        cache / f"tune_oos_channel_seed{args.seed + 1000}_250.npy",
        lambda: build_channel_features(tune, 250, args.seed + 1000))

    features = (test_clean, test_deployment, test_channel,
                tune_clean, tune_channel)
    outputs_a = [infer(model_a, value, device) for value in features]
    outputs_b = [infer(model_b, value, device) for value in features]
    candidates = []
    for weight, local_weight, reject_bias in itertools.product(
            (0.25, 0.50, 0.75), (0.25, 0.50, 0.75, 1.0),
            (-0.20, -0.10, 0.0, 0.10, 0.20)):
        views = [mix(a, b, weight) for a, b in zip(outputs_a, outputs_b)]
        for view in views:
            view[0] = apply_reject_bias(view[0], reject_bias)
        clean = score_view(test, target_a, target_b,
                           decode(*views[0][:3], local_weight))
        deployment = score_view(test, target_a, target_b,
                             decode(*views[1][:3], local_weight))
        channel = score_view(test, target_a, target_b,
                             decode(*views[2][:3], local_weight))
        clean_oos = oos_reject(views[3][0])
        channel_oos = oos_reject(views[4][0])
        objective = (0.20 * clean["e2e"] + 0.35 * deployment["e2e"]
                     + 0.35 * channel["e2e"]
                     + 0.05 * clean_oos + 0.05 * channel_oos)
        candidates.append({
            "weight_a": weight, "local_slot_weight": local_weight,
            "reject_bias": reject_bias, "clean": clean,
            "deployment": deployment, "channel": channel,
            "clean_oos_reject": clean_oos,
            "channel_oos_reject": channel_oos, "objective": objective,
            "views": views,
        })
    feasible = [candidate for candidate in candidates
                if candidate["clean"]["e2e"] >= 90.0
                and candidate["deployment"]["e2e"] >= 88.0
                and candidate["clean_oos_reject"] >= 65.0
                and candidate["channel_oos_reject"] >= 65.0]
    if not feasible:
        raise RuntimeError("no safe structured ensemble candidate")
    selected = max(feasible, key=lambda item: item["objective"])
    views = selected.pop("views")
    slot_gates = calibrate_slot_gates(
        target_a, target_b,
        [(views[0][0], views[0][1], views[0][2]),
         (views[1][0], views[1][1], views[1][2]),
         (views[2][0], views[2][1], views[2][2])],
        selected["local_slot_weight"])
    joint_config, joint_report = select_joint_config(
        target_a, target_b, (views[1], views[2]), vocabulary,
        selected["local_slot_weight"])

    args.destination.mkdir(parents=True, exist_ok=True)
    model_a_path = args.destination / "structured_a.onnx"
    model_b_path = args.destination / "structured_b.onnx"
    export_model(model_a, model_a_path)
    export_model(model_b, model_b_path)
    weight_a = selected["weight_a"]
    total_parameters = int(checkpoint_a["n_params"] + checkpoint_b["n_params"])
    if total_parameters > PARAMETER_CAP:
        raise RuntimeError(
            f"structured ensemble has {total_parameters:,} parameters; "
            f"cap is {PARAMETER_CAP:,}")
    ensemble = {
        "type": "probability_ensemble_onnx_v1",
        "selection": "TEST clean/deployment/channel plus tune-OOS; HOLDOUT excluded",
        "outputs": ["logits_a", "logits_b", "local_slot_logits", "word_logits"],
        "output_activations": ["softmax", "softmax", "grouped_softmax_3", "sigmoid"],
        "members": [
            {"name": "structured_a", "model": model_a_path.name,
             "weight": weight_a, "source_checkpoint": str(args.checkpoint_a)},
            {"name": "structured_b", "model": model_b_path.name,
             "weight": 1.0 - weight_a, "source_checkpoint": str(args.checkpoint_b)},
        ],
        "total_parameters": total_parameters,
        "parameter_cap": PARAMETER_CAP,
    }
    policy = {
        "model": "structured-slot attention ensemble",
        "decoder": {
            "type": "joint_intent_slot_v1", "temperature": 1.0,
            "intent_bias": {"REJECT": selected["reject_bias"]},
            "local_slot_weight": selected["local_slot_weight"],
            "slot_acceptance": {"per_intent": slot_gates},
            "joint_slot_rescue": joint_config,
        },
        "selection": ("Frozen on TEST clean/deployment/channel and tune-OOS; "
                      "HOLDOUT excluded"),
    }
    labels_path = ME2_ROOT / "vcm_model" / "labels.json"
    labels_payload = json.loads(labels_path.read_text())
    labels_payload["words"] = vocabulary
    (args.destination / "ensemble.json").write_text(json.dumps(ensemble, indent=2))
    (args.destination / "policy.json").write_text(json.dumps(policy, indent=2))
    (args.destination / "labels.json").write_text(json.dumps(labels_payload, indent=2))
    shutil.copy2(ME2_ROOT / "vcm_model" / "melbank.npy",
                 args.destination / "melbank.npy")
    report_selected = dict(selected)
    report = {
        "protocol": ("TEST clean/deployment/channel + tune-OOS only; "
                     "HOLDOUT never loaded"),
        "selected": report_selected, "slot_gates": slot_gates,
        "joint_rescue": joint_report,
    }
    (args.destination / "selection_report.json").write_text(
        json.dumps(report, indent=2))
    checksums = {path.name: sha256(path) for path in args.destination.iterdir()
                 if path.is_file() and path.name != "checksums.json"}
    (args.destination / "checksums.json").write_text(
        json.dumps(checksums, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
