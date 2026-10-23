"""Conservative source-balanced fine-tuning of the X6 VCM.

The model architecture is unchanged.  Training uses only the existing local
VCM training files.  Every epoch is checkpointed so an interrupted terminal
cannot discard the run, and model selection uses TEST-as-validation plus
tune-OOS only.  Historical HOLDOUT and final OOS-test are not loaded here.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
from battery import BatDataset, CRNN_N
from eval_ens5c import load_split
from experiment_weight_soup import build_features, command_metrics, labels, load_oos, predict
from train_centered import load_audio_split, load_noise_bank


def read_sources() -> list[str]:
    files = sorted(glob.glob(os.path.join(C.DATA_DIR, "train-*.parquet")))
    frame = pd.concat([pd.read_parquet(path, columns=["source"]) for path in files],
                      ignore_index=True)
    return frame["source"].fillna("unknown").astype(str).tolist()


def source_balanced_weights(rows, sources, real_voice_fraction,
                            other_real_fraction, reject_fraction):
    labels_a = np.asarray([row[1] for row in rows])
    is_real = np.asarray([row[3] for row in rows], dtype=bool)
    sources = np.asarray(sources, dtype=object)
    reject = labels_a == C.N_INTENT
    real_voice = is_real & ~reject & (sources == "real_voice")
    other_real = is_real & ~reject & ~real_voice
    synthetic = ~is_real & ~reject
    synthetic_fraction = 1.0 - real_voice_fraction - other_real_fraction - reject_fraction
    if synthetic_fraction <= 0:
        raise ValueError("sampling fractions leave no mass for synthetic commands")

    weights = np.zeros(len(rows), dtype=np.float64)
    groups = ((real_voice, real_voice_fraction),
              (other_real, other_real_fraction),
              (synthetic, synthetic_fraction))
    report = {}
    for name, (mask, fraction) in zip(("real_voice", "other_real", "synthetic"), groups):
        present = sorted(set(labels_a[mask].tolist()))
        if not present:
            raise RuntimeError(f"empty sampling group: {name}")
        for target in present:
            class_mask = mask & (labels_a == target)
            weights[class_mask] = fraction / len(present) / class_mask.sum()
        report[name] = {"rows": int(mask.sum()), "mass": float(weights[mask].sum())}
    weights[reject] = reject_fraction / reject.sum()
    report["reject"] = {"rows": int(reject.sum()), "mass": float(weights[reject].sum())}
    weights /= weights.sum()
    return weights, report


def validation_metrics(model, frame, features, target_a, target_b, tune_features):
    pred_a, pred_b = predict(model, features, batch=64)
    pred_oos, _ = predict(model, tune_features, batch=64)
    metrics = command_metrics(frame, target_a, target_b, pred_a, pred_b)
    metrics["tune_oos_reject"] = round(float((pred_oos == C.N_INTENT).mean() * 100), 3)
    return metrics


def selection_score(metrics):
    aggregate_penalty = max(0.0, 90.0 - metrics["overall_e2e"]) * 25.0
    return (metrics["real_e2e"] + 0.05 * metrics["overall_e2e"]
            + 0.05 * metrics["tune_oos_reject"] - aggregate_penalty)


def distillation_loss(student_logits, teacher_logits, temperature):
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature * temperature)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="vcm_crnn_X6_babble.pt")
    parser.add_argument("--tag", default="rv_safe")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--samples-per-epoch", type=int, default=12000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--cnn-lr-scale", type=float, default=0.1)
    parser.add_argument("--real-voice-fraction", type=float, default=0.25)
    parser.add_argument("--other-real-fraction", type=float, default=0.25)
    parser.add_argument("--reject-fraction", type=float, default=0.15)
    parser.add_argument("--babble-p", type=float, default=0.15)
    parser.add_argument("--speed-p", type=float, default=0.25)
    parser.add_argument("--kd-weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=811)
    args = parser.parse_args()

    torch.set_num_threads(min(args.threads, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")

    base_path = args.base if os.path.isabs(args.base) else os.path.join(C.OUT_DIR, args.base)
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    cfg = base["config"]
    model = CRNN_N(n_a=C.N_A, n_b=C.N_B, ch=tuple(cfg["ch"]),
                   gru=cfg["gru"], gru_layers=cfg["gru_layers"]).to(device)
    model.load_state_dict(base["model"])
    teacher = None
    if args.kd_weight > 0:
        teacher = CRNN_N(n_a=C.N_A, n_b=C.N_B, ch=tuple(cfg["ch"]),
                         gru=cfg["gru"], gru_layers=cfg["gru_layers"]).to(device)
        teacher.load_state_dict(base["model"])
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    n_params = sum(parameter.numel() for parameter in model.parameters())
    if n_params != 3_192_807:
        raise RuntimeError(f"unexpected parameter count: {n_params}")

    rows = load_audio_split(["train", "fil", "neg_train", "neg_noise"])
    train_sources = read_sources()
    if len(train_sources) != 10_733:
        raise RuntimeError(f"unexpected train source count: {len(train_sources)}")
    sources = train_sources + ["supplemental_fil"] * 14_120
    sources += ["negative"] * 1_000 + ["negative_noise"] * 88
    if len(sources) != len(rows):
        raise RuntimeError(f"source/row mismatch: {len(sources)} != {len(rows)}")
    weights, groups = source_balanced_weights(
        rows, sources, args.real_voice_fraction,
        args.other_real_fraction, args.reject_fraction)
    print(json.dumps({"groups": groups, "n_params": n_params}, sort_keys=True), flush=True)

    noise = load_noise_bank()
    dataset = BatDataset(
        rows, noise, aug=True, seed=args.seed, n_frames=250,
        snr_min=12.0, snr_max=28.0, gain_db=4.0, shift=0.10,
        babble_p=args.babble_p, speed_p=args.speed_p,
        speed_lo=0.94, speed_hi=1.06)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.from_numpy(weights), num_samples=args.samples_per_epoch,
        replacement=True, generator=generator)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch, sampler=sampler,
        num_workers=0, pin_memory=False)

    test_frame = load_split("test")
    tune_frame = load_oos(os.path.join(C.HF_DIR, "synthetic_negatives", "tune_oos-*.parquet"))
    print("building fixed selection features", flush=True)
    test_features = build_features(test_frame, 250)
    tune_features = build_features(tune_frame, 250)
    test_a, test_b = labels(test_frame)

    cnn = list(model.cnn.parameters())
    other = [parameter for name, parameter in model.named_parameters()
             if not name.startswith("cnn.")]
    optimizer = torch.optim.AdamW([
        {"params": cnn, "lr": args.lr * args.cnn_lr_scale},
        {"params": other, "lr": args.lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    model.eval()
    initial = validation_metrics(model, test_frame, test_features, test_a, test_b,
                                 tune_features)
    initial.update({"epoch": 0, "selection_score": selection_score(initial)})
    history.append(initial)
    best_metrics = initial
    best_state = copy.deepcopy(model.state_dict())
    print("VALID " + json.dumps(initial, sort_keys=True), flush=True)

    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        # Preserve the well-estimated X6 BatchNorm running statistics.
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        total_loss = 0.0
        seen = 0
        for features, target_a, target_b, _ in loader:
            features = features.to(device)
            target_a = target_a.to(device)
            target_b = target_b.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits_a, logits_b = model(features)
            supervised = (F.cross_entropy(logits_a, target_a)
                          + F.cross_entropy(logits_b, target_b))
            if teacher is None:
                loss = supervised
            else:
                with torch.inference_mode():
                    teacher_a, teacher_b = teacher(features)
                distilled = (
                    distillation_loss(logits_a, teacher_a, args.temperature)
                    + distillation_loss(logits_b, teacher_b, args.temperature)
                )
                loss = ((1.0 - args.kd_weight) * supervised
                        + args.kd_weight * distilled)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(features)
            seen += len(features)
        scheduler.step()
        model.eval()
        metrics = validation_metrics(model, test_frame, test_features, test_a, test_b,
                                     tune_features)
        metrics.update({
            "epoch": epoch,
            "train_loss": round(total_loss / seen, 6),
            "elapsed_s": round(time.time() - started, 1),
            "selection_score": selection_score(metrics),
        })
        history.append(metrics)
        state = copy.deepcopy(model.state_dict())
        epoch_path = os.path.join(C.OUT_DIR, f"vcm_crnn_{args.tag}_ep{epoch:02d}.pt")
        torch.save({
            "model": state, "config": dict(cfg), "intents": base["intents"],
            "value_tokens": base["value_tokens"], "n_params": n_params,
            "arch": "battery", "n_frames": 250, "tag": args.tag,
            "epoch": epoch, "selection": metrics,
            "recipe": "source-balanced conservative real_voice fine-tune of X6",
        }, epoch_path)
        if metrics["selection_score"] > best_metrics["selection_score"]:
            best_metrics = metrics
            best_state = state
        report = {
            "protocol": "existing training data only; TEST + tune-OOS selection; HOLDOUT untouched",
            "args": vars(args), "groups": groups, "n_params": n_params,
            "best": best_metrics, "history": history,
        }
        report_path = os.path.join(C.OUT_DIR, f"finetune_{args.tag}.json")
        with open(report_path, "w") as handle:
            json.dump(report, handle, indent=2)
        print("VALID " + json.dumps(metrics, sort_keys=True), flush=True)
        print(f"saved {epoch_path}", flush=True)

    best_path = os.path.join(C.OUT_DIR, f"vcm_crnn_{args.tag}.pt")
    torch.save({
        "model": best_state, "config": dict(cfg), "intents": base["intents"],
        "value_tokens": base["value_tokens"], "n_params": n_params,
        "arch": "battery", "n_frames": 250, "tag": args.tag,
        "selection": best_metrics,
        "recipe": "source-balanced conservative real_voice fine-tune of X6",
    }, best_path)
    print("BEST " + json.dumps(best_metrics, sort_keys=True), flush=True)
    print(f"saved {best_path}", flush=True)


if __name__ == "__main__":
    main()
