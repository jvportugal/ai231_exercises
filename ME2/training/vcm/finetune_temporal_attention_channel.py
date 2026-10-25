"""Safely adapt a selected temporal-attention VCM to playback channels.

Selection uses clean/channel TEST and tune-OOS only. HOLDOUT is never loaded.
Epoch zero is an eligible candidate, so fine-tuning cannot silently replace the
base checkpoint when every adapted epoch violates the preservation floors.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config as C
from eval_ens5c import load_split
from experiment_weight_soup import build_features, labels, load_oos
from finetune_realvoice_safe import build_channel_features
from finetune_source_curriculum import (
    cached_features,
    conditional_predictions,
    curriculum_weights,
    curriculum_worker_init,
    read_training_sources,
)
from train_centered import load_audio_split, load_noise_bank
from train_temporal_attention import (
    TemporalAttentionVCM,
    WordDataset,
    distillation_loss,
    infer,
    read_training_texts,
    word_targets,
)


def evaluate(model, frame, clean_features, channel_features, target_a,
             target_b, tune_clean, tune_channel, device):
    real = frame["is_synthetic"].fillna(1).astype(int).to_numpy() == 0
    synthetic = ~real
    source = frame["source"].astype(str).to_numpy()

    def view_metrics(features):
        logits_a, logits_b = infer(model, features, device)
        pred_a, pred_b = conditional_predictions(logits_a, logits_b)
        intent_ok = pred_a == target_a
        command_ok = intent_ok & (
            (target_b == C.N_VALUE) | (pred_b == target_b))
        return {
            "intent": round(float(intent_ok.mean() * 100), 3),
            "e2e": round(float(command_ok.mean() * 100), 3),
            "real_e2e": round(float(command_ok[real].mean() * 100), 3),
            "synthetic_e2e": round(
                float(command_ok[synthetic].mean() * 100), 3),
            "slurp_e2e": round(float(command_ok[
                real & (source == "SLURP")].mean() * 100), 3),
        }

    clean = view_metrics(clean_features)
    channel = view_metrics(channel_features)
    tune_clean_logits, _ = infer(model, tune_clean, device)
    tune_channel_logits, _ = infer(model, tune_channel, device)
    return {
        "clean": clean,
        "channel": channel,
        "tune_oos_clean": round(float(
            (tune_clean_logits.argmax(1) == C.N_INTENT).mean() * 100), 3),
        "tune_oos_channel": round(float(
            (tune_channel_logits.argmax(1) == C.N_INTENT).mean() * 100), 3),
    }


def selection_score(metrics):
    clean = metrics["clean"]
    channel = metrics["channel"]
    return (
        0.45 * clean["real_e2e"]
        + 0.45 * channel["real_e2e"]
        + 0.05 * clean["e2e"]
        + 0.025 * metrics["tune_oos_clean"]
        + 0.025 * metrics["tune_oos_channel"]
    )


def is_eligible(metrics, baseline, clean_e2e_drop, clean_real_drop,
                oos_drop):
    return (
        metrics["clean"]["e2e"]
        >= baseline["clean"]["e2e"] - clean_e2e_drop
        and metrics["clean"]["real_e2e"]
        >= baseline["clean"]["real_e2e"] - clean_real_drop
        and metrics["tune_oos_clean"]
        >= baseline["tune_oos_clean"] - oos_drop
        and metrics["tune_oos_channel"]
        >= baseline["tune_oos_channel"] - oos_drop
    )


def checkpoint_payload(model, base, tag, epoch, metrics, n_params):
    return {
        "model": copy.deepcopy(model.state_dict()),
        "config": dict(base["config"]),
        "vocabulary": list(base.get("vocabulary", [])),
        "n_params": n_params,
        "arch": base.get("arch", "temporal_attention_v1"),
        "n_frames": 250,
        "tag": tag,
        "epoch": epoch,
        "selection": metrics,
        "recipe": (
            "channel-heavy real-speech adaptation with frozen-base "
            "self-distillation; holdout sealed"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--tag", default="attention_channel_ft")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--samples-per-epoch", type=int, default=18000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--kd-weight", type=float, default=0.35)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--speed-p", type=float, default=0.20)
    parser.add_argument("--reverb-p", type=float, default=0.45)
    parser.add_argument("--channel-p", type=float, default=0.65)
    parser.add_argument("--babble-p", type=float, default=0.10)
    parser.add_argument("--specaug-p", type=float, default=0.10)
    parser.add_argument("--formant-p", type=float, default=0.25)
    parser.add_argument("--clean-e2e-drop", type=float, default=0.5)
    parser.add_argument("--clean-real-drop", type=float, default=1.0)
    parser.add_argument("--oos-drop", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=7101)
    args = parser.parse_args()

    if not 0.0 <= args.kd_weight < 1.0:
        parser.error("--kd-weight must be in [0, 1)")
    torch.set_num_threads(min(args.threads, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; HOLDOUT SEALED; base={args.base}", flush=True)

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    config = base["config"]
    model = TemporalAttentionVCM(
        auxiliary_words=config["auxiliary_words"],
        width=config["width"],
        gru_layers=config["gru_layers"],
    ).to(device)
    model.load_state_dict(base["model"])
    teacher = copy.deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    n_params = sum(parameter.numel() for parameter in model.parameters())

    rows = load_audio_split(["train", "fil", "neg_train", "neg_noise"])
    sources = read_training_sources()
    texts = read_training_texts()
    if not (len(rows) == len(sources) == len(texts)):
        raise RuntimeError(
            f"metadata mismatch: {len(rows)} {len(sources)} {len(texts)}")
    vocabulary = list(base.get("vocabulary", []))
    if len(vocabulary) != config["auxiliary_words"]:
        raise RuntimeError("base vocabulary/config mismatch")
    targets_word = word_targets(texts, vocabulary)
    group_masses = {
        "SLURP": 0.30,
        "real_voice": 0.15,
        "FluentSpeechCommands": 0.08,
        "SNIPS": 0.07,
        "other_real": 0.05,
        "synthetic_command": 0.20,
        "real_reject": 0.05,
        "synthetic_reject": 0.10,
    }
    sample_weights, groups = curriculum_weights(rows, sources, group_masses)
    noise = load_noise_bank()
    dataset = WordDataset(
        rows, noise, word_labels=targets_word, aug=True, seed=args.seed,
        n_frames=250, snr_min=12.0, snr_max=28.0, gain_db=5.0,
        shift=0.08, speed_p=args.speed_p, speed_lo=0.92, speed_hi=1.08,
        reverb_p=args.reverb_p, channel_p=args.channel_p,
        babble_p=args.babble_p, specaug_p=args.specaug_p,
        spec_tmask=1, spec_tlen=15, spec_fmask=1, spec_flen=4,
        formant_p=args.formant_p)
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.from_numpy(sample_weights), args.samples_per_epoch,
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch, sampler=sampler,
        num_workers=args.workers, pin_memory=device.type == "cuda",
        worker_init_fn=curriculum_worker_init,
        persistent_workers=args.workers > 0)

    test = load_split("test")
    tune = load_oos(str(
        Path(C.HF_DIR) / "synthetic_negatives" / "tune_oos-*.parquet"))
    target_a, target_b = labels(test)
    cache_root = Path(os.environ.get(
        "AI231_VCM_FEATURE_CACHE", "/tmp/vcm_feature_cache"))
    test_clean = cached_features(
        cache_root / "test_clean_250.npy", lambda: build_features(test, 250))
    tune_clean = cached_features(
        cache_root / "tune_oos_clean_250.npy", lambda: build_features(tune, 250))
    test_channel = cached_features(
        cache_root / "test_channel_seed2024_250.npy",
        lambda: build_channel_features(test, seed=2024))
    tune_channel = cached_features(
        cache_root / "tune_oos_channel_seed3024_250.npy",
        lambda: build_channel_features(tune, seed=3024))

    output = Path(C.OUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    model.eval()
    baseline = evaluate(
        model, test, test_clean, test_channel, target_a, target_b,
        tune_clean, tune_channel, device)
    baseline.update(epoch=0, train_loss=None, elapsed_s=0.0,
                    selection_score=selection_score(baseline), eligible=True)
    history = [baseline]
    best = baseline
    best_state = copy.deepcopy(model.state_dict())
    print("BASELINE " + json.dumps(baseline, sort_keys=True), flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for features, intent, slot, _, _ in loader:
            features = features.to(device, non_blocking=True)
            intent = intent.to(device, non_blocking=True)
            slot = slot.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits_a, logits_b, _ = model(features)
            supervised = (
                F.cross_entropy(logits_a, intent, label_smoothing=0.03)
                + F.cross_entropy(logits_b, slot, label_smoothing=0.02))
            with torch.inference_mode():
                teacher_a, teacher_b, _ = teacher(features)
            distilled = (
                distillation_loss(logits_a, teacher_a, args.temperature)
                + distillation_loss(logits_b, teacher_b, args.temperature))
            loss = ((1.0 - args.kd_weight) * supervised
                    + args.kd_weight * distilled)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(features)
            seen += len(features)
        scheduler.step()

        model.eval()
        metrics = evaluate(
            model, test, test_clean, test_channel, target_a, target_b,
            tune_clean, tune_channel, device)
        metrics.update(
            epoch=epoch,
            train_loss=round(total / seen, 6),
            elapsed_s=round(time.time() - started, 1),
            lr=optimizer.param_groups[0]["lr"],
        )
        metrics["selection_score"] = selection_score(metrics)
        metrics["eligible"] = is_eligible(
            metrics, baseline, args.clean_e2e_drop,
            args.clean_real_drop, args.oos_drop)
        history.append(metrics)
        payload = checkpoint_payload(
            model, base, args.tag, epoch, metrics, n_params)
        torch.save(
            payload, output / f"vcm_attention_channel_{args.tag}_ep{epoch:02d}.pt")
        if (metrics["eligible"]
                and metrics["selection_score"] > best["selection_score"]):
            best = metrics
            best_state = copy.deepcopy(model.state_dict())
        report = {
            "protocol": (
                "clean/channel TEST + tune-OOS selection; HOLDOUT sealed; "
                "epoch zero retained when adaptation is unsafe"),
            "args": {**vars(args), "base": str(args.base)},
            "groups": groups,
            "baseline": baseline,
            "best": best,
            "history": history,
        }
        with (output / f"finetune_attention_channel_{args.tag}.json").open("w") as handle:
            json.dump(report, handle, indent=2)
        print("VALID " + json.dumps(metrics, sort_keys=True), flush=True)

    final = checkpoint_payload(model, base, args.tag, best["epoch"], best,
                               n_params)
    final["model"] = best_state
    torch.save(final, output / f"vcm_attention_channel_{args.tag}.pt")
    print("BEST " + json.dumps(best, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
