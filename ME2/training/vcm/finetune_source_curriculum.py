"""Source-aware real-speech curriculum for the compact VCM CRNN.

This experiment deliberately never loads HOLDOUT.  TEST is the development
set, tune-OOS constrains rejection, and every epoch is saved.  The final
holdout evaluator must be run separately only after weights and policy are
frozen.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
from battery import (BatDataset, CRNN_N, _make_rirs, _need,
                     add_playback_channel, add_reverb, wav_to_mel)
from eval_ens5c import load_split
from experiment_weight_soup import build_features, labels, load_oos
from finetune_realvoice_safe import distillation_loss, seed_worker
from train_centered import (center_crop, decode_wav, load_audio_split,
                            load_noise_bank)


HARD_INTENT_MULTIPLIER = {
    "PLAY_MUSIC": 3.0,
    "WEATHER": 2.5,
    "CALL": 2.5,
    "VOLUME_UP": 2.2,
    "VOLUME_DOWN": 2.2,
    "TIME": 2.0,
    "LIGHT_ON": 2.0,
    "MESSAGE": 1.8,
    "LIST_REMINDERS": 1.5,
}


def curriculum_worker_init(worker_id):
    """Prevent N workers from each creating a full CPU thread pool."""
    torch.set_num_threads(1)
    seed_worker(worker_id)


def guarded_split(name):
    """Make accidental use of the sealed split impossible in this trainer."""
    if name.lower() == "holdout":
        raise RuntimeError("HOLDOUT is sealed during training and selection")
    return load_split(name)


def read_training_sources():
    def sources(pattern, fallback):
        files = sorted(glob.glob(pattern))
        frames = []
        for path in files:
            frame = pd.read_parquet(path, columns=["source"])
            frames.extend(frame["source"].fillna(fallback).astype(str).tolist())
        return frames

    result = sources(os.path.join(C.DATA_DIR, "train-*.parquet"), "unknown")
    result += sources(os.path.join(C.HF_DIR, "supplemental_fil", "*.parquet"),
                      "fil50_persona")
    result += sources(os.path.join(C.HF_DIR, "synthetic_negatives",
                                   "train-*.parquet"), "synthetic_negative")
    result += sources(os.path.join(C.HF_DIR, "synthetic_negatives",
                                   "noise-*.parquet"), "negative_noise")
    return np.asarray(result, dtype=object)


def _assign_group(source, is_real, is_reject):
    if is_reject:
        return "real_reject" if is_real else "synthetic_reject"
    if not is_real:
        return "synthetic_command"
    if source in {"SLURP", "real_voice", "FluentSpeechCommands", "SNIPS"}:
        return str(source)
    return "other_real"


def curriculum_weights(rows, sources, group_masses):
    targets = np.asarray([row[1] for row in rows], dtype=np.int64)
    real = np.asarray([row[3] for row in rows], dtype=bool)
    reject = targets == C.N_INTENT
    groups = np.asarray([
        _assign_group(source, is_real, is_reject)
        for source, is_real, is_reject in zip(sources, real, reject)
    ], dtype=object)
    weights = np.zeros(len(rows), dtype=np.float64)
    report = {}
    for group, mass in group_masses.items():
        mask = groups == group
        if not mask.any():
            raise RuntimeError(f"empty curriculum group: {group}")
        if group.endswith("reject"):
            weights[mask] = mass / mask.sum()
        else:
            present = sorted(set(targets[mask].tolist()))
            multipliers = np.asarray([
                HARD_INTENT_MULTIPLIER.get(C.INTENTS[target], 1.0)
                for target in present
            ], dtype=np.float64)
            multipliers /= multipliers.sum()
            for target, class_mass in zip(present, multipliers):
                class_mask = mask & (targets == target)
                weights[class_mask] = mass * class_mass / class_mask.sum()
        report[group] = {
            "rows": int(mask.sum()),
            "sampling_mass": float(weights[mask].sum()),
        }
    if not np.isclose(weights.sum(), 1.0):
        raise RuntimeError(f"curriculum mass is {weights.sum()}, expected 1")
    return weights, report


def frequency_warp(mel, rng, maximum=0.12):
    """Small mel-axis warp approximating vocal-tract/speaker variation."""
    scale = float(rng.uniform(1.0 - maximum, 1.0 + maximum))
    bins = np.arange(mel.shape[0], dtype=np.float32)
    center = 0.5 * (mel.shape[0] - 1)
    source = center + (bins - center) / scale
    warped = np.empty_like(mel)
    for index in range(mel.shape[1]):
        warped[:, index] = np.interp(
            source, bins, mel[:, index],
            left=float(mel[0, index]), right=float(mel[-1, index]))
    return warped


class CurriculumDataset(BatDataset):
    def __init__(self, *args, formant_p=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.formant_p = formant_p

    def __getitem__(self, index):
        features, target_a, target_b, weight = super().__getitem__(index)
        if (self.aug and self.rows[index][3] and self.formant_p > 0
                and self.rng.random() < self.formant_p):
            features = torch.from_numpy(
                frequency_warp(features[0].numpy(), self.rng))[None]
        return features, target_a, target_b, weight


class CorrectBiGRUPool(CRNN_N):
    """Pool the final hidden state from both BiGRU directions.

    CRNN_N uses the final output time step, whose backward half has only seen
    that one step.  The GRU hidden tensor contains the proper final state for
    each direction.  This changes representation semantics, so it is only
    valid for a model trained from scratch.
    """

    def forward(self, x):
        h = self.cnn(x)
        h = h.permute(0, 3, 1, 2).mean(dim=3)
        _, hidden = self.gru(h)
        feat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        feat = self.fc(feat)
        return self.head_a(feat), self.head_b(feat)


def build_channel_features(frame, n_frames, seed):
    rng = np.random.RandomState(seed)
    rirs = _make_rirs(n=40, rng=np.random.RandomState(seed + 1))
    need = _need(n_frames)
    output = np.empty((len(frame), 1, C.N_MELS, n_frames), np.float32)
    for index in range(len(frame)):
        audio = frame["audio"].iloc[index]
        raw = audio["bytes"] if isinstance(audio, dict) else audio
        try:
            waveform = decode_wav(raw)
            waveform = add_reverb(waveform, rirs, rng)
            waveform = add_playback_channel(waveform, rng)
            output[index, 0] = wav_to_mel(
                center_crop(waveform, n=need), n_frames)
        except Exception:
            output[index, 0] = 0.0
    return output


def infer_logits(model, features, device, batch=256):
    intent, slot = [], []
    with torch.inference_mode():
        for start in range(0, len(features), batch):
            x = torch.from_numpy(features[start:start + batch]).to(device)
            out_a, out_b = model(x)
            intent.append(out_a.cpu().numpy())
            slot.append(out_b.cpu().numpy())
    return np.concatenate(intent), np.concatenate(slot)


def cached_features(path, builder):
    path = Path(path)
    if path.exists():
        print(f"loading feature cache {path}", flush=True)
        return np.load(path, mmap_mode="r")
    value = builder()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value)
    print(f"saved feature cache {path}", flush=True)
    return value


def conditional_predictions(logits_a, logits_b):
    pred_a = logits_a.argmax(1)
    pred_b = np.full(len(pred_a), C.N_VALUE, dtype=np.int64)
    for intent, values in C.SLOTTED.items():
        intent_id = C.INTENT2IDX[intent]
        valid = np.asarray([
            C.VALUE2IDX[C.PAIR2TOKEN[(intent, value)]] for value in values
        ])
        rows = np.flatnonzero(pred_a == intent_id)
        if len(rows):
            pred_b[rows] = valid[logits_b[np.ix_(rows, valid)].argmax(1)]
    return pred_a, pred_b


def evaluate(model, frame, features, target_a, target_b, tune_features,
             device, channel_features=None):
    logits_a, logits_b = infer_logits(model, features, device)
    pred_a, pred_b = conditional_predictions(logits_a, logits_b)
    intent_ok = pred_a == target_a
    command_ok = intent_ok & ((target_b == C.N_VALUE) | (pred_b == target_b))
    real = frame["is_synthetic"].fillna(1).astype(int).to_numpy() == 0
    synthetic = ~real
    tune_a, _ = infer_logits(model, tune_features, device)
    metrics = {
        "overall_intent": round(float(intent_ok.mean() * 100), 3),
        "overall_e2e": round(float(command_ok.mean() * 100), 3),
        "real_e2e": round(float(command_ok[real].mean() * 100), 3),
        "synthetic_e2e": round(float(command_ok[synthetic].mean() * 100), 3),
        "slurp_e2e": round(float(command_ok[
            real & (frame["source"].to_numpy() == "SLURP")].mean() * 100), 3),
        "tune_oos_reject": round(float(
            (tune_a.argmax(1) == C.N_INTENT).mean() * 100), 3),
    }
    if channel_features is not None:
        channel_a, channel_b = infer_logits(model, channel_features, device)
        channel_pa, channel_pb = conditional_predictions(channel_a, channel_b)
        channel_ok = ((channel_pa == target_a)
                      & ((target_b == C.N_VALUE) | (channel_pb == target_b)))
        metrics["channel_e2e"] = round(float(channel_ok.mean() * 100), 3)
        metrics["channel_real_e2e"] = round(
            float(channel_ok[real].mean() * 100), 3)
    return metrics


def selection_score(metrics):
    penalty = 30.0 * max(0.0, 90.0 - metrics["overall_e2e"])
    penalty += 8.0 * max(0.0, 95.0 - metrics["synthetic_e2e"])
    penalty += 2.0 * max(0.0, 68.0 - metrics["tune_oos_reject"])
    return (metrics["real_e2e"] + 0.20 * metrics["slurp_e2e"]
            + 0.05 * metrics["overall_e2e"]
            + 0.03 * metrics["tune_oos_reject"] - penalty)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--from-scratch", action="store_true",
                        help="initialize the architecture randomly instead of loading base weights")
    parser.add_argument("--pooling", choices=("legacy", "correct"),
                        default="legacy",
                        help="BiGRU pooling (correct may adapt legacy weights)")
    parser.add_argument("--tag", default="source_curriculum")
    parser.add_argument("--nframes", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--samples-per-epoch", type=int, default=30000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--lr", type=float, default=4e-5)
    parser.add_argument("--cnn-lr-scale", type=float, default=0.5)
    parser.add_argument("--kd-weight", type=float, default=0.15)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--speed-p", type=float, default=0.25)
    parser.add_argument("--reverb-p", type=float, default=0.25)
    parser.add_argument("--channel-p", type=float, default=0.30)
    parser.add_argument("--babble-p", type=float, default=0.10)
    parser.add_argument("--specaug-p", type=float, default=0.20)
    parser.add_argument("--formant-p", type=float, default=0.35)
    parser.add_argument("--curriculum", choices=("balanced", "real-heavy"),
                        default="balanced")
    parser.add_argument("--channel-val", action="store_true")
    parser.add_argument("--seed", type=int, default=3101)
    args = parser.parse_args()
    torch.set_num_threads(min(args.threads, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; HOLDOUT SEALED; nframes={args.nframes}", flush=True)

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    cfg = base["config"]
    model_class = CorrectBiGRUPool if args.pooling == "correct" else CRNN_N
    model = model_class(n_a=C.N_A, n_b=C.N_B, ch=tuple(cfg["ch"]),
                        gru=cfg["gru"],
                        gru_layers=cfg["gru_layers"]).to(device)
    if not args.from_scratch:
        model.load_state_dict(base["model"])
    teacher = None if args.from_scratch or args.kd_weight <= 0 else copy.deepcopy(model).eval()
    model.gru.flatten_parameters()
    if teacher is not None:
        teacher.gru.flatten_parameters()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)

    rows = load_audio_split(["train", "fil", "neg_train", "neg_noise"])
    sources = read_training_sources()
    if len(rows) != len(sources):
        raise RuntimeError(f"row/source mismatch: {len(rows)} != {len(sources)}")
    if args.curriculum == "real-heavy":
        group_masses = {
            "SLURP": 0.38,
            "real_voice": 0.14,
            "FluentSpeechCommands": 0.08,
            "SNIPS": 0.07,
            "other_real": 0.05,
            "synthetic_command": 0.13,
            "real_reject": 0.06,
            "synthetic_reject": 0.09,
        }
    else:
        group_masses = {
            "SLURP": 0.25,
            "real_voice": 0.12,
            "FluentSpeechCommands": 0.07,
            "SNIPS": 0.06,
            "other_real": 0.05,
            "synthetic_command": 0.27,
            "real_reject": 0.06,
            "synthetic_reject": 0.12,
        }
    sample_weights, groups = curriculum_weights(rows, sources, group_masses)
    print(json.dumps({"groups": groups}, sort_keys=True), flush=True)

    noise = load_noise_bank()
    dataset = CurriculumDataset(
        rows, noise, aug=True, seed=args.seed, n_frames=args.nframes,
        snr_min=12.0, snr_max=28.0, gain_db=5.0, shift=0.08,
        speed_p=args.speed_p, speed_lo=0.92, speed_hi=1.08,
        reverb_p=args.reverb_p, channel_p=args.channel_p,
        babble_p=args.babble_p, specaug_p=args.specaug_p,
        spec_tmask=1, spec_tlen=15, spec_fmask=1, spec_flen=4,
        formant_p=args.formant_p)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = torch.utils.data.WeightedRandomSampler(
        torch.from_numpy(sample_weights), args.samples_per_epoch,
        replacement=True, generator=generator)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch, sampler=sampler,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
        worker_init_fn=curriculum_worker_init,
        persistent_workers=args.workers > 0)

    test = guarded_split("test")
    tune = load_oos(str(Path(C.HF_DIR) / "synthetic_negatives" /
                        "tune_oos-*.parquet"))
    target_a, target_b = labels(test)
    print("building clean TEST/tune-OOS features", flush=True)
    cache_root = Path(os.environ.get(
        "AI231_VCM_FEATURE_CACHE", "/tmp/vcm_feature_cache"))
    test_features = cached_features(
        cache_root / f"test_clean_{args.nframes}.npy",
        lambda: build_features(test, args.nframes))
    tune_features = cached_features(
        cache_root / f"tune_oos_clean_{args.nframes}.npy",
        lambda: build_features(tune, args.nframes))
    channel_features = None
    if args.channel_val:
        print("building fixed channel TEST features", flush=True)
        channel_features = cached_features(
            cache_root / f"test_channel_{args.nframes}_{args.seed + 1000}.npy",
            lambda: build_channel_features(
                test, args.nframes, args.seed + 1000))

    cnn = list(model.cnn.parameters())
    other = [p for name, p in model.named_parameters()
             if not name.startswith("cnn.")]
    optimizer = torch.optim.AdamW([
        {"params": cnn, "lr": args.lr * args.cnn_lr_scale},
        {"params": other, "lr": args.lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    output = Path(C.OUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best_state = copy.deepcopy(model.state_dict())
    model.eval()
    best = evaluate(model, test, test_features, target_a, target_b,
                    tune_features, device, channel_features)
    best.update(epoch=0, selection_score=selection_score(best))
    history.append(best)
    print("VALID " + json.dumps(best, sort_keys=True), flush=True)
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        # Preserve pretrained BatchNorm statistics while fine-tuning, but a
        # random-start CNN must learn its own running mean and variance.
        if not args.from_scratch:
            for module in model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        total, seen = 0.0, 0
        for features, target_a_batch, target_b_batch, _ in loader:
            features = features.to(device, non_blocking=True)
            target_a_batch = target_a_batch.to(device, non_blocking=True)
            target_b_batch = target_b_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits_a, logits_b = model(features)
            supervised = (F.cross_entropy(logits_a, target_a_batch,
                                           label_smoothing=0.03)
                          + F.cross_entropy(logits_b, target_b_batch,
                                            label_smoothing=0.02))
            if teacher is None:
                loss = supervised
            else:
                with torch.inference_mode():
                    teacher_a, teacher_b = teacher(features)
                distilled = (distillation_loss(
                    logits_a, teacher_a, args.temperature)
                    + distillation_loss(logits_b, teacher_b,
                                        args.temperature))
                loss = ((1.0 - args.kd_weight) * supervised
                        + args.kd_weight * distilled)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(features)
            seen += len(features)
        scheduler.step()
        model.eval()
        metrics = evaluate(model, test, test_features, target_a, target_b,
                           tune_features, device, channel_features)
        metrics.update(epoch=epoch, train_loss=round(total / seen, 6),
                       elapsed_s=round(time.time() - started, 1))
        metrics["selection_score"] = selection_score(metrics)
        history.append(metrics)
        state = copy.deepcopy(model.state_dict())
        checkpoint = {
            "model": state,
            "config": dict(cfg, nframes=args.nframes),
            "intents": base.get("intents", C.INTENTS),
            "value_tokens": base.get("value_tokens", C.VALUE_TOKENS),
            "n_params": sum(p.numel() for p in model.parameters()),
            "arch": "battery", "pooling": args.pooling,
            "n_frames": args.nframes,
            "tag": args.tag, "epoch": epoch, "selection": metrics,
            "recipe": "holdout-sealed source-aware real-speech curriculum",
        }
        torch.save(checkpoint, output / f"vcm_crnn_{args.tag}_ep{epoch:02d}.pt")
        if metrics["selection_score"] > best["selection_score"]:
            best = metrics
            best_state = state
        report = {
            "protocol": ("TEST + tune-OOS development selection only; "
                         "HOLDOUT is not loaded"),
            "args": vars(args), "groups": groups,
            "hard_intent_multiplier": HARD_INTENT_MULTIPLIER,
            "best": best, "history": history,
        }
        with open(output / f"finetune_{args.tag}.json", "w") as handle:
            json.dump(report, handle, indent=2, default=str)
        print("VALID " + json.dumps(metrics, sort_keys=True), flush=True)

    final = {
        "model": best_state,
        "config": dict(cfg, nframes=args.nframes),
        "intents": base.get("intents", C.INTENTS),
        "value_tokens": base.get("value_tokens", C.VALUE_TOKENS),
        "n_params": sum(p.numel() for p in model.parameters()),
        "arch": "battery", "pooling": args.pooling,
        "n_frames": args.nframes,
        "tag": args.tag, "selection": best,
        "recipe": "holdout-sealed source-aware real-speech curriculum",
    }
    torch.save(final, output / f"vcm_crnn_{args.tag}.pt")
    print("BEST " + json.dumps(best, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
