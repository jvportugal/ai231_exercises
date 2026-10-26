"""Train a compact, time-preserving attention VCM from scratch.

This is intentionally a clean architectural break from the legacy CRNN.  The
CNN preserves substantially more time resolution, the BiGRU is consumed as a
full sequence, and self-attention plus attentive statistics pooling decides
which frames matter.  An optional auxiliary bag-of-words objective uses only
the transcripts already present in the supplied training set.

HOLDOUT is never loaded here.  TEST and tune-OOS are development sets.
Training and development speech come only from the cached
`airimonda/ai231-me2-voice-commands` repository; the approved noise bank is
used only for augmentation/rejection examples.
"""
from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
from battery import CRNN_N, build_deployment_features
from eval_ens5c import load_split
from experiment_weight_soup import build_features, labels, load_oos
from finetune_source_curriculum import (
    CurriculumDataset,
    cached_features,
    conditional_predictions,
    curriculum_weights,
    curriculum_worker_init,
    read_training_sources,
)
from train_centered import load_audio_split, load_noise_bank


STOPWORDS = {
    "a", "an", "and", "are", "at", "be", "can", "could", "do", "for",
    "from", "get", "give", "i", "in", "is", "it", "me", "my", "of",
    "please", "some", "the", "this", "to", "what", "whats",
    "would", "you",
}

# These words carry the command family or the value itself.  The auxiliary
# recognizer receives extra weight on them so the shared encoder cannot solve
# the task only from surrounding template words such as "set" or "change".
SCHEMA_KEYWORDS = {
    "alarm", "brightness", "color", "degrees", "exercise", "green",
    "minute", "percent", "red", "reminder", "seconds", "study",
    "temperature", "timer", "water", "blue",
    "on", "off", "up", "down", "play", "pause", "stop",
}


def read_training_texts():
    patterns = [
        os.path.join(C.DATA_DIR, "train-*.parquet"),
        os.path.join(C.HF_DIR, "supplemental_fil", "*.parquet"),
        os.path.join(C.HF_DIR, "synthetic_negatives", "train-*.parquet"),
        os.path.join(C.HF_DIR, "synthetic_negatives", "noise-*.parquet"),
    ]
    texts = []
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            frame = pd.read_parquet(path, columns=["transcript"])
            texts.extend(frame["transcript"].fillna("").astype(str).tolist())
    return texts


def words(text):
    return set(re.findall(r"[a-z]+", text.lower().replace("'", "")))


def make_vocabulary(texts, rows, size):
    """Reserve observed command keywords, then fill with frequent real words.

    This defines NEW training runs only. Existing checkpoints must retain
    their stored vocabulary and word-head ordering.
    """
    counts = Counter()
    for text, (_, intent, _, is_real) in zip(texts, rows):
        if not is_real or intent == C.N_INTENT:
            continue
        counts.update(token for token in words(text)
                      if token not in STOPWORDS and len(token) > 1)
    candidates = [(count, token) for token, count in counts.items()
                  if count >= 3 or token in SCHEMA_KEYWORDS]
    candidates.sort(key=lambda item: (item[1] not in SCHEMA_KEYWORDS,
                                      -item[0], item[1]))
    return [token for _, token in candidates[:size]]


def word_targets(texts, vocabulary):
    lookup = {word: index for index, word in enumerate(vocabulary)}
    result = np.zeros((len(texts), len(vocabulary)), dtype=np.float32)
    for row, text in enumerate(texts):
        for token in words(text):
            if token in lookup:
                result[row, lookup[token]] = 1.0
    return result


class WordDataset(CurriculumDataset):
    def __init__(self, *args, word_labels, **kwargs):
        super().__init__(*args, **kwargs)
        self.word_labels = word_labels

    def __getitem__(self, index):
        features, intent, slot, weight = super().__getitem__(index)
        return (features, intent, slot, weight,
                torch.from_numpy(self.word_labels[index]))


class ResidualDepthwiseBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1)):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 3, stride=stride, padding=1,
            groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = (nn.Identity() if in_channels == out_channels
                     and stride == (1, 1) else nn.Sequential(
                         nn.Conv2d(in_channels, out_channels, 1,
                                   stride=stride, bias=False),
                         nn.BatchNorm2d(out_channels)))

    def forward(self, x):
        residual = self.skip(x)
        x = F.silu(self.bn1(self.depthwise(x)))
        x = self.bn2(self.pointwise(x))
        return F.silu(x + residual)


class TemporalAttentionVCM(nn.Module):
    """Time-preserving CNN -> BiGRU -> self-attention -> attentive stats."""
    def __init__(self, n_a=C.N_A, n_b=C.N_B, auxiliary_words=0,
                 width=96, gru_layers=2, dropout=0.15,
                 structured_slots=False):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=(2, 1), padding=1, bias=False),
            nn.BatchNorm2d(32), nn.SiLU())
        self.cnn = nn.Sequential(
            ResidualDepthwiseBlock(32, 48, stride=(2, 2)),
            ResidualDepthwiseBlock(48, 64, stride=(2, 2)),
            ResidualDepthwiseBlock(64, width, stride=(2, 2)),
            ResidualDepthwiseBlock(width, width),
        )
        self.gru = nn.GRU(
            width, width, num_layers=gru_layers, batch_first=True,
            bidirectional=True, dropout=dropout if gru_layers > 1 else 0.0)
        dimension = width * 2
        self.attention = nn.MultiheadAttention(
            dimension, num_heads=4, dropout=dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(
            nn.Linear(dimension, dimension * 2), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(dimension * 2, dimension))
        self.ffn_norm = nn.LayerNorm(dimension)
        self.pool_score = nn.Sequential(
            nn.Linear(dimension, width), nn.Tanh(), nn.Linear(width, 1))
        self.embedding = nn.Sequential(
            nn.Linear(dimension * 2, dimension), nn.LayerNorm(dimension),
            nn.SiLU(), nn.Dropout(0.25))
        self.intent_head = nn.Linear(dimension, n_a)
        self.slot_head = nn.Linear(dimension, n_b)
        self.structured_slots = bool(structured_slots)
        # One independent three-way value classifier per slotted intent.
        # The original global slot head is deliberately retained: its
        # cross-intent probabilities provide the reverse evidence needed by
        # guarded joint intent-slot decoding.
        self.local_slot_heads = (nn.ModuleList([
            nn.Linear(dimension, len(values))
            for values in C.SLOTTED.values()
        ]) if self.structured_slots else None)
        self.word_head = (nn.Linear(dimension, auxiliary_words)
                          if auxiliary_words else None)
        self.auxiliary_words = auxiliary_words

    def encode(self, x):
        x = self.cnn(self.stem(x))
        sequence = x.mean(dim=2).transpose(1, 2)
        sequence, _ = self.gru(sequence)
        attended, _ = self.attention(
            sequence, sequence, sequence, need_weights=False)
        sequence = self.attention_norm(sequence + attended)
        sequence = self.ffn_norm(sequence + self.ffn(sequence))
        weights = torch.softmax(self.pool_score(sequence), dim=1)
        mean = torch.sum(weights * sequence, dim=1)
        variance = torch.sum(
            weights * torch.square(sequence - mean[:, None, :]), dim=1)
        std = torch.sqrt(torch.clamp(variance, min=1e-5))
        return self.embedding(torch.cat((mean, std), dim=1))

    def forward(self, x):
        embedding = self.encode(x)
        intent = self.intent_head(embedding)
        slot = self.slot_head(embedding)
        words_out = (self.word_head(embedding)
                     if self.word_head is not None else None)
        if self.local_slot_heads is not None:
            local_slots = torch.cat(
                [head(embedding) for head in self.local_slot_heads], dim=1)
            # Exported structured models expose all evidence used by the
            # runtime joint decoder.  Legacy models keep their three-output
            # contract for checkpoint and screening-script compatibility.
            return intent, slot, local_slots, words_out
        return intent, slot, words_out


def infer(model, features, device, batch_size=256):
    intents, slots, local_slots = [], [], []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(
                np.asarray(features[start:start + batch_size])).to(device)
            outputs = model(batch)
            intent, slot = outputs[:2]
            intents.append(intent.cpu().numpy())
            slots.append(slot.cpu().numpy())
            if len(outputs) >= 4:
                local_slots.append(outputs[2].cpu().numpy())
    return (np.concatenate(intents), np.concatenate(slots),
            np.concatenate(local_slots) if local_slots else None)


def evaluate(model, frame, features, target_a, target_b, tune_features,
             device):
    logits_a, logits_b, logits_local = infer(model, features, device)
    pred_a, pred_b = conditional_predictions(logits_a, logits_b)
    if logits_local is not None:
        slotted_start = len(C.FIXED)
        for group_index, (intent, values) in enumerate(C.SLOTTED.items()):
            intent_id = C.INTENT2IDX[intent]
            rows = np.flatnonzero(pred_a == intent_id)
            if len(rows):
                base = group_index * len(values)
                local = logits_local[
                    np.ix_(rows, np.arange(base, base + len(values)))
                ].argmax(1)
                pred_b[rows] = base + local
    intent_ok = pred_a == target_a
    command_ok = intent_ok & ((target_b == C.N_VALUE) | (pred_b == target_b))
    real = frame["is_synthetic"].fillna(1).astype(int).to_numpy() == 0
    source = frame["source"].astype(str).to_numpy()
    tune_a, _, _ = infer(model, tune_features, device)
    return {
        "overall_intent": round(float(intent_ok.mean() * 100), 3),
        "overall_e2e": round(float(command_ok.mean() * 100), 3),
        "real_e2e": round(float(command_ok[real].mean() * 100), 3),
        "synthetic_e2e": round(float(command_ok[~real].mean() * 100), 3),
        "slurp_e2e": round(float(command_ok[real & (source == "SLURP")].mean() * 100), 3),
        "tune_oos_reject": round(float(
            (tune_a.argmax(1) == C.N_INTENT).mean() * 100), 3),
    }


def selection_score(metrics):
    penalty = 30.0 * max(0.0, 90.0 - metrics["overall_e2e"])
    penalty += 8.0 * max(0.0, 95.0 - metrics["synthetic_e2e"])
    penalty += 2.0 * max(0.0, 68.0 - metrics["tune_oos_reject"])
    if "deployment_e2e" in metrics:
        penalty += 10.0 * max(0.0, 88.0 - metrics["deployment_e2e"])
    return (metrics["real_e2e"]
            + 0.50 * metrics.get("deployment_real_e2e", 0.0)
            + 0.25 * metrics["slurp_e2e"]
            + 0.05 * metrics["overall_e2e"]
            + 0.03 * metrics["tune_oos_reject"] - penalty)


def distillation_loss(student_logits, teacher_logits, temperature):
    """Temperature-scaled KL loss used for cross-architecture distillation."""
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature * temperature)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--samples-per-epoch", type=int, default=36000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--gru-layers", type=int, default=2)
    parser.add_argument("--aux-words", type=int, default=96)
    parser.add_argument("--aux-weight", type=float, default=0.15)
    parser.add_argument("--keyword-weight", type=float, default=2.0,
                        help="extra auxiliary loss weight for schema/value words")
    parser.add_argument("--structured-slots", action="store_true",
                        help="train six local 3-way heads and retain the global slot head")
    parser.add_argument("--local-slot-weight", type=float, default=0.75)
    parser.add_argument("--pause-p", type=float, default=0.25)
    parser.add_argument("--pause-min", type=float, default=0.20)
    parser.add_argument("--pause-max", type=float, default=0.80)
    parser.add_argument("--deployment-frontend-p", type=float, default=0.0,
                        help="probability of VAD-center/peak-normalize matching runtime")
    parser.add_argument("--short-form-boost", type=float, default=2.0,
                        help="sampling multiplier for <=3-word slotted commands")
    parser.add_argument("--color-short-boost", type=float, default=1.0,
                        help="additional multiplier for <=3-word COLOR commands")
    parser.add_argument("--base-checkpoint", default="",
                        help="optional structured checkpoint to fine-tune")
    parser.add_argument("--curriculum", choices=("balanced", "real-heavy"),
                        default="balanced")
    parser.add_argument("--teacher", default="",
                        help="optional CRNN checkpoint used for logit distillation")
    parser.add_argument("--kd-weight", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=5101)
    args = parser.parse_args()

    if not 0.0 <= args.kd_weight < 1.0:
        parser.error("--kd-weight must be in [0, 1)")
    if args.kd_weight > 0.0 and not args.teacher:
        parser.error("--teacher is required when --kd-weight is positive")

    torch.set_num_threads(min(args.threads, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; HOLDOUT SEALED; architecture=temporal_attention", flush=True)

    rows = load_audio_split(["train", "fil", "neg_train", "neg_noise"])
    sources = read_training_sources()
    texts = read_training_texts()
    if not (len(rows) == len(sources) == len(texts)):
        raise RuntimeError(
            f"metadata mismatch rows={len(rows)} sources={len(sources)} texts={len(texts)}")
    vocabulary = make_vocabulary(texts, rows, args.aux_words)
    targets_word = word_targets(texts, vocabulary)
    print(json.dumps({"vocabulary": vocabulary}, sort_keys=True), flush=True)

    if args.curriculum == "real-heavy":
        group_masses = {
            "SLURP": 0.34,
            "real_voice": 0.15,
            "FluentSpeechCommands": 0.08,
            "SNIPS": 0.07,
            "other_real": 0.05,
            "synthetic_command": 0.18,
            "real_reject": 0.05,
            "synthetic_reject": 0.08,
        }
    else:
        group_masses = {
            "SLURP": 0.30,
            "real_voice": 0.13,
            "FluentSpeechCommands": 0.07,
            "SNIPS": 0.06,
            "other_real": 0.04,
            "synthetic_command": 0.25,
            "real_reject": 0.05,
            "synthetic_reject": 0.10,
        }
    sample_weights, groups = curriculum_weights(rows, sources, group_masses)
    short_form = np.asarray([
        len(words(text)) <= 3 and intent != C.N_INTENT and slot != C.N_VALUE
        for text, (_, intent, slot, _) in zip(texts, rows)
    ], dtype=bool)
    if args.short_form_boost != 1.0:
        sample_weights[short_form] *= args.short_form_boost
    color_short = np.asarray([
        is_short and intent == C.INTENT2IDX["COLOR"]
        for is_short, (_, intent, _, _) in zip(short_form, rows)
    ], dtype=bool)
    if args.color_short_boost != 1.0:
        sample_weights[color_short] *= args.color_short_boost
    sample_weights /= sample_weights.sum()
    groups["short_form_boost"] = {
        "rows": int(short_form.sum()), "multiplier": args.short_form_boost,
    }
    groups["color_short_boost"] = {
        "rows": int(color_short.sum()), "multiplier": args.color_short_boost,
    }
    print(json.dumps({"groups": groups}, sort_keys=True), flush=True)

    noise = load_noise_bank()
    dataset = WordDataset(
        rows, noise, word_labels=targets_word, aug=True, seed=args.seed,
        n_frames=250, snr_min=12.0, snr_max=28.0, gain_db=5.0,
        shift=0.08, speed_p=0.20, speed_lo=0.92, speed_hi=1.08,
        reverb_p=0.10, channel_p=0.15, babble_p=0.05,
        specaug_p=0.10, spec_tmask=1, spec_tlen=15,
        spec_fmask=1, spec_flen=4, formant_p=0.25,
        pause_p=args.pause_p, pause_min=args.pause_min,
        pause_max=args.pause_max,
        deployment_frontend_p=args.deployment_frontend_p)
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
    tune = load_oos(str(Path(C.HF_DIR) / "synthetic_negatives" /
                        "tune_oos-*.parquet"))
    target_a, target_b = labels(test)
    cache_root = Path(os.environ.get(
        "AI231_VCM_FEATURE_CACHE", "/tmp/vcm_feature_cache"))
    test_features = cached_features(
        cache_root / "test_clean_250.npy",
        lambda: build_features(test, 250))
    deployment_features = cached_features(
        cache_root / "test_deployment_250.npy",
        lambda: build_deployment_features(test, 250))
    tune_features = cached_features(
        cache_root / "tune_oos_clean_250.npy",
        lambda: build_features(tune, 250))

    model = TemporalAttentionVCM(
        auxiliary_words=len(vocabulary), width=args.width,
        gru_layers=args.gru_layers,
        structured_slots=args.structured_slots).to(device)
    if args.base_checkpoint:
        base = torch.load(args.base_checkpoint, map_location="cpu",
                          weights_only=False)
        base_vocabulary = base.get("vocabulary", [])
        if base_vocabulary != vocabulary:
            raise ValueError("base checkpoint word vocabulary does not match")
        base_config = base.get("config", {})
        if (int(base_config.get("width", -1)) != args.width
                or int(base_config.get("gru_layers", -1)) != args.gru_layers
                or bool(base_config.get("structured_slots", False))
                != args.structured_slots):
            raise ValueError("base checkpoint architecture does not match arguments")
        model.load_state_dict(base["model"])
        print(f"base_checkpoint={args.base_checkpoint}", flush=True)
    teacher = None
    if args.kd_weight > 0.0:
        teacher_checkpoint = torch.load(
            args.teacher, map_location="cpu", weights_only=False)
        teacher_config = teacher_checkpoint["config"]
        teacher = CRNN_N(
            n_a=C.N_A, n_b=C.N_B,
            ch=tuple(teacher_config["ch"]),
            gru=teacher_config["gru"],
            gru_layers=teacher_config["gru_layers"],
        ).to(device)
        teacher.load_state_dict(teacher_checkpoint["model"])
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
        print(f"teacher={args.teacher} kd_weight={args.kd_weight} "
              f"temperature={args.temperature}", flush=True)
    n_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"parameters={n_params:,}", flush=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4)

    def lr_factor(epoch):
        if epoch < 3:
            return float(epoch + 1) / 3.0
        progress = (epoch - 3) / max(1, args.epochs - 3)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    positives = targets_word.mean(axis=0)
    word_pos_weight = torch.from_numpy(np.clip(
        (1.0 - positives) / np.maximum(positives, 1e-4), 1.0, 12.0)
        .astype(np.float32)).to(device)
    word_importance = torch.from_numpy(np.asarray([
        args.keyword_weight if token in SCHEMA_KEYWORDS else 1.0
        for token in vocabulary
    ], dtype=np.float32)).to(device)

    output = Path(C.OUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    history = []
    best, best_state = None, None
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = seen = 0
        for features, intent, slot, _, target_words in loader:
            features = features.to(device, non_blocking=True)
            intent = intent.to(device, non_blocking=True)
            slot = slot.to(device, non_blocking=True)
            target_words = target_words.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(features)
            logits_a, logits_b = outputs[:2]
            if args.structured_slots:
                logits_local, logits_words = outputs[2], outputs[3]
            else:
                logits_local, logits_words = None, outputs[2]
            supervised = (
                F.cross_entropy(logits_a, intent, label_smoothing=0.03)
                + F.cross_entropy(logits_b, slot, label_smoothing=0.02))
            if logits_local is not None:
                slotted = slot != C.N_VALUE
                if bool(slotted.any()):
                    group = intent[slotted] - len(C.FIXED)
                    grouped_logits = logits_local[slotted].reshape(
                        -1, len(C.SLOTTED), 3)
                    selected = grouped_logits[
                        torch.arange(len(group), device=device), group]
                    local_target = slot[slotted] % 3
                    supervised = supervised + args.local_slot_weight * F.cross_entropy(
                        selected, local_target, label_smoothing=0.02)
            if teacher is None:
                loss = supervised
            else:
                with torch.inference_mode():
                    teacher_a, teacher_b = teacher(features)
                distilled = (
                    distillation_loss(logits_a, teacher_a, args.temperature)
                    + distillation_loss(logits_b, teacher_b, args.temperature))
                loss = ((1.0 - args.kd_weight) * supervised
                        + args.kd_weight * distilled)
            if args.aux_weight > 0 and logits_words is not None:
                word_loss = F.binary_cross_entropy_with_logits(
                    logits_words, target_words, pos_weight=word_pos_weight,
                    reduction="none")
                loss = loss + args.aux_weight * (
                    word_loss * word_importance[None, :]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * len(features)
            seen += len(features)
        scheduler.step()
        model.eval()
        metrics = evaluate(model, test, test_features, target_a, target_b,
                           tune_features, device)
        deployment_metrics = evaluate(
            model, test, deployment_features, target_a, target_b,
            tune_features, device)
        metrics["deployment_e2e"] = deployment_metrics["overall_e2e"]
        metrics["deployment_real_e2e"] = deployment_metrics["real_e2e"]
        metrics.update(
            epoch=epoch, train_loss=round(total / seen, 6),
            elapsed_s=round(time.time() - started, 1))
        metrics["selection_score"] = selection_score(metrics)
        history.append(metrics)
        state = copy.deepcopy(model.state_dict())
        checkpoint = {
            "model": state,
            "config": {
                "architecture": ("temporal_attention_structured_v2"
                                 if args.structured_slots
                                 else "temporal_attention_v1"),
                "nframes": 250, "width": args.width,
                "gru_layers": args.gru_layers,
                "auxiliary_words": len(vocabulary),
                "structured_slots": args.structured_slots,
                "deployment_frontend_p": args.deployment_frontend_p,
            },
            "vocabulary": vocabulary,
            "n_params": n_params,
            "arch": ("temporal_attention_structured_v2"
                     if args.structured_slots else "temporal_attention_v1"),
            "n_frames": 250,
            "tag": args.tag,
            "epoch": epoch,
            "selection": metrics,
            "recipe": ("time-preserving attention with optional transcript "
                       f"supervision, CRNN distillation, and {args.curriculum} curriculum"),
        }
        torch.save(checkpoint, output / f"vcm_attention_{args.tag}_ep{epoch:02d}.pt")
        if best is None or metrics["selection_score"] > best["selection_score"]:
            best = metrics
            best_state = state
        report = {
            "protocol": "TEST + tune-OOS selection only; HOLDOUT not loaded",
            "args": vars(args), "groups": groups,
            "vocabulary": vocabulary, "n_params": n_params,
            "best": best, "history": history,
        }
        with open(output / f"train_attention_{args.tag}.json", "w") as handle:
            json.dump(report, handle, indent=2)
        print("VALID " + json.dumps(metrics, sort_keys=True), flush=True)

    final = {
        "model": best_state,
        "config": {
            "architecture": ("temporal_attention_structured_v2"
                             if args.structured_slots
                             else "temporal_attention_v1"),
            "nframes": 250, "width": args.width,
            "gru_layers": args.gru_layers,
            "auxiliary_words": len(vocabulary),
            "structured_slots": args.structured_slots,
            "deployment_frontend_p": args.deployment_frontend_p,
        },
        "vocabulary": vocabulary,
        "n_params": n_params,
        "arch": ("temporal_attention_structured_v2"
                 if args.structured_slots else "temporal_attention_v1"),
        "n_frames": 250,
        "tag": args.tag,
        "selection": best,
        "recipe": ("time-preserving attention with optional transcript "
                   f"supervision, CRNN distillation, and {args.curriculum} curriculum"),
    }
    torch.save(final, output / f"vcm_attention_{args.tag}.pt")
    print("BEST " + json.dumps(best, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
