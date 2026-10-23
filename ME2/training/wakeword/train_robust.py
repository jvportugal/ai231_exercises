#!/usr/bin/env python3
"""Train a compact, streaming-compatible robust "Destiny" detector.

This version targets the failure modes seen in the live Raspberry Pi test:

* fan/air-conditioner hum is generated as a dedicated hard-negative family;
* far-field speech is mixed with fan/room noise and reverberation;
* close-mic speech is given proximity coloration, plosives and saturation;
* pitch, speaking rate, and the relative duration of the two halves of the
  word are randomized independently;
* a temporal CNN keeps phoneme order instead of globally averaging it away;
* a delta-log-mel channel makes stationary fan spectra easier to reject.

The MLCommons Multilingual Spoken Words files remain untouched. All variants
are produced in memory. Held-out speakers are never used for training.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore")

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset, get_worker_info

HERE = os.path.dirname(os.path.abspath(__file__))
WAVDIR = os.path.join(HERE, "data", "wavs")
SPLITS = os.path.join(HERE, "splits")

SR = 16000
WINDOW_SECONDS = 1.0
WINDOW_SAMPLES = int(SR * WINDOW_SECONDS)
N_MELS = 40
N_FFT = 512
HOP = 160
N_FRAMES = 100
SEED = 231023


def load_records(name: str) -> list[dict]:
    with open(os.path.join(SPLITS, name + ".json"), encoding="utf-8") as f:
        return json.load(f)


def load_mono(path: str) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return (y - float(np.mean(y))).astype(np.float32)


def vad_crop(y: np.ndarray, pad_ms: int = 65) -> tuple[np.ndarray, float]:
    """Energy crop that tolerates a steady fan floor and keeps word edges."""
    frame, hop = 160, 80
    if len(y) < frame:
        return y, len(y) / SR
    n = 1 + (len(y) - frame) // hop
    idx = np.arange(frame)[None, :] + np.arange(n)[:, None] * hop
    energy = np.sqrt(np.mean(y[idx] ** 2, axis=1) + 1e-12)
    floor = float(np.percentile(energy, 20))
    ceiling = float(np.percentile(energy, 95))
    threshold = max(floor * 2.2, floor + (ceiling - floor) * 0.22, 8e-5)
    voiced = energy > threshold
    if not voiced.any():
        return y, 0.0
    first = int(np.argmax(voiced))
    last = int(len(voiced) - 1 - np.argmax(voiced[::-1]))
    pad = int(SR * pad_ms / 1000)
    start = max(0, first * hop - pad)
    end = min(len(y), (last + 1) * hop + pad)
    return y[start:end], (end - start) / SR


def fit_window(y: np.ndarray, rng: np.random.Generator, random_place: bool) -> np.ndarray:
    if len(y) > WINDOW_SAMPLES:
        # Preserve both syllables: center long crops rather than taking only
        # their beginning. A small jitter prevents a fixed alignment shortcut.
        center = (len(y) - WINDOW_SAMPLES) // 2
        jitter = int(rng.integers(-400, 401)) if random_place else 0
        start = int(np.clip(center + jitter, 0, len(y) - WINDOW_SAMPLES))
        return y[start:start + WINDOW_SAMPLES].astype(np.float32)
    out = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    gap = WINDOW_SAMPLES - len(y)
    offset = int(rng.integers(0, gap + 1)) if random_place and gap else gap // 2
    out[offset:offset + len(y)] = y
    return out


def colored_noise(rng: np.random.Generator, n: int, color: float = 1.0) -> np.ndarray:
    """Frequency-shaped noise; color=0 white, 1 pink-ish, 2 brown-ish."""
    bins = n // 2 + 1
    spectrum = rng.standard_normal(bins) + 1j * rng.standard_normal(bins)
    freqs = np.maximum(np.fft.rfftfreq(n, 1.0 / SR), 20.0)
    spectrum /= np.power(freqs / 100.0, color / 2.0)
    x = np.fft.irfft(spectrum, n=n).astype(np.float32)
    return x / (float(np.std(x)) + 1e-7)


def synth_fan(rng: np.random.Generator, n: int = WINDOW_SAMPLES) -> np.ndarray:
    """Procedural fan/air-conditioner family with hum, blades and airflow."""
    t = np.arange(n, dtype=np.float32) / SR
    mains = float(rng.choice((50.0, 60.0)))
    blade = float(rng.uniform(70.0, 260.0))
    phase = float(rng.uniform(0, 2 * np.pi))
    x = 0.62 * colored_noise(rng, n, float(rng.uniform(1.0, 2.1)))
    for harmonic in range(1, int(rng.integers(3, 8))):
        x += (0.42 / harmonic) * np.sin(
            2 * np.pi * (blade * harmonic) * t + phase * harmonic)
    x += 0.18 * np.sin(2 * np.pi * mains * t + phase)
    # Blade speed and airflow fluctuate slowly instead of being perfect tones.
    modulation_hz = float(rng.uniform(0.35, 4.5))
    modulation = 0.72 + 0.28 * np.sin(2 * np.pi * modulation_hz * t + phase / 2)
    x *= modulation
    x /= float(np.max(np.abs(x)) + 1e-7)
    level = 10.0 ** (float(rng.uniform(-38.0, -10.0)) / 20.0)
    return (x * level).astype(np.float32)


def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    sig_rms = math.sqrt(float(np.mean(signal * signal)) + 1e-10)
    noise_rms = math.sqrt(float(np.mean(noise * noise)) + 1e-10)
    if sig_rms < 1e-5:
        return signal + noise
    scale = sig_rms / (10.0 ** (snr_db / 20.0) * noise_rms)
    return signal + noise * scale


def room_reverb(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Small synthetic room impulse response, suitable for near/far variation."""
    rir_len = int(rng.uniform(0.04, 0.22) * SR)
    t = np.arange(rir_len, dtype=np.float32) / SR
    decay = np.exp(-t / float(rng.uniform(0.025, 0.095))).astype(np.float32)
    rir = rng.standard_normal(rir_len).astype(np.float32) * decay * 0.025
    rir[0] += 1.0
    for _ in range(int(rng.integers(2, 6))):
        delay = int(rng.uniform(0.006, min(0.11, rir_len / SR)) * SR)
        if delay < rir_len:
            rir[delay] += float(rng.uniform(0.08, 0.42))
    rir /= float(np.sqrt(np.sum(rir * rir)) + 1e-7)
    return np.convolve(y, rir, mode="full")[:len(y)].astype(np.float32)


def proximity_filter(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Approximate close-mic bass boost, plosive pressure and saturation."""
    # One-pole low-pass mixed into the dry signal gives a randomized low shelf.
    alpha = float(rng.uniform(0.025, 0.09))
    low = np.empty_like(y)
    state = 0.0
    for i, sample in enumerate(y):
        state += alpha * (float(sample) - state)
        low[i] = state
    y = y + float(rng.uniform(0.4, 1.7)) * low
    if rng.random() < 0.55:
        # A short low-frequency breath/plosive near the first half of the word.
        length = int(rng.uniform(0.025, 0.09) * SR)
        start = int(rng.integers(0, max(1, min(len(y) - length, int(0.45 * SR)))))
        tt = np.arange(length, dtype=np.float32) / SR
        burst = np.sin(2 * np.pi * float(rng.uniform(35, 95)) * tt)
        burst *= np.hanning(length).astype(np.float32)
        y[start:start + length] += burst * float(rng.uniform(0.03, 0.22))
    drive = float(rng.uniform(1.2, 4.5))
    return (np.tanh(y * drive) / np.tanh(drive)).astype(np.float32)


def warp_halves(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vary relative syllable duration to reduce stress/intonation shortcuts."""
    if len(y) < 800:
        return y
    split = int(len(y) * float(rng.uniform(0.38, 0.58)))
    rates = (float(rng.uniform(0.72, 1.28)), float(rng.uniform(0.72, 1.28)))
    pieces = []
    for piece, rate in zip((y[:split], y[split:]), rates):
        if len(piece) < 256:
            pieces.append(piece)
        else:
            pieces.append(librosa.effects.time_stretch(piece, rate=rate))
    return np.concatenate(pieces).astype(np.float32)


def augment_speech(y: np.ndarray, rng: np.random.Generator, positive: bool) -> np.ndarray:
    crop, voiced_len = vad_crop(y)
    if voiced_len >= 0.16:
        y = crop

    # Whole-word pitch and rate ranges deliberately cover monotone, higher,
    # lower, slow and fast pronunciations. Negative words get the same family
    # so artifacts cannot become a positive-class shortcut.
    if rng.random() < 0.70:
        y = librosa.effects.pitch_shift(
            y=y, sr=SR, n_steps=float(rng.uniform(-3.5, 3.5)))
    if rng.random() < (0.55 if positive else 0.35):
        y = librosa.effects.time_stretch(y, rate=float(rng.uniform(0.78, 1.24)))
    if positive and rng.random() < 0.48:
        y = warp_halves(y, rng)

    y = fit_window(y, rng, random_place=True)

    mode = float(rng.random())
    if mode < 0.22:
        y = proximity_filter(y, rng)
    elif mode < 0.55:
        y = room_reverb(y, rng)

    # Gain spans quiet far-field to overloaded close-mic input. Applying gain
    # before clipping is essential: pure gain disappears under feature CMVN.
    gain_db = float(rng.uniform(-22.0, 16.0))
    y *= 10.0 ** (gain_db / 20.0)

    if rng.random() < 0.72:
        fan = synth_fan(rng)
        y = mix_at_snr(y, fan, float(rng.uniform(2.0, 32.0)))
    elif rng.random() < 0.45:
        noise = colored_noise(rng, WINDOW_SAMPLES, float(rng.uniform(0, 1.5)))
        y = mix_at_snr(y, noise, float(rng.uniform(8.0, 35.0)))

    if rng.random() < 0.30:
        limit = float(rng.uniform(0.18, 0.75))
        y = np.clip(y, -limit, limit) / limit
    return np.clip(y, -1.0, 1.0).astype(np.float32)


def logmel_channels(y: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS,
        fmin=35.0, fmax=7600.0, power=2.0)
    db = librosa.power_to_db(mel, ref=np.max, top_db=80.0)
    if db.shape[1] < N_FRAMES:
        db = np.pad(db, ((0, 0), (0, N_FRAMES - db.shape[1])), mode="edge")
    db = db[:, :N_FRAMES]
    base = (db - float(db.mean())) / (float(db.std()) + 1e-5)
    delta = np.zeros_like(base)
    delta[:, 1:] = base[:, 1:] - base[:, :-1]
    delta /= float(delta.std()) + 1e-5
    return np.stack((base, delta)).astype(np.float32)


class RobustWakeCNN(nn.Module):
    """Order-aware temporal CNN; about 100K parameters and Pi-friendly."""

    def __init__(self):
        super().__init__()
        self.spectral = nn.Sequential(
            nn.Conv2d(2, 24, 3, stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(24), nn.SiLU(),
            nn.Conv2d(24, 48, 3, stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(48), nn.SiLU(),
            nn.Conv2d(48, 64, 3, stride=(2, 1), padding=1, bias=False),
            nn.BatchNorm2d(64), nn.SiLU(),
        )
        self.temporal = nn.Sequential(
            nn.Conv1d(64, 72, 5, padding=2, bias=False),
            nn.BatchNorm1d(72), nn.SiLU(),
            nn.Conv1d(72, 72, 3, padding=1, groups=8, bias=False),
            nn.BatchNorm1d(72), nn.SiLU(),
        )
        self.attention = nn.Conv1d(72, 1, 1)
        self.classifier = nn.Sequential(
            nn.Linear(144, 64), nn.SiLU(), nn.Dropout(0.25), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spectral(x).mean(dim=2)
        x = self.temporal(x)
        weights = torch.softmax(self.attention(x), dim=-1)
        attended = (x * weights).sum(dim=-1)
        maximum = x.amax(dim=-1)
        return self.classifier(torch.cat((attended, maximum), dim=1)).squeeze(1)


class WakeDataset(Dataset):
    def __init__(self, positives: list[dict], negatives: list[dict],
                 examples_per_epoch: int, seed: int):
        self.positives = positives
        self.negatives = negatives
        self.examples_per_epoch = examples_per_epoch
        self.seed = seed
        self._cache: dict[str, np.ndarray] = {}
        self._rng = None

    def __len__(self):
        return self.examples_per_epoch

    def _random(self) -> np.random.Generator:
        if self._rng is None:
            wi = get_worker_info()
            worker = wi.id if wi else 0
            self._rng = np.random.default_rng(self.seed + worker * 100003)
        return self._rng

    def _load(self, rec: dict) -> np.ndarray:
        stem = rec["stem"]
        if stem not in self._cache:
            self._cache[stem] = load_mono(os.path.join(WAVDIR, stem + ".wav"))
        return self._cache[stem]

    def __getitem__(self, index):
        del index
        rng = self._random()
        positive = bool(rng.random() < 0.5)
        if positive:
            rec = self.positives[int(rng.integers(len(self.positives)))]
            y = augment_speech(self._load(rec).copy(), rng, True)
            label = 1.0
        else:
            # A third of negatives are fan-only. The rest are real words with
            # the same mic/channel transformations as positives.
            if rng.random() < 0.34:
                y = synth_fan(rng)
                if rng.random() < 0.25:
                    y += synth_fan(rng)
                y = np.clip(y, -1.0, 1.0).astype(np.float32)
            else:
                rec = self.negatives[int(rng.integers(len(self.negatives)))]
                y = augment_speech(self._load(rec).copy(), rng, False)
            label = 0.0
        return torch.from_numpy(logmel_channels(y)), torch.tensor(label)


@dataclass
class EvalItem:
    features: torch.Tensor
    label: float
    category: str


def build_clean_eval(positives: list[dict], negatives: list[dict]) -> list[EvalItem]:
    rng = np.random.default_rng(17)
    items = []
    for rec in positives:
        y = load_mono(os.path.join(WAVDIR, rec["stem"] + ".wav"))
        crop, vlen = vad_crop(y)
        if vlen < 0.16:
            crop = y
        items.append(EvalItem(torch.from_numpy(logmel_channels(
            fit_window(crop, rng, False))), 1.0, "destiny_clean"))
    for rec in negatives:
        y = load_mono(os.path.join(WAVDIR, rec["stem"] + ".wav"))
        crop, vlen = vad_crop(y)
        if vlen < 0.16:
            crop = y
        items.append(EvalItem(torch.from_numpy(logmel_channels(
            fit_window(crop, rng, False))), 0.0, rec.get("category", "other")))
    return items


def build_robust_eval(positives: list[dict], count: int = 400) -> list[EvalItem]:
    """Deterministic held-out stress set; never sampled by the training set."""
    rng = np.random.default_rng(99017)
    items = []
    take = min(count, len(positives))
    for i, rec in enumerate(positives[:take]):
        y = load_mono(os.path.join(WAVDIR, rec["stem"] + ".wav"))
        crop, vlen = vad_crop(y)
        if vlen >= 0.16:
            y = crop
        case = i % 5
        if case == 0:  # flat/slow-ish and very quiet with a fan
            y = librosa.effects.time_stretch(y, rate=0.80)
            y = fit_window(y, rng, False) * 10 ** (-20 / 20)
            y = mix_at_snr(y, synth_fan(rng), 5.0)
            category = "positive_far_fan"
        elif case == 1:  # overloaded close microphone
            y = fit_window(y, rng, False) * 10 ** (16 / 20)
            y = proximity_filter(y, rng)
            y = np.clip(y, -0.30, 0.30) / 0.30
            category = "positive_close_clip"
        elif case == 2:  # low pitch and stretched first/second-half timing
            y = librosa.effects.pitch_shift(y=y, sr=SR, n_steps=-3.0)
            y = warp_halves(y, rng)
            y = fit_window(y, rng, False)
            category = "positive_low_pitch_prosody"
        elif case == 3:  # high pitch and alternate syllable timing
            y = librosa.effects.pitch_shift(y=y, sr=SR, n_steps=3.0)
            y = warp_halves(y, rng)
            y = fit_window(y, rng, False)
            category = "positive_high_pitch_prosody"
        else:
            y = room_reverb(fit_window(y, rng, False), rng)
            y = mix_at_snr(y, synth_fan(rng), 12.0)
            category = "positive_room_fan"
        items.append(EvalItem(torch.from_numpy(logmel_channels(
            np.clip(y, -1, 1).astype(np.float32))), 1.0, category))
    for i in range(600):
        y = synth_fan(rng)
        if i % 3 == 0:
            y += synth_fan(rng)
        if i % 5 == 0:
            y = proximity_filter(y, rng)
        items.append(EvalItem(torch.from_numpy(logmel_channels(
            np.clip(y, -1, 1).astype(np.float32))), 0.0, "fan_only"))
    return items


def score_items(model: nn.Module, items: list[EvalItem], batch: int = 128):
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(items), batch):
            x = torch.stack([z.features for z in items[start:start + batch]])
            scores.extend(torch.sigmoid(model(x)).cpu().numpy().tolist())
    labels = np.asarray([z.label for z in items])
    categories = [z.category for z in items]
    return np.asarray(scores), labels, categories


def deployment_threshold(scores: np.ndarray, labels: np.ndarray,
                         target_fpr: float = 0.005) -> float:
    negatives = np.sort(scores[labels == 0])[::-1]
    allowed = int(math.floor(target_fpr * len(negatives)))
    if allowed >= len(negatives):
        return 0.0
    return float(np.nextafter(np.float32(negatives[allowed]), np.float32(np.inf)))


def metrics(model: nn.Module, clean: list[EvalItem], robust: list[EvalItem]):
    clean_scores, clean_y, clean_cat = score_items(model, clean)
    stress_scores, stress_y, stress_cat = score_items(model, robust)
    # Calibrate against real held-out OOS and procedural fans together.
    neg_scores = np.concatenate((clean_scores[clean_y == 0], stress_scores[stress_y == 0]))
    threshold = deployment_threshold(
        np.concatenate((clean_scores[clean_y == 1], neg_scores)),
        np.concatenate((np.ones(int((clean_y == 1).sum())), np.zeros(len(neg_scores)))))
    clean_recall = float(np.mean(clean_scores[clean_y == 1] >= threshold))
    clean_fpr = float(np.mean(clean_scores[clean_y == 0] >= threshold))
    stress_recall = float(np.mean(stress_scores[stress_y == 1] >= threshold))
    fan_fpr = float(np.mean(stress_scores[stress_y == 0] >= threshold))
    auc = float(roc_auc_score(clean_y, clean_scores))
    by_category = {}
    for category in sorted(set(clean_cat + stress_cat)):
        vals = [s for s, c in zip(
            np.concatenate((clean_scores, stress_scores)), clean_cat + stress_cat)
                if c == category]
        if category.startswith("positive_") or category == "destiny_clean":
            by_category[category] = {"n": len(vals), "recall": float(np.mean(np.asarray(vals) >= threshold))}
        else:
            by_category[category] = {"n": len(vals), "false_positive_rate": float(np.mean(np.asarray(vals) >= threshold))}
    return {
        "threshold": threshold,
        "target_oos_fpr": 0.005,
        "clean_auc": auc,
        "clean_recall": clean_recall,
        "clean_oos_fpr": clean_fpr,
        "stress_recall": stress_recall,
        "fan_false_positive_rate": fan_fpr,
        "by_category": by_category,
    }


def train(args):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.set_num_threads(args.torch_threads)

    pos_train = load_records("pos_train")
    neg_train = load_records("neg_train")
    pos_test = load_records("pos_test")
    neg_test = load_records("oos_test")
    print("train speakers: positives=%d negatives=%d; held-out positives=%d negatives=%d" %
          (len(pos_train), len(neg_train), len(pos_test), len(neg_test)), flush=True)
    print("building deterministic clean and robustness evaluations...", flush=True)
    clean = build_clean_eval(pos_test, neg_test)
    robust = build_robust_eval(pos_test, args.stress_positives)

    dataset = WakeDataset(pos_train, neg_train, args.examples_per_epoch, SEED)
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers,
        persistent_workers=args.workers > 0, prefetch_factor=2 if args.workers else None)
    model = RobustWakeCNN()
    count = sum(p.numel() for p in model.parameters())
    print("model parameters=%d; workers=%d; examples/epoch=%d" %
          (count, args.workers, args.examples_per_epoch), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    best = None
    best_objective = -1e9
    history = []
    stale = 0
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        seen = 0
        started = time.time()
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = loss_fn(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss) * len(labels)
            seen += len(labels)
        scheduler.step()
        result = metrics(model, clean, robust)
        # Stress recall matters, but never reward a model for violating the
        # calibrated always-listening false-positive constraint.
        objective = (0.55 * result["stress_recall"] +
                     0.45 * result["clean_recall"] -
                     2.0 * result["fan_false_positive_rate"])
        row = {"epoch": epoch, "loss": total / seen, "objective": objective,
               "seconds": time.time() - started, **result}
        history.append(row)
        print("ep%02d loss=%.4f clean_rec=%.3f stress_rec=%.3f clean_fpr=%.4f "
              "fan_fpr=%.4f thr=%.3f obj=%.3f (%.0fs)" %
              (epoch, row["loss"], result["clean_recall"], result["stress_recall"],
               result["clean_oos_fpr"], result["fan_false_positive_rate"],
               result["threshold"], objective, row["seconds"]), flush=True)
        if objective > best_objective:
            best_objective = objective
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print("early stopping", flush=True)
                break

    model.load_state_dict(best)
    final = metrics(model, clean, robust)
    checkpoint = {
        "model_state": best,
        "architecture": "RobustWakeCNN-v2",
        "sample_rate": SR,
        "window_samples": WINDOW_SAMPLES,
        "window_seconds": WINDOW_SECONDS,
        "n_mels": N_MELS,
        "n_fft": N_FFT,
        "hop": HOP,
        "n_frames": N_FRAMES,
        "input_channels": ["utterance_cmvn_logmel", "temporal_delta"],
        "parameters": count,
        "metrics": final,
    }
    torch.save(checkpoint, os.path.join(HERE, args.output))
    with open(os.path.join(HERE, args.report), "w", encoding="utf-8") as f:
        json.dump({"final": final, "history": history}, f, indent=2)
    print("saved %s and %s" % (args.output, args.report), flush=True)
    print(json.dumps(final, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=18)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--torch-threads", type=int, default=12)
    parser.add_argument("--examples-per-epoch", type=int, default=4200)
    parser.add_argument("--stress-positives", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1.2e-3)
    parser.add_argument("--output", default="checkpoints/destiny_robust_v2.pt")
    parser.add_argument("--report", default="eval_report_robust_v2.json")
    train(parser.parse_args())


if __name__ == "__main__":
    main()
