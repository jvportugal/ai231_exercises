#!/usr/bin/env python3
"""Evaluate a robust checkpoint exactly like the multi-placement Pi runtime."""
import argparse
import json
import math
import os

import librosa
import numpy as np
import torch

import train_robust as T


def candidate_windows(audio):
    y = np.asarray(audio, dtype=np.float32)
    y = y - float(y.mean())
    crop, voiced_len = T.vad_crop(y)
    if voiced_len < 0.16:
        crop = y
    if len(crop) >= T.WINDOW_SAMPLES:
        gap = len(crop) - T.WINDOW_SAMPLES
        starts = sorted(set((0, gap // 2, gap)))
        return [crop[s:s + T.WINDOW_SAMPLES].astype(np.float32) for s in starts]
    gap = T.WINDOW_SAMPLES - len(crop)
    offsets = sorted(set((0, gap // 2, gap)))
    result = []
    for offset in offsets:
        window = np.zeros(T.WINDOW_SAMPLES, dtype=np.float32)
        window[offset:offset + len(crop)] = crop
        result.append(window)
    return result


def score_candidates(model, candidates, batch=256):
    features = []
    groups = []
    for group, audio in enumerate(candidates):
        for window in candidate_windows(audio):
            features.append(torch.from_numpy(T.logmel_channels(window)))
            groups.append(group)
    raw = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch):
            x = torch.stack(features[start:start + batch])
            raw.extend(torch.sigmoid(model(x)).numpy().tolist())
    scores = np.full(len(candidates), -np.inf, dtype=np.float32)
    np.maximum.at(scores, np.asarray(groups), np.asarray(raw, dtype=np.float32))
    return scores


def build_stress(positive_records, count=400, external_noise=None,
                 real_fan_count=400):
    rng = np.random.default_rng(99017)
    audio, labels, categories = [], [], []
    external_noise = list(external_noise or [])
    noise_cache = {}

    def real_fan(index, samples=T.WINDOW_SAMPLES):
        path = external_noise[index % len(external_noise)]
        if path not in noise_cache:
            noise_cache[path] = T.load_mono(path)
        return T.noise_window(noise_cache[path], rng, samples)
    for i, rec in enumerate(positive_records[:min(count, len(positive_records))]):
        y = T.load_mono(os.path.join(T.WAVDIR, rec["stem"] + ".wav"))
        crop, voiced_len = T.vad_crop(y)
        if voiced_len >= 0.16:
            y = crop
        case = i % 5
        if case == 0:
            y = librosa.effects.time_stretch(y, rate=0.80)
            y = T.fit_window(y, rng, False) * 10 ** (-20 / 20)
            y = T.mix_at_snr(y, T.synth_fan(rng), 5.0)
            category = "positive_far_fan"
        elif case == 1:
            y = T.fit_window(y, rng, False) * 10 ** (16 / 20)
            y = T.proximity_filter(y, rng)
            y = np.clip(y, -0.30, 0.30) / 0.30
            category = "positive_close_clip"
        elif case == 2:
            y = librosa.effects.pitch_shift(y=y, sr=T.SR, n_steps=-3.0)
            y = T.warp_halves(y, rng)
            y = T.fit_window(y, rng, False)
            category = "positive_low_pitch_prosody"
        elif case == 3:
            y = librosa.effects.pitch_shift(y=y, sr=T.SR, n_steps=3.0)
            y = T.warp_halves(y, rng)
            y = T.fit_window(y, rng, False)
            category = "positive_high_pitch_prosody"
        else:
            y = T.room_reverb(T.fit_window(y, rng, False), rng)
            if external_noise:
                y = T.mix_at_snr(y, real_fan(i), 12.0)
                category = "positive_real_fan"
            else:
                y = T.mix_at_snr(y, T.synth_fan(rng), 12.0)
                category = "positive_room_fan"
        audio.append(np.clip(y, -1, 1).astype(np.float32))
        labels.append(1)
        categories.append(category)
    for i in range(600):
        # Two seconds matches the runtime's maximum wake candidate, so the
        # start/center/end maximum is included in the false-trigger test.
        y = T.synth_fan(rng, 2 * T.WINDOW_SAMPLES)
        if i % 3 == 0:
            y += T.synth_fan(rng, 2 * T.WINDOW_SAMPLES)
        if i % 5 == 0:
            y = T.proximity_filter(y, rng)
        audio.append(np.clip(y, -1, 1).astype(np.float32))
        labels.append(0)
        categories.append("fan_only")
    for i in range(real_fan_count if external_noise else 0):
        y = real_fan(i, 2 * T.WINDOW_SAMPLES)
        level = 10.0 ** (float(rng.uniform(-38.0, -8.0)) / 20.0)
        audio.append(np.clip(y * level, -1, 1).astype(np.float32))
        labels.append(0)
        categories.append("real_fan_only")
    return audio, np.asarray(labels), categories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(os.path.dirname(__file__), "checkpoints",
                             "destiny_robust_v2.pt"),
    )
    parser.add_argument("--report", default="deployment_eval_robust_v2.json")
    parser.add_argument("--target-fpr", type=float, default=0.005)
    parser.add_argument("--noise-dir", default=None,
                        help="recursive directory of held-out real fan/HVAC recordings")
    parser.add_argument("--wav-dir", default=None,
                        help="prepared split WAV directory (default: wakeword/data/wavs)")
    parser.add_argument("--allow-missing-audio", action="store_true")
    parser.add_argument("--noise-holdout-fraction", type=float, default=0.20)
    parser.add_argument("--real-fan-count", type=int, default=400)
    parser.add_argument("--torch-threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.torch_threads)
    if args.wav_dir:
        T.WAVDIR = os.path.abspath(os.path.expanduser(args.wav_dir))

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = T.RobustWakeCNN()
    model.load_state_dict(checkpoint["model_state"])

    pos_records = T.available_records(
        T.load_records("pos_test"), args.allow_missing_audio)
    neg_records = T.available_records(
        T.load_records("oos_test"), args.allow_missing_audio)
    _, external_test = T.discover_external_noise(
        args.noise_dir, args.noise_holdout_fraction)
    print("held-out external fan files=%d" % len(external_test), flush=True)
    print("loading clean held-out candidates...", flush=True)
    pos_audio = [T.load_mono(os.path.join(T.WAVDIR, r["stem"] + ".wav"))
                 for r in pos_records]
    neg_audio = [T.load_mono(os.path.join(T.WAVDIR, r["stem"] + ".wav"))
                 for r in neg_records]
    pos_scores = score_candidates(model, pos_audio)
    neg_scores = score_candidates(model, neg_audio)

    print("building and scoring robustness candidates...", flush=True)
    stress_audio, stress_y, stress_cat = build_stress(
        pos_records, external_noise=external_test,
        real_fan_count=args.real_fan_count)
    stress_scores = score_candidates(model, stress_audio)
    combined_neg = np.concatenate((neg_scores, stress_scores[stress_y == 0]))
    allowed = int(math.floor(args.target_fpr * len(combined_neg)))
    descending = np.sort(combined_neg)[::-1]
    threshold = float(np.nextafter(
        np.float32(descending[allowed]), np.float32(np.inf)))

    categories = [r.get("category", "other") for r in neg_records] + stress_cat
    all_scores = np.concatenate((neg_scores, stress_scores))
    by_category = {}
    for category in sorted(set(categories)):
        values = np.asarray([score for score, cat in zip(all_scores, categories)
                             if cat == category])
        positive = category.startswith("positive_")
        by_category[category] = {
            "n": len(values),
            "recall" if positive else "false_positive_rate":
                float(np.mean(values >= threshold)),
        }
    operating_points = {}
    for candidate_threshold in (0.85, 0.875, 0.90, 0.925, 0.95):
        operating_points[str(candidate_threshold)] = {
            "clean_recall": float(np.mean(pos_scores >= candidate_threshold)),
            "clean_oos_false_positive_rate": float(np.mean(
                neg_scores >= candidate_threshold)),
            "stress_recall": float(np.mean(
                stress_scores[stress_y == 1] >= candidate_threshold)),
            "fan_false_positive_rate": float(np.mean(
                stress_scores[stress_y == 0] >= candidate_threshold)),
        }
    report = {
        "checkpoint": args.checkpoint,
        "threshold": threshold,
        "target_combined_false_positive_rate": args.target_fpr,
        "clean_recall": float(np.mean(pos_scores >= threshold)),
        "clean_oos_false_positive_rate": float(np.mean(neg_scores >= threshold)),
        "stress_recall": float(np.mean(stress_scores[stress_y == 1] >= threshold)),
        "fan_false_positive_rate": float(np.mean(stress_scores[stress_y == 0] >= threshold)),
        "by_category": by_category,
        "operating_points": operating_points,
        "placement_policy": "maximum of VAD-cropped start/center/end windows",
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
