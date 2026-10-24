"""Screen same-cost weight soups of aligned E8/X5/X6 VCM checkpoints.

All three checkpoints have the same CRNN architecture and seed (301), so their
neurons are aligned.  Interpolating their weights produces one ordinary model:
the parameter count and inference graph are unchanged.  Candidate selection is
performed on the historical TEST-as-validation split plus tune_oos; HOLDOUT and
the final OOS test are deliberately not read here.
"""
from __future__ import annotations

import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import config as C
from battery import CRNN_N, _need, wav_to_mel
from eval_ens5c import load_split
from train_centered import center_crop, decode_wav


ROOT = C.OUT_DIR
CHECKPOINTS = {
    "e8": os.path.join(ROOT, "vcm_crnn_E8_gru_s301.pt"),
    "reverb": os.path.join(ROOT, "vcm_crnn_X5_reverb.pt"),
    "babble": os.path.join(ROOT, "vcm_crnn_X6_babble.pt"),
}


def load_oos(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def build_features(df: pd.DataFrame, n_frames: int) -> np.ndarray:
    need = _need(n_frames)
    out = np.empty((len(df), 1, C.N_MELS, n_frames), dtype=np.float32)
    t0 = time.time()
    for i in range(len(df)):
        audio = df["audio"].iloc[i]
        raw = audio["bytes"] if isinstance(audio, dict) else audio
        try:
            wav = decode_wav(raw)
        except Exception:
            wav = np.zeros(need, dtype=np.float32)
        out[i, 0] = wav_to_mel(center_crop(wav, n=need), n_frames)
        if (i + 1) % 500 == 0:
            print(f"features {i + 1}/{len(df)} ({time.time() - t0:.0f}s)", flush=True)
    return out


def labels(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ya, yb = [], []
    for command, value in zip(df["command"], df["slot_value"]):
        if command not in C.INTENT2IDX:
            ya.append(C.N_INTENT)
            yb.append(C.N_VALUE)
            continue
        ya.append(C.INTENT2IDX[command])
        token = C.PAIR2TOKEN.get((command, value), "")
        yb.append(C.VALUE2IDX.get(token, C.N_VALUE))
    return np.asarray(ya), np.asarray(yb)


def load_objects():
    objects = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in CHECKPOINTS.items()
    }
    keys = list(objects["e8"]["model"])
    for name, obj in objects.items():
        if list(obj["model"]) != keys:
            raise RuntimeError(f"state keys differ for {name}")
        if obj.get("n_params") != 3_192_807:
            raise RuntimeError(f"unexpected parameter count for {name}: {obj.get('n_params')}")
    return objects


def soup_state(objects, weights):
    result = {}
    for key, ref in objects["e8"]["model"].items():
        if not torch.is_floating_point(ref):
            # BatchNorm counters do not affect inference predictions.
            result[key] = ref.clone()
            continue
        value = torch.zeros_like(ref)
        for name, weight in weights.items():
            value.add_(objects[name]["model"][key], alpha=float(weight))
        result[key] = value
    return result


def make_model(state):
    model = CRNN_N(n_a=C.N_A, n_b=C.N_B, ch=(32, 64, 128), gru=256, gru_layers=3)
    model.load_state_dict(state)
    model.eval()
    return model


def predict(model, features, batch=64):
    pred_a, pred_b = [], []
    device = next(model.parameters()).device
    with torch.inference_mode():
        for start in range(0, len(features), batch):
            inputs = torch.from_numpy(features[start:start + batch]).to(device)
            logits_a, logits_b = model(inputs)
            pred_a.append(logits_a.argmax(1).cpu().numpy())
            pred_b.append(logits_b.argmax(1).cpu().numpy())
    return np.concatenate(pred_a), np.concatenate(pred_b)


def command_metrics(df, ya, yb, pa, pb):
    intent = pa == ya
    e2e = intent & ((pb == yb) | (yb == C.N_VALUE))
    real = df["is_synthetic"].fillna(1).astype(int).to_numpy() == 0
    synthetic = ~real
    return {
        "overall_intent": round(float(intent.mean() * 100), 3),
        "overall_e2e": round(float(e2e.mean() * 100), 3),
        "real_e2e": round(float(e2e[real].mean() * 100), 3),
        "synthetic_e2e": round(float(e2e[synthetic].mean() * 100), 3),
        "n": int(len(df)),
        "n_real": int(real.sum()),
        "n_synthetic": int(synthetic.sum()),
    }


def candidate_grid():
    # Fine pairwise interpolation around the shipped and babble endpoints.
    for i in range(21):
        babble = i / 20
        yield f"e8_babble_{babble:.2f}", {"e8": 1 - babble, "babble": babble}
    # Coarser simplex adds the reverb solution without an excessive CPU sweep.
    for e8_i in range(0, 5):
        for reverb_i in range(0, 5 - e8_i):
            babble_i = 4 - e8_i - reverb_i
            weights = {"e8": e8_i / 4, "reverb": reverb_i / 4,
                       "babble": babble_i / 4}
            yield (f"tri_{e8_i}_{reverb_i}_{babble_i}",
                   {k: v for k, v in weights.items() if v})


def main():
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    torch.set_num_interop_threads(1)
    objects = load_objects()
    n_frames = 250

    test_df = load_split("test")
    tune_df = load_oos(os.path.join(C.HF_DIR, "synthetic_negatives", "tune_oos-*.parquet"))
    print(f"building TEST features ({len(test_df)})", flush=True)
    test_x = build_features(test_df, n_frames)
    print(f"building tune-OOS features ({len(tune_df)})", flush=True)
    tune_x = build_features(tune_df, n_frames)
    test_ya, test_yb = labels(test_df)

    results = []
    seen = set()
    for name, weights in candidate_grid():
        signature = tuple(sorted(weights.items()))
        if signature in seen:
            continue
        seen.add(signature)
        state = soup_state(objects, weights)
        model = make_model(state)
        test_pa, test_pb = predict(model, test_x)
        tune_pa, _ = predict(model, tune_x)
        metrics = command_metrics(test_df, test_ya, test_yb, test_pa, test_pb)
        metrics["tune_oos_reject"] = round(float((tune_pa == C.N_INTENT).mean() * 100), 3)
        record = {"name": name, "weights": weights, "test": metrics}
        results.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    feasible = [r for r in results if r["test"]["overall_e2e"] >= 90.0]
    # Real E2E is the primary objective.  OOS and overall E2E break ties.
    best = max(feasible, key=lambda r: (r["test"]["real_e2e"],
                                        r["test"]["tune_oos_reject"],
                                        r["test"]["overall_e2e"]))
    final_state = soup_state(objects, best["weights"])
    output = os.path.join(ROOT, "vcm_crnn_soup_candidate.pt")
    template = objects["e8"]
    torch.save({
        "model": final_state,
        "config": dict(template["config"], tag="soup_candidate"),
        "intents": template["intents"],
        "value_tokens": template["value_tokens"],
        "n_params": 3_192_807,
        "arch": "battery",
        "n_frames": 250,
        "tag": "soup_candidate",
        "recipe": "aligned weight soup; weights=" + json.dumps(best["weights"], sort_keys=True),
        "selection": best,
    }, output)
    report = {"protocol": "selected on historical TEST-as-validation + tune_oos only; HOLDOUT untouched",
              "best": best, "candidates": results, "checkpoint": output}
    report_path = os.path.join(ROOT, "weight_soup_screen.json")
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)
    print(f"BEST {json.dumps(best, sort_keys=True)}", flush=True)
    print(f"saved {output} and {report_path}", flush=True)


if __name__ == "__main__":
    main()
