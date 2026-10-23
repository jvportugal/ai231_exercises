"""Centered 5-seed ensemble eval.
Averages softmax probs across the 5 centered cseed checkpoints, scores TEST /
HOLDOUT / OOS-TEST, AND breaks TEST+HOLDOUT out as overall/synthetic/real with
the real-by-duration table (via centered feature extraction).
"""
import glob, io, json, time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.io import wavfile
from mel import wav_to_mel, _get_fb
import config as C
from train import TwoHeadCNN

SEEDS = [f"vcm_twohead_cseed{s}.pt" for s in (101, 102, 103, 104, 105)]
DATA = C.HF_DIR + "/data"
NEED = 1 + (C.N_FRAMES - 1) * C.HOP
HOP = C.HOP


def load_model(path, device):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    sd = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    m = TwoHeadCNN(C.N_A, C.N_B)
    m.load_state_dict(sd)
    m.eval()
    return m.to(device)


def decode(b):
    sr, d = wavfile.read(io.BytesIO(b))
    if d.dtype == np.int16:
        d = d.astype(np.float32) / 32768.0
    else:
        d = d.astype(np.float32)
    if d.ndim > 1:
        d = d[:, 0]
    if sr != C.SR:
        n = int(len(d) * C.SR / sr)
        d = np.interp(np.linspace(0, len(d) - 1, n), np.arange(len(d)), d).astype(np.float32)
    return d


def center_crop(d):
    if len(d) <= NEED:
        out = np.zeros(NEED, np.float32)
        out[:len(d)] = d
        return out
    nwin = (len(d) - NEED) // HOP + 1
    cs = np.concatenate([[0.0], np.cumsum(d.astype(np.float64) ** 2)])
    starts = np.arange(nwin) * HOP
    ends = starts + NEED
    rms = (cs[ends] - cs[starts]) / NEED
    best = int(np.argmax(rms)) * HOP
    return d[best:best + NEED].astype(np.float32)


def _crop_to_mel(seg):
    from scipy.signal import get_window
    win = get_window("hann", C.WIN, fftbins=True).astype(np.float32)
    fb = _get_fb()
    mags = np.empty((C.N_FRAMES, C.N_FFT // 2 + 1), np.float32)
    for t in range(C.N_FRAMES):
        s = seg[t * C.HOP: t * C.HOP + C.WIN]
        if len(s) < C.WIN:
            s = np.pad(s, (0, C.WIN - len(s)))
        s = s * win
        sp = np.fft.rfft(s, C.N_FFT)
        mags[t] = (np.abs(sp) ** 2).astype(np.float32)
    mel = mags @ fb.T
    mel = np.maximum(mel, 1e-10)
    lm = np.log10(mel).T
    lo, hi = -10.0, 1.4
    lm = (lm - lo) / (hi - lo)
    return np.clip(lm, 0.0, 1.0).astype(np.float32)


def build_centered(df):
    X = np.empty((len(df), 1, C.N_MELS, C.N_FRAMES), np.float32)
    t0 = time.time()
    for i, b in enumerate(df["audio"].map(lambda x: x["bytes"])):
        X[i] = _crop_to_mel(center_crop(decode(b)))
        if (i + 1) % 500 == 0:
            print(f"  feat {i+1}/{len(df)}  ({time.time()-t0:.0f}s)", flush=True)
    return X


def avg_probs(models, X, device, bs=256):
    Xt = torch.from_numpy(X).float()
    pa_sum = np.zeros((len(Xt), C.N_A), dtype=np.float64)
    pb_sum = np.zeros((len(Xt), C.N_B), dtype=np.float64)
    for m in models:
        pa_all, pb_all = [], []
        with torch.no_grad():
            for i in range(0, len(Xt), bs):
                la, lb = m(Xt[i:i + bs].to(device))
                pa_all.append(F.softmax(la, dim=1).cpu().numpy())
                pb_all.append(F.softmax(lb, dim=1).cpu().numpy())
        pa_sum += np.concatenate(pa_all)
        pb_sum += np.concatenate(pb_all)
    return pa_sum.argmax(1), pb_sum.argmax(1)


def load_split(split):
    fs = sorted(glob.glob(f"{DATA}/{split}-*.parquet"))
    return pd.read_parquet(fs).reset_index(drop=True)


def report(name, df, ya, yb, pa, pb):
    intent = (ya == pa)
    slot_ok = (pb == yb) | (yb == C.N_VALUE)
    e2e = intent & slot_ok
    syn = (df["is_synthetic"].values == 1)
    real = ~syn

    def acc(mask, arr):
        return float(arr[mask].mean() * 100) if mask.any() else float("nan"), int(mask.sum())

    oi, _ = acc(np.ones(len(ya), bool), intent)
    oe, _ = acc(np.ones(len(ya), bool), e2e)
    si, sn = acc(syn, intent); se, _ = acc(syn, e2e)
    ri, rn = acc(real, intent); re_, _ = acc(real, e2e)
    print(f"\n{'='*72}\n{name}   n={len(ya)}  (synth={sn}  real={rn})\n{'='*72}")
    print(f"  OVERALL    intent={oi:5.1f}%   e2e={oe:5.1f}%")
    print(f"  SYNTHETIC  intent={si:5.1f}%   e2e={se:5.1f}%   (n={sn})")
    print(f"  REAL       intent={ri:5.1f}%   e2e={re_:5.1f}%   (n={rn})")

    def db(s):
        return "<=1.5s" if s <= 1.5 else "1.5-2.5s" if s <= 2.5 else "2.5-4.0s" if s <= 4.0 else ">4.0s"
    df2 = df.copy()
    df2["_i"] = intent
    df2["_d"] = df2["duration_s"].apply(db)
    df2["_real"] = real
    g = df2[df2["_real"]].groupby("_d").agg(n=("_i", "size"), ok=("_i", "sum"))
    g["acc%"] = (g.ok / g.n * 100).round(1)
    g = g.reindex([x for x in ["<=1.5s", "1.5-2.5s", "2.5-4.0s", ">4.0s"] if x in g.index])
    print("  [REAL by duration]")
    print(g.to_string())
    return dict(overall_intent=oi, overall_e2e=oe, synth_intent=si, synth_e2e=se,
                real_intent=ri, real_e2e=re_, n=len(ya), n_syn=sn, n_real=rn)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [load_model(p, device) for p in SEEDS]
    print(f"\n=== ENS5C  ({len(models)} centered seeds)  device={device} ===", flush=True)
    for p in SEEDS:
        print(f"  member: {p}")

    out = {}
    # 1) TEST + HOLDOUT: centered extraction, real/synth breakdown
    for split, label in [("test", "TEST"), ("holdout", "HOLDOUT")]:
        df = load_split(split)
        print(f"\n>> {label}: extracting centered features for {len(df)} clips...", flush=True)
        X = build_centered(df)
        ya = torch.from_numpy(df["intent_label"].values).long() if "intent_label" in df else None
        # labels live in the npz; pull from there to stay consistent
        z = np.load(f"{C.OUT_DIR}/eval_{split}.npz")
        ya = z["ya"]; yb = z["yb"]
        pa, pb = avg_probs(models, X, device)
        out[label.lower()] = report(label, df, ya, yb, pa, pb)

    # 2) OOS-TEST: reject rate (use prebuilt centered npz if present, else front)
    z = np.load(f"{C.OUT_DIR}/eval_oostest.npz")
    pa, pb = avg_probs(models, z["X"], device)
    ya = z["ya"]; yb = z["yb"]
    rej_idx = C.N_INTENT
    oos = (ya == rej_idx)
    rej_acc = float((pa[oos] == rej_idx).mean()) if oos.any() else float("nan")
    e2e = float(((pa == ya) & ((pb == yb) | (yb == C.N_VALUE))).mean())
    print(f"\n{'='*72}\nOOS-TEST   n={len(ya)}  (oos={int(oos.sum())})\n{'='*72}")
    print(f"  e2e={e2e*100:5.1f}%   reject={rej_acc*100:5.1f}%")
    out["oostest"] = dict(e2e=e2e, rej=rej_acc, n=len(ya), n_oos=int(oos.sum()))

    with open(f"{C.OUT_DIR}/eval_ens5c.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nsaved {C.OUT_DIR}/eval_ens5c.json")


if __name__ == "__main__":
    main()
