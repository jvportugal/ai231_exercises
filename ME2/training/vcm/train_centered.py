"""Run B retrain with TWO changes, same recipe otherwise:

  (1) CENTERED 1.5s window at training time.
      Run B was trained on FRONT-TRIM mels (first 1.5s). Real clips >1.5s had
      their command chopped off (leading silence kept instead). We proved
      centering (crop the 1.5s window with max short-time RMS) lifts holdout
      real e2e 30% -> 59% at inference WITHOUT retraining. This bakes the same
      crop into TRAINING so the model learns from correctly-centered real clips.

  (2) REAL-VOICE + WEAK-CLASS loss weighting.
      78% of training is synthetic (easy). Real-voice rows are up-weighted so
      the model stops averaging toward the synthetic majority, and the weak
      1-syllable twin intents (VOLUME_UP/DOWN, LIGHT_ON/OFF, STOP, NEXT) get an
      extra boost.

Everything else mirrors train_step2.py (Run B): same TwoHeadCNN (808,903
params, Pi-sized), same room-noise-only augmentation (SNR 12-28dB, gain +-4dB,
time-shift +-10%, NO babble/stretch/SpecAug), AdamW 3e-4 + cosine, early-stop
on val-e2e (patience 6), leak-safe splits.

Usage:
  CUDA_VISIBLE_DEVICES=1 python train_centered.py --epochs 36 --tag runBc
"""
import argparse
import glob
import io
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import wavfile
from scipy.signal import get_window
try:
    import soundfile as sf
except Exception:
    sf = None

import config as C
from train import TwoHeadCNN, _class_weights as class_weights

# ----------------------------------------------------------------- audio ----
SR = C.SR
HOP, WIN, N_FFT, N_FRAMES = C.HOP, C.WIN, C.N_FFT, C.N_FRAMES
NEED = 1 + (N_FRAMES - 1) * HOP          # 39488 samples = 2.468 s
WIN_ARR = get_window("hann", WIN, fftbins=True).astype(np.float32)
NOISE_DIR = C.NOISE_DIR


def _resample(x, sr, target=SR):
    if sr == target:
        return x
    n = int(len(x) * target / sr)
    return np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.float32)


def decode_wav(wb):
    """Decode WAV or FLAC bytes -> float32 mono @ SR."""
    try:
        data, sr = sf.read(io.BytesIO(wb), dtype="float32")
        if data.ndim > 1:
            data = data[:, 0]
        return _resample(data.astype(np.float32), sr)
    except Exception:
        sr, data = wavfile.read(io.BytesIO(wb))
        data = data.astype(np.float32) / 32768.0 if data.dtype == np.int16 else data.astype(np.float32)
        if data.ndim > 1:
            data = data[:, 0]
        return _resample(data, sr)


def center_crop(x, n=NEED):
    """Crop/pad to n samples at the LOUDEST (max short-time RMS) position.

    For clips shorter than n: zero-pad so the signal is centered.
    For clips longer than n: slide a window of n and keep the one with the
    highest mean short-time RMS (the actual spoken part, not leading silence).
    """
    if len(x) <= n:
        pad = n - len(x)
        left = pad // 2
        return np.concatenate([np.zeros(left, np.float32), x,
                               np.zeros(pad - left, np.float32)])
    # long clip: compute short-time RMS over a sliding window of n
    frame = 512
    hop = 256
    n_frames = (len(x) - n) // hop + 1
    if n_frames <= 1:
        return x[:n]
    rms = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        s = i * hop
        seg = x[s:s + n]
        # coarse RMS: mean of |seg| over 512-sample blocks
        nb = n // frame
        rms[i] = np.abs(seg[:nb * frame]).reshape(nb, frame).mean(axis=1).mean()
    best = int(np.argmax(rms))
    return x[best * hop: best * hop + n].astype(np.float32)


def stft_mag(x):
    mags = np.empty((N_FRAMES, N_FFT // 2 + 1), dtype=np.float32)
    for t in range(N_FRAMES):
        seg = x[t * HOP: t * HOP + WIN]
        if len(seg) < WIN:
            seg = np.pad(seg, (0, WIN - len(seg)))
        mags[t] = np.abs(np.fft.rfft(seg * WIN_ARR, N_FFT)) ** 2
    return mags


def mag_to_logmel(mags, fb):
    mel = mags @ fb.T
    mel = np.maximum(mel, 1e-10)
    logmel = np.log10(mel).T
    lo, hi = -10.0, 1.4
    return np.clip((logmel - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def _mel_fb():
    from mel import _get_fb
    return _get_fb()


FB = _mel_fb()


def wav_to_mel_centered(x):
    """Center-crop to NEED then compute the (40,150) log-mel (same norm as mel.py)."""
    x = center_crop(x, NEED)
    mags = stft_mag(x)
    return mag_to_logmel(mags, FB)


# ------------------------------------------------------------- augmenters ----
def time_shift(x, rng, frac):
    s = int(rng.uniform(-frac, frac) * len(x))
    if s >= 0:
        return np.concatenate([np.zeros(s, np.float32), x[:len(x) - s]])
    return np.concatenate([x[-s:], np.zeros(-s, np.float32)])


def add_noise(x, noise_bank, rng, snr_min, snr_max):
    src = noise_bank[rng.randint(len(noise_bank))]
    snr_db = rng.uniform(snr_min, snr_max)
    n = len(x)
    start = rng.randint(0, max(1, len(src) - n))
    nz = src[start:start + n].copy()
    if len(nz) < n:
        nz = np.pad(nz, (0, n - len(nz)))
    px = np.sqrt((x ** 2).mean() + 1e-12)
    pn = np.sqrt((nz ** 2).mean() + 1e-12)
    nz = nz * (px / (pn * (10 ** (snr_db / 20.0))))
    return x + nz


def random_gain(x, rng, db):
    g = 10 ** (rng.uniform(-db, db) / 20.0)
    return np.clip(x * g, -1.0, 1.0)


# ------------------------------------------------------------- data loading ----
def read_parquets(pattern):
    fs = sorted(glob.glob(pattern))
    assert fs, f"no parquet for {pattern}"
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)


def load_noise_bank():
    fs = sorted(glob.glob(os.path.join(NOISE_DIR, "*.wav")))
    bank = []
    for f in fs:
        try:
            sr, d = wavfile.read(f)
            d = d.astype(np.float32) / 32768.0 if d.dtype == np.int16 else d.astype(np.float32)
            if d.ndim > 1:
                d = d[:, 0]
            d = _resample(d, sr)
            if len(d) > 8000:
                bank.append(d)
        except Exception:
            pass
    assert bank, f"no noise wavs in {NOISE_DIR}"
    return bank


def load_audio_split(tags):
    """tags = list of (label, glob_pattern). Returns list of (wav_float, ya, yb, is_real)."""
    data = C.HF_DIR + "/data"
    fil = C.HF_DIR + "/supplemental_fil"
    neg = C.HF_DIR + "/synthetic_negatives"
    parts = {
        "train": (f"{data}/train-*.parquet", False),
        "test": (f"{data}/test-*.parquet", False),
        "fil": (f"{fil}/*.parquet", False),
        "neg_train": (f"{neg}/train-*.parquet", False),
        "neg_noise": (f"{neg}/noise-*.parquet", False),
        "tune_oos": (f"{neg}/tune_oos-*.parquet", False),
    }
    REJ_A, REJ_B = C.N_INTENT, C.N_VALUE
    rows = []
    for tag in tags:
        pat, _ = parts[tag]
        df = read_parquets(pat)
        # real-voice flag from the dataset's own is_synthetic column (0 = real)
        if "is_synthetic" in df.columns:
            real_flags = (df["is_synthetic"].fillna(1).astype(int) == 0).values
        else:
            real_flags = np.zeros(len(df), dtype=bool)
        n_real = 0
        for i in range(len(df)):
            aud = df["audio"].iloc[i]
            wb = aud["bytes"] if isinstance(aud, dict) else aud
            try:
                x = decode_wav(wb)
            except Exception:
                continue
            cmd = df["command"].iloc[i]
            _sv = df["slot_value"].iloc[i]
            sv = _sv.strip() if isinstance(_sv, str) else ""
            if cmd == "OUT_OF_SCOPE":
                ya, yb = REJ_A, REJ_B
            else:
                ya = C.INTENT2IDX[cmd]
                if sv and (cmd, sv) in C.PAIR2TOKEN:
                    yb = C.VALUE2IDX[C.PAIR2TOKEN[(cmd, sv)]]
                else:
                    yb = REJ_B
            rows.append((x, ya, yb, bool(real_flags[i])))
            if real_flags[i]:
                n_real += 1
        print(f"    [{tag}] +{len(df)} rows (real {n_real}, total {len(rows)})", flush=True)
    return rows


# ------------------------------------------------------------- dataset ----
class CenteredAugDataset(torch.utils.data.Dataset):
    def __init__(self, rows, noise_bank, aug=True, seed=0, weights=None,
                 snr_min=12.0, snr_max=28.0, gain_db=4.0, shift=0.10):
        self.rows = rows
        self.noise = noise_bank
        self.aug = aug
        self.rng = np.random.RandomState(seed)
        self.snr_min, self.snr_max = snr_min, snr_max
        self.gain_db, self.shift = gain_db, shift
        # per-row scalar loss weight (None -> all ones)
        self.weights = weights if weights is not None else np.ones(len(rows), np.float32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        x, ya, yb, _ = self.rows[i]
        if self.aug:
            if self.rng.random() < 0.5:
                x = add_noise(x, self.noise, self.rng, self.snr_min, self.snr_max)
            x = random_gain(x, self.rng, self.gain_db)
            x = time_shift(x, self.rng, self.shift)
        mel = wav_to_mel_centered(x)
        return (torch.from_numpy(mel)[None], torch.tensor(ya, dtype=torch.long),
                torch.tensor(yb, dtype=torch.long),
                torch.tensor(self.weights[i], dtype=torch.float32))


# ------------------------------------------------------------- weighting ----
def row_weights(ya_arr, is_real_arr, real_boost, weak_boost, weak_set):
    """Per-row scalar weight: base 1.0; x real_boost if real; x weak_boost if
    ya in weak_set. Stacked multiplicatively."""
    w = np.ones(len(ya_arr), dtype=np.float32)
    w[is_real_arr] *= real_boost
    w[np.isin(ya_arr, list(weak_set))] *= weak_boost
    return w


def weighted_ce(logits, targets, w, class_w=None):
    """Cross-entropy with per-row weight w and optional per-class weight class_w."""
    if class_w is not None:
        base = F.cross_entropy(logits, targets, weight=class_w, reduction="none")
    else:
        base = F.cross_entropy(logits, targets, reduction="none")
    return (base * w).sum() / w.sum().clamp(min=1.0)


# ------------------------------------------------------------- eval ----
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ya_l, yb_l, pa_l, pb_l = [], [], [], []
    for X, ya, yb, _w in loader:
        X = X.to(device)
        la, lb = model(X)
        pa_l.append(la.argmax(1).cpu()); pb_l.append(lb.argmax(1).cpu())
        ya_l.append(ya); yb_l.append(yb)
    ya = torch.cat(ya_l); yb = torch.cat(yb_l)
    pa = torch.cat(pa_l); pb = torch.cat(pb_l)
    intent = (pa == ya).float().mean().item()
    slotted = yb != C.N_VALUE
    slot = (pb[slotted] == yb[slotted]).float().mean().item() if slotted.any() else float("nan")
    e2e = ((pa == ya) & ((pb == yb) | (yb == C.N_VALUE))).float().mean().item()
    return intent, slot, e2e


# ------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=36)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--tag", default="runBc")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--real-boost", type=float, default=2.5,
                    help="loss weight multiplier for real-voice rows")
    ap.add_argument("--weak-boost", type=float, default=1.5,
                    help="extra loss weight multiplier for weak twin intents")
    ap.add_argument("--class-boost", type=float, default=2.0,
                    help="inverse-frequency class weight clamp (head_a)")
    ap.add_argument("--snr-min", type=float, default=12.0)
    ap.add_argument("--snr-max", type=float, default=28.0)
    ap.add_argument("--gain-db", type=float, default=4.0)
    ap.add_argument("--shift", type=float, default=0.10)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[centered] device={device}", flush=True)

    # weak twin intents (1-syllable, currently worst on real voices)
    weak_names = ["VOLUME_UP", "VOLUME_DOWN", "LIGHT_ON", "LIGHT_OFF", "STOP", "NEXT"]
    weak_set = np.array([C.INTENT2IDX[n] for n in weak_names])
    print(f"[centered] weak intents: {weak_names}", flush=True)

    print("[centered] loading noise ...", flush=True)
    noise_bank = load_noise_bank()

    print("[centered] loading TRAIN audio (centered mels computed on the fly) ...", flush=True)
    tr_rows = load_audio_split(["train", "fil", "neg_train", "neg_noise"])
    print("[centered] loading VAL audio (test + tune_oos) ...", flush=True)
    va_rows = load_audio_split(["test", "tune_oos"])

    kw = dict(snr_min=args.snr_min, snr_max=args.snr_max, gain_db=args.gain_db, shift=args.shift)
    va = CenteredAugDataset(va_rows, noise_bank, aug=False, seed=args.seed)
    val = torch.utils.data.DataLoader(va, batch_size=256, shuffle=False,
                                      num_workers=args.workers, pin_memory=True)
    print(f"[centered] train={len(tr_rows)} val={len(va_rows)}", flush=True)

    # per-row weights (real + weak)
    ya_arr = np.array([r[1] for r in tr_rows])
    real_arr = np.array([r[3] for r in tr_rows])
    rw = row_weights(ya_arr, real_arr, args.real_boost, args.weak_boost, weak_set)
    print(f"[centered] real rows={int(real_arr.sum())} ({real_arr.mean()*100:.1f}%), "
          f"weak rows={int(np.isin(ya_arr, weak_set).sum())}, "
          f"w mean={rw.mean():.3f} max={rw.max():.3f}", flush=True)
    # per-class inverse-freq weights for head_a (clamped)
    cw = class_weights(torch.tensor(ya_arr), C.N_A, device, boost=args.class_boost)

    tr = CenteredAugDataset(tr_rows, noise_bank, aug=True, seed=args.seed,
                            weights=rw, **kw)
    trl = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True,
                                      num_workers=args.workers, pin_memory=True)
    model = TwoHeadCNN(n_a=C.N_A, n_b=C.N_B).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[centered] params={n_params:,}", flush=True)

    ce_b = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_e2e, best_state, bad = -1.0, None, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for X, ya, yb, w in trl:
            X, ya, yb, w = X.to(device), ya.to(device), yb.to(device), w.to(device)
            opt.zero_grad()
            la, lb = model(X)
            loss = weighted_ce(la, ya, w, class_w=cw) + ce_b(lb, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * X.size(0); n += X.size(0)
        sched.step()
        vi, vs, ve = evaluate(model, val, device)
        dt = time.time() - t0
        print(f"[centered] ep{ep:02d} loss={tot/n:.4f} val_intent={vi*100:.2f} "
              f"val_slot={vs*100:.2f} val_e2e={ve*100:.2f}  ({dt:.0f}s)", flush=True)
        if ve > best_e2e:
            best_e2e = ve
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 6:
                print("[centered] early stop", flush=True)
                break

    out = os.path.join(C.OUT_DIR, f"vcm_twohead_{args.tag}.pt")
    torch.save({"model": best_state, "n_a": C.N_A, "n_b": C.N_B,
                "intents": C.INTENTS, "value_tokens": C.VALUE_TOKENS,
                "best_val": best_e2e, "n_params": n_params, "arch": "TwoHeadCNN",
                "tag": args.tag, "epochs_run": ep,
                "recipe": "centered-window + real/weak weighting + runB noise"}, out)
    print(f"[centered] saved {out}  best_val_e2e={best_e2e*100:.2f}", flush=True)


if __name__ == "__main__":
    main()
