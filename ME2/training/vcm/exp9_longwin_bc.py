"""EXP 6: CRNN with LONGER time window (250 frames ~2.5s) + per-utterance CMVN.

Motivation (from audit_raw.json, TEST real-voice, n=824):
  - Duration collapses monotonically: <=1.5s e2e 52% -> >4.0s e2e 28%.
    The 150-frame (1.5s) centered window CHOPS OFF long real commands.
    SLURP (403 clips, conversational/long) is the dominant real failure at 38%.
  - Real-voice avg confidence ~0.52 vs synthetic 0.914 -> a domain/acoustic
    gap (speaker/channel), not a capacity gap. Per-utterance CMVN makes the
    features invariant to per-speaker loudness/spectral tilt.

Changes vs exp5_crnn.py (everything else identical: same CRNN 663,527 params,
same centered crop, same runB room-noise aug, same real/weak/reject weighting,
same AdamW+cosine, same early-stop):
  (1) N_FRAMES 150 -> 250  (window 1.5s -> 2.5s) so long commands are not cut.
  (2) Per-utterance cepstral mean-variance normalization on the log-mel
      (zero-mean / unit-var along the time axis) AFTER the [-10,1.4]->[0,1]
      rescale. Mild, standard, and leaves short clean synthetic clips ~flat.

Synthetic safety: synthetic commands are short & clean -> the extra 100 frames
are zero-padded silence (model already handles padding) and CMVN on a clean
clip is near-identity in the informative region. Expect synthetic to hold.

Usage:
  CUDA_VISIBLE_DEVICES=0 python exp6_longwin.py --epochs 36 --seed 301 --tag lw301 --reject-boost 2.0
"""
import argparse, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import get_window
import config as C
from train_centered import (load_noise_bank, load_audio_split,
                            CenteredAugDataset, row_weights, weighted_ce,
                            evaluate, decode_wav, center_crop, add_noise,
                            random_gain, time_shift)
from train import _class_weights as class_weights
from exp5_crnn import CRNN

# ------------------------------------------------- longer window constants ----
N_FRAMES_LONG = 250                      # 2.5 s @ 10 ms hop (was 150)
NEED_LONG = 1 + (N_FRAMES_LONG - 1) * C.HOP   # 39840 samples = 2.49 s
WIN_ARR = get_window("hann", C.WIN, fftbins=True).astype(np.float32)


def _mel_filterbank_long(n_frames):
    """Rebuild the mel filterbank (same as mel.py) and compute a 250-frame
    log-mel from an already-decoded+centered waveform."""
    from mel import _get_fb
    fb = _get_fb()
    mags = np.empty((n_frames, C.N_FFT // 2 + 1), dtype=np.float32)
    return mags, fb


def wav_to_mel_long(x):
    """x: float32 waveform of exactly NEED_LONG samples -> (40, 250) log-mel
    in [0,1], then per-utterance CMVN along the time axis.

    Mirrors train_centered.wav_to_mel_centered but with 250 frames and a
    trailing cepstral mean-variance normalization.
    """
    from mel import _get_fb
    assert len(x) >= NEED_LONG, f"waveform too short: {len(x)} < {NEED_LONG}"
    fb = _get_fb()
    mags = np.empty((N_FRAMES_LONG, C.N_FFT // 2 + 1), dtype=np.float32)
    for t in range(N_FRAMES_LONG):
        seg = x[t * C.HOP: t * C.HOP + C.WIN]
        if len(seg) < C.WIN:
            seg = np.pad(seg, (0, C.WIN - len(seg)))
        seg = seg * WIN_ARR
        spec = np.fft.rfft(seg, C.N_FFT)
        mags[t] = (np.abs(spec) ** 2).astype(np.float32)
    mel = mags @ fb.T
    mel = np.maximum(mel, 1e-10)
    logmel = np.log10(mel).T                       # (40, 250)
    lo, hi = -10.0, 1.4                            # same anchors as mel.py
    logmel = (logmel - lo) / (hi - lo)
    logmel = np.clip(logmel, 0.0, 1.0).astype(np.float32)
    # ---- per-utterance CMVN along time axis (axis=1) ----
    mu = logmel.mean(axis=1, keepdims=True)
    sd = logmel.std(axis=1, keepdims=True) + 1e-6
    logmel = (logmel - mu) / sd
    # re-anchor to a sane range after CMVN (roughly [-3, 3] -> [0, 1])
    logmel = (logmel + 3.0) / 6.0
    return np.clip(logmel, 0.0, 1.0).astype(np.float32)


# ============================================================ LEVER B: RIR ===
_RIR_CACHE = None
_RIR_LOCK = None


def _make_rirs(n=40, sr=16000, rng=None):
    """Synthesize room impulse responses (no download). Mix of:
      - exponentially-decaying direct+tail (RT60 0.15..0.6s)
      - Schroeder network (4 comb + 4 allpass, random delays)
    Each RIR normalized to unit energy. Cached per-process (DataLoader workers
    each build their own copy on first use; deterministic seed -> same bank)."""
    global _RIR_CACHE, _RIR_LOCK
    if _RIR_CACHE is not None:
        return _RIR_CACHE
    import threading
    if _RIR_LOCK is None:
        _RIR_LOCK = threading.Lock()
    with _RIR_LOCK:
        if _RIR_CACHE is not None:
            return _RIR_CACHE
        if rng is None:
            rng = np.random.RandomState(0)
    rirs = []
    for k in range(n):
        rt60 = rng.uniform(0.15, 0.6)
        if k % 2 == 0:
            # exponential decay tail
            dur = int(rt60 * 3 * sr)
            t = np.arange(dur) / sr
            ir = np.exp(-6.9 * t / rt60) * rng.rand(dur) ** 0.5
            ir = ir * (rng.rand() > 0.3)  # sparse taps
        else:
            # Schroeder network
            dur = int(rt60 * 3 * sr)
            ir = rng.randn(dur)
            for _ in range(4):  # combs
                d = rng.randint(30, 120)
                g = rng.uniform(0.3, 0.7)
                for i in range(d, len(ir)):
                    ir[i] += g * ir[i - d]
            for _ in range(4):  # allpasses
                d = rng.randint(10, 40)
                g = rng.uniform(0.5, 0.9)
                for i in range(d, len(ir)):
                    ir[i] = ir[i] - g * ir[i - d] + g * ir[i]
        ir = ir / (np.sqrt(np.sum(ir ** 2)) + 1e-9)
        rirs.append(ir.astype(np.float32))
    _RIR_CACHE = rirs
    return rirs


def add_reverb(x, rirs, rng):
    """Convolve waveform x with a random RIR (early-reflection + late-reverb)."""
    if len(rirs) == 0:
        return x
    ir = rirs[rng.randint(len(rirs))]
    wet = np.convolve(x, ir, mode="full")[:len(x)]
    # normalize wet to match x's RMS so reverb level varies naturally
    rms_x = np.sqrt(np.mean(x ** 2)) + 1e-9
    rms_w = np.sqrt(np.mean(wet ** 2)) + 1e-9
    wet = wet * (rms_x / rms_w) * rng.uniform(0.3, 1.0)
    out = x + wet
    # keep peak bounded
    pk = np.max(np.abs(out))
    if pk > 0.98:
        out = out * (0.98 / pk)
    return out.astype(np.float32)


def specaug_mel(mel, rng, t_mask=2, t_len=20, f_mask=2, f_len=8):
    """Time + frequency masking on a (40, 250) log-mel. Applied AFTER CMVN
    (so it masks the normalized feature, standard practice)."""
    mel = mel.copy()
    nt, nf = mel.shape
    for _ in range(t_mask):
        t0 = rng.randint(0, max(1, nt - t_len))
        mel[t0:t0 + t_len, :] = 0.0
    for _ in range(f_mask):
        f0 = rng.randint(0, max(1, nf - f_len))
        mel[:, f0:f0 + f_len] = 0.0
    return mel


def speed_perturb(x, rng, factor=(0.9, 1.1)):
    """Resample waveform by a random speed factor (pitch-preserving-ish via
    linear interp). 0.9 = slower, 1.1 = faster."""
    f = rng.uniform(*factor)
    n_new = int(len(x) / f)
    if n_new < 100:
        return x
    idx = np.linspace(0, len(x) - 1, n_new)
    out = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
    return out


# ============================================== LEVER C: real-voice OOS ====
def load_numerals_reject(max_rows=20000):
    """Load the 66K real-voice OUT_OF_SCOPE 'numerals' split as reject rows.
    Returns list of (wav_float, ya=REJ_A, yb=REJ_B, is_real=True).
    These are REAL speech (not synthetic) -> is_real=True so they get the
    real-boost weight, teaching the reject head what real non-commands sound
    like (attacks over-rejection bucket 2)."""
    import glob
    import pandas as pd
    from train_centered import decode_wav as _dw
    REJ_A, REJ_B = C.N_INTENT, C.N_VALUE
    files = sorted(glob.glob(C.HF_DIR + "/data/numerals-*.parquet"))
    rng = np.random.RandomState(1234)
    # read all, then subsample to a GLOBAL cap (keeps train balanced:
    # ~8K real OOS reject vs ~24K command rows)
    dfs = [pd.read_parquet(fp) for fp in files]
    full = pd.concat(dfs, ignore_index=True)
    if len(full) > max_rows:
        idx = rng.choice(len(full), size=max_rows, replace=False)
        full = full.iloc[idx]
    print(f"[BC] numerals: {len(full)} of {sum(len(d) for d in dfs)} sampled "
          f"(global cap {max_rows})", flush=True)
    rows = []
    bad = 0
    for i in range(len(full)):
        aud = full["audio"].iloc[i]
        wb = aud["bytes"] if isinstance(aud, dict) else aud
        try:
            x = _dw(wb)
        except Exception:
            bad += 1
            continue
        if x is None or len(x) < 8000:
            bad += 1
            continue
        rows.append((x, REJ_A, REJ_B, True))
    print(f"[BC] loaded {len(rows)} real-voice OOS reject rows ({bad} decode-fail)",
          flush=True)
    return rows


class LongWinDataset(CenteredAugDataset):
    """CenteredAugDataset + 2.5s window + CMVN, PLUS Lever B aug:
    RIR reverb, SpecAug (time/freq masking), speed perturbation.
    All B-aug applied with configurable probability (default 0.5 each)."""

    def __init__(self, rows, noise_bank, aug=True, seed=0, weights=None,
                 snr_min=12.0, snr_max=28.0, gain_db=4.0, shift=0.10,
                 reverb_p=0.5, specaug_p=0.5, speed_p=0.5, rirs=None):
        super().__init__(rows, noise_bank, aug=aug, seed=seed, weights=weights,
                         snr_min=snr_min, snr_max=snr_max, gain_db=gain_db,
                         shift=shift)
        self.reverb_p = reverb_p
        self.specaug_p = specaug_p
        self.speed_p = speed_p
        self.rirs = rirs if rirs is not None else []

    def __getitem__(self, i):
        x, ya, yb, _ = self.rows[i]
        if self.aug:
            # --- original runB aug ---
            if self.rng.random() < 0.5:
                x = add_noise(x, self.noise, self.rng, self.snr_min, self.snr_max)
            x = random_gain(x, self.rng, self.gain_db)
            x = time_shift(x, self.rng, self.shift)
            # --- Lever B: reverb (SLURP is far-field reverberant) ---
            # lazy-build the RIR bank in this (worker) process on first use
            if self.rng.random() < self.reverb_p:
                if not self.rirs:
                    self.rirs = _make_rirs(40, sr=C.SR)
                x = add_reverb(x, self.rirs, self.rng)
            # --- Lever B: speed perturbation (0.9..1.1x) ---
            if self.rng.random() < self.speed_p:
                x = speed_perturb(x, self.rng)
        # center-crop to the LONGER window (loudest 2.5s), then 250-frame mel
        x = center_crop(x, n=NEED_LONG)
        mel = wav_to_mel_long(x)
        # --- Lever B: SpecAug on the log-mel (after CMVN) ---
        if self.aug and self.rng.random() < self.specaug_p:
            mel = specaug_mel(mel, self.rng)
        return (torch.from_numpy(mel)[None], torch.tensor(ya, dtype=torch.long),
                torch.tensor(yb, dtype=torch.long),
                torch.tensor(self.weights[i], dtype=torch.float32))


# ------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=36)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--tag", default="lw1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--real-boost", type=float, default=2.5)
    ap.add_argument("--weak-boost", type=float, default=1.5)
    ap.add_argument("--class-boost", type=float, default=2.0)
    ap.add_argument("--reject-boost", type=float, default=2.0,
                    help="extra multiplier on the REJECT class weight (idx 19)")
    ap.add_argument("--snr-min", type=float, default=12.0)
    ap.add_argument("--snr-max", type=float, default=28.0)
    ap.add_argument("--gain-db", type=float, default=4.0)
    ap.add_argument("--shift", type=float, default=0.10)
    # ---- Lever B: reverb / SpecAug / speed ----
    ap.add_argument("--reverb-p", type=float, default=0.5,
                    help="prob of applying RIR reverb per clip")
    ap.add_argument("--specaug-p", type=float, default=0.5,
                    help="prob of applying time/freq masking per clip")
    ap.add_argument("--speed-p", type=float, default=0.5,
                    help="prob of speed perturbation per clip")
    ap.add_argument("--no-rir", action="store_true",
                    help="disable reverb (B without RIR)")
    # ---- Lever C: real-voice OOS from numerals ----
    ap.add_argument("--numerals", type=int, default=8000,
                    help="num real-voice OOS reject rows to add (0=off)")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[longwin] device={device} N_FRAMES={N_FRAMES_LONG} "
          f"(window {N_FRAMES_LONG*C.HOP/16000:.2f}s) reject_boost={args.reject_boost}",
          flush=True)

    weak_names = ["VOLUME_UP", "VOLUME_DOWN", "LIGHT_ON", "LIGHT_OFF", "STOP", "NEXT"]
    weak_set = np.array([C.INTENT2IDX[n] for n in weak_names])

    noise_bank = load_noise_bank()
    tr_rows = load_audio_split(["train", "fil", "neg_train", "neg_noise"])
    # ---- Lever C: append real-voice OOS reject rows (numerals split) ----
    if args.numerals > 0:
        num_rows = load_numerals_reject(max_rows=args.numerals)
        tr_rows = tr_rows + num_rows
        print(f"[BC] train rows after numerals: {len(tr_rows)}", flush=True)
    va_rows = load_audio_split(["test", "tune_oos"])
    # ---- Lever B: RIR bank is lazy-built per worker process (spawn-safe) ----
    if not args.no_rir:
        print("[BC] RIR reverb enabled (bank built lazily per worker)", flush=True)
    kw = dict(snr_min=args.snr_min, snr_max=args.snr_max, gain_db=args.gain_db,
              shift=args.shift, reverb_p=(0.0 if args.no_rir else args.reverb_p),
              specaug_p=args.specaug_p, speed_p=args.speed_p, rirs=[])
    va = LongWinDataset(va_rows, noise_bank, aug=False, seed=args.seed)
    val = torch.utils.data.DataLoader(va, batch_size=256, shuffle=False,
                                      num_workers=args.workers, pin_memory=True)
    print(f"[longwin] train={len(tr_rows)} val={len(va_rows)}", flush=True)

    ya_arr = np.array([r[1] for r in tr_rows])
    real_arr = np.array([r[3] for r in tr_rows])
    rw = row_weights(ya_arr, real_arr, args.real_boost, args.weak_boost, weak_set)
    cw = class_weights(torch.tensor(ya_arr), C.N_A, device, boost=args.class_boost)
    if args.reject_boost != 1.0:
        rej_idx = C.N_A - 1
        cw[rej_idx] = cw[rej_idx] * args.reject_boost
        print(f"[longwin] reject_boost={args.reject_boost} -> cw[{rej_idx}]="
              f"{cw[rej_idx].item():.3f}", flush=True)

    tr = LongWinDataset(tr_rows, noise_bank, aug=True, seed=args.seed,
                        weights=rw, **kw)
    print(f"[BC] aug: reverb_p={args.reverb_p} specaug_p={args.specaug_p} "
          f"speed_p={args.speed_p} numerals={args.numerals}", flush=True)
    trl = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True,
                                      num_workers=args.workers, pin_memory=True)

    model = CRNN(n_a=C.N_A, n_b=C.N_B).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[longwin] params={n_params:,}", flush=True)

    ce_b = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=C.WEIGHT_DECAY)
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
        print(f"[longwin] ep{ep:02d} loss={tot/n:.4f} val_intent={vi*100:.2f} "
              f"val_slot={vs*100:.2f} val_e2e={ve*100:.2f}  ({dt:.0f}s)", flush=True)
        if ve > best_e2e:
            best_e2e = ve
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 6:
                print("[longwin] early stop", flush=True)
                break

    model.load_state_dict(best_state)
    out = f"{C.OUT_DIR}/vcm_crnn_{args.tag}.pt"
    torch.save({"model": best_state, "config": vars(args),
                "intents": C.INTENTS, "value_tokens": C.VALUE_TOKENS,
                "best_val": best_e2e, "n_params": n_params, "arch": "CRNN-longwin",
                "n_frames": N_FRAMES_LONG, "tag": args.tag, "epochs_run": ep,
                "recipe": "CRNN(CNN+GRU) + 2.5s window + CMVN + centered + "
                          "weighting + runB noise + LeverB(reverb/specaug/speed)"
                          f"+ LeverC(numerals={args.numerals})"}, out)
    print(f"[longwin] saved {out}  best_val_e2e={best_e2e*100:.2f}", flush=True)


if __name__ == "__main__":
    main()
