"""AUTONOMOUS BATTERY: 10 non-pretrained experiments to lift longwin real-voice e2e.

Base recipe = exp6_longwin.py (the winner): CRNN 663K (3 conv 32/64/128 + pool-mel
+ BiGRU(128,2) + FC + dual heads), 250-frame (2.5s) centered window, per-utt CMVN,
runB room-noise aug (SNR 12-28, gain +-4dB, shift +-10%), real/weak/reject weighting,
AdamW 3e-4 + cosine, early-stop patience 6 on val_e2e (val = test + tune_oos).

This file is ONE parameterized trainer. Experiments differ only by flags:
  DATA-AUG (targets the SLURP / over-rejection buckets):
    E1  reverb-only            --reverb-p 0.3
    E2  reverb + babble        --reverb-p 0.3 --babble-p 0.3
    E3  reverb + SpecAug-lite  --reverb-p 0.3 --specaug-p 0.3 --spec-tmask 1 --spec-tlen 10 --spec-fmask 1 --spec-flen 4
    E4  numerals-reject light  --numerals 3000 --reject-boost 1.0
    E5  reverb + numerals      --reverb-p 0.3 --numerals 3000 --reject-boost 1.0
  ARCHITECTURE (targets the intra-slot / twin buckets):
    E6  wider CNN              --ch 64 128 256
    E7  deeper CNN             --ch 32 64 128 256
    E8  GRU head               --gru 256 --gru-layers 3
    E9  label smoothing        --label-smooth 0.1
    E10 longer window 3.0s     --nframes 300

Every variant keeps the SAME heads, SAME weighting, SAME optimizer, SAME val set,
so val_e2e is directly comparable to longwin's curve (go/no-go for the sweep).

Usage:
  CUDA_VISIBLE_DEVICES=0 python battery.py --tag E1_reverb --epochs 8 --seed 13 --reverb-p 0.3
"""
import argparse, os, time, threading
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
from exp9_longwin_bc import (load_numerals_reject, _make_rirs, add_reverb,
                             speed_perturb,
                             specaug_mel)

# ------------------------------------------------- window constants (dynamic) ----
def _need(nf):
    return 1 + (nf - 1) * C.HOP

WIN_ARR = get_window("hann", C.WIN, fftbins=True).astype(np.float32)

_FB_CACHE = {}
def _mel_filterbank(n_frames):
    """Mel filterbank (same math as mel.py) sized for n_frames."""
    if n_frames in _FB_CACHE:
        return _FB_CACHE[n_frames]
    import librosa
    fb = librosa.filters.mel(sr=C.SR, n_fft=C.N_FFT, n_mels=C.N_MELS,
                              fmin=C.FMIN, fmax=C.FMAX)
    _FB_CACHE[n_frames] = fb
    return fb


def wav_to_mel(x, n_frames):
    """Centered-window log-mel with per-utt CMVN (same as exp6, generalized)."""
    need = _need(n_frames)
    assert len(x) >= need, f"short clip {len(x)} < {need}"
    fb = _mel_filterbank(n_frames)
    mags = np.empty((n_frames, C.N_FFT // 2 + 1), dtype=np.float32)
    for t in range(n_frames):
        seg = x[t * C.HOP: t * C.HOP + C.WIN]
        if len(seg) < C.WIN:
            seg = np.pad(seg, (0, C.WIN - len(seg)))
        seg = seg * WIN_ARR
        spec = np.fft.rfft(seg, C.N_FFT)
        mags[t] = (np.abs(spec) ** 2).astype(np.float32)
    mel = mags @ fb.T
    mel = np.maximum(mel, 1e-10)
    logmel = np.log10(mel).T                       # (40, n_frames)
    lo, hi = -10.0, 1.4
    logmel = (logmel - lo) / (hi - lo)
    logmel = np.clip(logmel, 0.0, 1.0).astype(np.float32)
    mu = logmel.mean(axis=1, keepdims=True)
    sd = logmel.std(axis=1, keepdims=True) + 1e-6
    logmel = (logmel - mu) / sd
    logmel = (logmel + 3.0) / 6.0
    return np.clip(logmel, 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------- babble noise ----
_BABBLE_CACHE = None
_BABBLE_LOCK = threading.Lock()
def _babble_bank(n=24, sr=16000):
    """Synthesize multi-talker babble (sum of 3-5 independent noise sources with
    random amplitude modulation). No download. Cached per process."""
    global _BABBLE_CACHE
    if _BABBLE_CACHE is not None:
        return _BABBLE_CACHE
    with _BABBLE_LOCK:
        if _BABBLE_CACHE is not None:
            return _BABBLE_CACHE
        rng = np.random.RandomState(7)
        bank = []
        for _ in range(n):
            t = np.arange(sr * 3) / sr
            nsrc = rng.randint(3, 6)
            sig = np.zeros_like(t)
            for _ in range(nsrc):
                # pinkish noise via cumulative gaussian, amplitude-modulated
                nz = np.cumsum(rng.randn(len(t)))
                nz = nz - nz.mean()
                nz /= (nz.std() + 1e-9)
                rate = rng.uniform(2.0, 6.0)
                am = 0.5 + 0.5 * np.sin(2 * np.pi * rate * t + rng.uniform(0, 6))
                sig += nz * am * rng.uniform(0.5, 1.0)
            sig /= (np.sqrt(np.mean(sig ** 2)) + 1e-9)
            bank.append(sig.astype(np.float32))
        _BABBLE_CACHE = bank
    return _BABBLE_CACHE


def add_babble(x, rng, p=0.3, snr_min=8.0, snr_max=20.0):
    if rng.random() >= p:
        return x
    bank = _babble_bank()
    nb = bank[rng.randint(len(bank))]
    # tile/trim to length of x
    if len(nb) < len(x):
        reps = len(x) // len(nb) + 1
        nb = np.tile(nb, reps)[:len(x)]
    else:
        nb = nb[:len(x)]
    snr = rng.uniform(snr_min, snr_max)
    px = np.sqrt(np.mean(x ** 2)) + 1e-9
    pn = np.sqrt(np.mean(nb ** 2)) + 1e-9
    target = px / (10 ** (snr / 20.0))
    out = x + nb * (target / pn)
    pk = np.max(np.abs(out))
    if pk > 0.98:
        out *= 0.98 / pk
    return out.astype(np.float32)


def add_playback_channel(x, rng):
    """Approximate a laptop-speaker -> room -> USB-microphone channel.

    This uses no external audio.  A smooth random frequency response models
    speaker/microphone coloration, optional low-rate resampling models cheap
    audio interfaces, and a soft limiter models playback/capture compression.
    Peak normalization alone cannot create these distortions, so this augments
    the exact failure mode seen in the live acoustic benchmark.
    """
    y = np.asarray(x, dtype=np.float32).copy()
    if y.size < 2:
        return y

    # Smooth broad-band EQ.  Speech-band anchors receive modest independent
    # variation while the extremes are attenuated like small speakers/mics.
    n_fft = 1 << (len(y) - 1).bit_length()
    spectrum = np.fft.rfft(y, n=n_fft)
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / C.SR)
    anchors_hz = np.asarray([0, 70, 180, 500, 1200, 3000, 5500, 8000], np.float32)
    anchors_db = np.asarray([
        rng.uniform(-30.0, -16.0), rng.uniform(-18.0, -8.0),
        rng.uniform(-7.0, 2.0), rng.uniform(-4.0, 4.0),
        rng.uniform(-4.0, 4.0), rng.uniform(-5.0, 3.0),
        rng.uniform(-12.0, 0.0), rng.uniform(-30.0, -12.0),
    ], np.float32)
    response_db = np.interp(frequencies, anchors_hz, anchors_db)
    response = np.power(10.0, response_db / 20.0)
    y = np.fft.irfft(spectrum * response, n=n_fft)[:len(y)].astype(np.float32)

    # Simulate a restricted capture rate followed by the runtime's 16 kHz
    # resampling.  Linear interpolation deliberately matches the live helper.
    if rng.random() < 0.65:
        rate = int(rng.choice([8000, 11025, 12000, 16000]))
        if rate != C.SR:
            n_low = max(2, int(round(len(y) * rate / C.SR)))
            low = np.interp(np.linspace(0, len(y) - 1, n_low),
                            np.arange(len(y)), y).astype(np.float32)
            y = np.interp(np.linspace(0, n_low - 1, len(y)),
                          np.arange(n_low), low).astype(np.float32)

    # Mild automatic-gain compression / overload.  Random gain by itself is
    # removed by runtime normalization; the nonlinear shape is not.
    drive = float(rng.uniform(1.0, 3.0))
    y = np.tanh(y * drive) / np.tanh(drive)
    if rng.random() < 0.25:
        ceiling = float(rng.uniform(0.55, 0.95))
        y = np.clip(y, -ceiling, ceiling) / ceiling
    peak = float(np.max(np.abs(y)))
    if peak > 0.98:
        y *= 0.98 / peak
    return y.astype(np.float32)


# ============================================================ dataset ==========
class BatDataset(CenteredAugDataset):
    """LongWinDataset + optional reverb / babble (waveform) and SpecAug (mel)."""

    def __init__(self, rows, noise_bank, aug=True, seed=0, weights=None,
                 snr_min=12.0, snr_max=28.0, gain_db=4.0, shift=0.10,
                 n_frames=250, reverb_p=0.0, babble_p=0.0, specaug_p=0.0,
                 spec_tmask=1, spec_tlen=10, spec_fmask=1, spec_flen=4,
                 speed_p=0.0, speed_lo=0.9, speed_hi=1.1,
                 channel_p=0.0):
        super().__init__(rows, noise_bank, aug=aug, seed=seed, weights=weights,
                         snr_min=snr_min, snr_max=snr_max, gain_db=gain_db,
                         shift=shift)
        self.n_frames = n_frames
        self.reverb_p = reverb_p
        self.babble_p = babble_p
        self.specaug_p = specaug_p
        self.speed_p = speed_p
        self.channel_p = channel_p
        self.speed_lo, self.speed_hi = speed_lo, speed_hi
        self.spec_tmask = spec_tmask
        self.spec_tlen = spec_tlen
        self.spec_fmask = spec_fmask
        self.spec_flen = spec_flen
        self._rirs = None

    def _get_rirs(self):
        if self._rirs is None:
            self._rirs = _make_rirs(n=40, rng=np.random.RandomState(int(self.rng.randint(1, 2**31 - 1))))
        return self._rirs

    def __getitem__(self, i):
        x, ya, yb, _ = self.rows[i]
        if self.aug:
            if self.rng.random() < 0.5:
                x = add_noise(x, self.noise, self.rng, self.snr_min, self.snr_max)
            x = random_gain(x, self.rng, self.gain_db)
            x = time_shift(x, self.rng, self.shift)
            if self.speed_p > 0 and self.rng.random() < self.speed_p:
                x = speed_perturb(x, self.rng, factor=(self.speed_lo, self.speed_hi))
            if self.reverb_p > 0 and self.rng.random() < self.reverb_p:
                x = add_reverb(x, self._get_rirs(), self.rng)
            if self.babble_p > 0:
                x = add_babble(x, self.rng, p=self.babble_p)
            if self.channel_p > 0 and self.rng.random() < self.channel_p:
                x = add_playback_channel(x, self.rng)
        x = center_crop(x, n=_need(self.n_frames))
        mel = wav_to_mel(x, self.n_frames)
        if self.aug and self.specaug_p > 0 and self.rng.random() < self.specaug_p:
            mel = specaug_mel(mel, self.rng, t_mask=self.spec_tmask,
                              t_len=self.spec_tlen, f_mask=self.spec_fmask,
                              f_len=self.spec_flen)
        return (torch.from_numpy(mel)[None], torch.tensor(ya, dtype=torch.long),
                torch.tensor(yb, dtype=torch.long),
                torch.tensor(self.weights[i], dtype=torch.float32))


# ============================================================ general CRNN ====
class CRNN_N(nn.Module):
    """Generalized CRNN: N conv blocks (any channel list) + pool-mel + BiGRU +
    FC + dual heads. Reproduces exp5 CRNN exactly when ch=(32,64,128)."""

    def __init__(self, n_a=C.N_A, n_b=C.N_B, ch=(32, 64, 128), gru=128,
                 gru_layers=2, mid=256):
        super().__init__()
        ch = list(ch)
        layers = []
        prev = 1
        for c in ch:
            layers += [nn.Conv2d(prev, c, 3, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU()]
            prev = c
        self.cnn = nn.Sequential(*layers)
        c2 = ch[-1]
        self.gru = nn.GRU(c2, gru, num_layers=gru_layers, batch_first=True,
                          bidirectional=True)
        gdim = gru * 2
        self.fc = nn.Sequential(nn.Linear(gdim, mid), nn.ReLU(), nn.Dropout(0.3))
        self.head_a = nn.Linear(mid, n_a)
        self.head_b = nn.Linear(mid, n_b)

    def forward(self, x):
        h = self.cnn(x)                       # (B, c2, 40, T)
        h = h.permute(0, 3, 1, 2)             # (B, T, c2, 40)
        hp = h.mean(dim=3)                    # (B, T, c2)
        out, _ = self.gru(hp)                 # (B, T, gru*2*layers)
        feat = out[:, -1, :]
        feat = self.fc(feat)
        return self.head_a(feat), self.head_b(feat)


def label_smooth_ce(logits, targets, eps=0.1):
    """Cross-entropy with label smoothing (regularizer -> better generalization).
    Standard: loss = (1-eps)*CE + eps*mean(log-softmax over all classes)."""
    logp = F.log_softmax(logits, dim=-1)
    n = logits.size(-1)
    smooth = torch.full_like(logp, eps / n)
    smooth.scatter_(1, targets.unsqueeze(1), 1.0 - eps + eps / n)
    return -(smooth * logp).sum(dim=-1).mean()


# ============================================================ main =============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--tag", default="bat")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--real-boost", type=float, default=2.5)
    ap.add_argument("--weak-boost", type=float, default=1.5)
    ap.add_argument("--class-boost", type=float, default=2.0)
    ap.add_argument("--reject-boost", type=float, default=2.0)
    ap.add_argument("--snr-min", type=float, default=12.0)
    ap.add_argument("--snr-max", type=float, default=28.0)
    ap.add_argument("--gain-db", type=float, default=4.0)
    ap.add_argument("--shift", type=float, default=0.10)
    # speed / time-stretch perturbation (suggestion 5: real-voice robustness)
    ap.add_argument("--speed-p", type=float, default=0.0,
                    help="prob of time-stretching a training clip (0=off)")
    ap.add_argument("--speed-lo", type=float, default=0.9)
    ap.add_argument("--speed-hi", type=float, default=1.1)
    # window
    ap.add_argument("--nframes", type=int, default=250)
    # arch
    ap.add_argument("--ch", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--gru", type=int, default=128)
    ap.add_argument("--gru-layers", type=int, default=2)
    ap.add_argument("--label-smooth", type=float, default=0.0,
                    help="label smoothing eps on the intent head (0=off)")
    # data-aug
    ap.add_argument("--reverb-p", type=float, default=0.0)
    ap.add_argument("--babble-p", type=float, default=0.0)
    ap.add_argument("--specaug-p", type=float, default=0.0)
    ap.add_argument("--spec-tmask", type=int, default=1)
    ap.add_argument("--spec-tlen", type=int, default=10)
    ap.add_argument("--spec-fmask", type=int, default=1)
    ap.add_argument("--spec-flen", type=int, default=4)
    ap.add_argument("--channel-p", type=float, default=0.0,
                    help="probability of procedural speaker/mic channel augmentation")
    ap.add_argument("--numerals", type=int, default=0,
                    help="cap on real-voice OOS numerals reject rows")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[bat:{args.tag}] device={device} nframes={args.nframes} "
          f"(win {args.nframes*C.HOP/16000:.2f}s) ch={args.ch} gru={args.gru}x{args.gru_layers} "
          f"ls={args.label_smooth} reverb={args.reverb_p} babble={args.babble_p} "
          f"specaug={args.specaug_p} numerals={args.numerals} rb={args.reject_boost}",
          flush=True)

    weak_names = ["VOLUME_UP", "VOLUME_DOWN", "LIGHT_ON", "LIGHT_OFF", "STOP", "NEXT"]
    weak_set = np.array([C.INTENT2IDX[n] for n in weak_names])

    noise_bank = load_noise_bank()
    tr_rows = load_audio_split(["train", "fil", "neg_train", "neg_noise"])
    if args.numerals > 0:
        num_rows = load_numerals_reject(max_rows=args.numerals)
        tr_rows = tr_rows + num_rows
        print(f"[bat:{args.tag}] train rows after numerals: {len(tr_rows)}", flush=True)
    va_rows = load_audio_split(["test", "tune_oos"])
    print(f"[bat:{args.tag}] train={len(tr_rows)} val={len(va_rows)}", flush=True)

    ya_arr = np.array([r[1] for r in tr_rows])
    real_arr = np.array([r[3] for r in tr_rows])
    rw = row_weights(ya_arr, real_arr, args.real_boost, args.weak_boost, weak_set)
    cw = class_weights(torch.tensor(ya_arr), C.N_A, device, boost=args.class_boost)
    if args.reject_boost != 1.0:
        rej_idx = C.N_A - 1
        cw[rej_idx] = cw[rej_idx] * args.reject_boost
        print(f"[bat:{args.tag}] reject_boost={args.reject_boost} -> cw[{rej_idx}]="
              f"{cw[rej_idx].item():.3f}", flush=True)

    ds_kw = dict(snr_min=args.snr_min, snr_max=args.snr_max, gain_db=args.gain_db,
                 shift=args.shift, n_frames=args.nframes, reverb_p=args.reverb_p,
                 babble_p=args.babble_p, specaug_p=args.specaug_p,
                 spec_tmask=args.spec_tmask, spec_tlen=args.spec_tlen,
                 spec_fmask=args.spec_fmask, spec_flen=args.spec_flen,
                 speed_p=args.speed_p, speed_lo=args.speed_lo,
                 speed_hi=args.speed_hi, channel_p=args.channel_p)
    va = BatDataset(va_rows, noise_bank, aug=False, seed=args.seed, **ds_kw)
    val = torch.utils.data.DataLoader(va, batch_size=256, shuffle=False,
                                      num_workers=args.workers, pin_memory=True)
    tr = BatDataset(tr_rows, noise_bank, aug=True, seed=args.seed, weights=rw, **ds_kw)
    trl = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True,
                                      num_workers=args.workers, pin_memory=True)

    use_ls = args.label_smooth > 0.0
    model = CRNN_N(n_a=C.N_A, n_b=C.N_B, ch=tuple(args.ch), gru=args.gru,
                   gru_layers=args.gru_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[bat:{args.tag}] params={n_params:,}", flush=True)

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
            if use_ls:
                loss = label_smooth_ce(la, ya, eps=args.label_smooth) + ce_b(lb, yb)
            else:
                loss = weighted_ce(la, ya, w, class_w=cw) + ce_b(lb, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * X.size(0); n += X.size(0)
        sched.step()
        vi, vs, ve = evaluate(model, val, device)
        dt = time.time() - t0
        print(f"[bat:{args.tag}] ep{ep:02d} loss={tot/n:.4f} val_intent={vi*100:.2f} "
              f"val_slot={vs*100:.2f} val_e2e={ve*100:.2f}  ({dt:.0f}s)", flush=True)
        if ve > best_e2e:
            best_e2e = ve
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 6:
                print(f"[bat:{args.tag}] early stop", flush=True)
                break

    model.load_state_dict(best_state)
    out = f"{C.OUT_DIR}/vcm_crnn_{args.tag}.pt"
    torch.save({"model": best_state, "config": vars(args),
                "intents": C.INTENTS, "value_tokens": C.VALUE_TOKENS,
                "best_val": best_e2e, "n_params": n_params, "arch": "battery",
                "n_frames": args.nframes, "tag": args.tag, "epochs_run": ep,
                "recipe": f"battery {args.tag}: ch={args.ch} gru={args.gru}x{args.gru_layers} "
                          f"ls={args.label_smooth} reverb={args.reverb_p} babble={args.babble_p} "
                          f"specaug={args.specaug_p} numerals={args.numerals} rb={args.reject_boost}"},
               out)
    print(f"[bat:{args.tag}] saved {out}  best_val_e2e={best_e2e*100:.2f}", flush=True)


if __name__ == "__main__":
    main()
