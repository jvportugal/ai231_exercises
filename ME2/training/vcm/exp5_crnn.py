"""EXP 5 (conditional): small CRNN = CNN feature extractor + lightweight GRU.

Only run because the plan asks for a temporal-structure probe. The audit (Exp 1)
did NOT implicate temporal structure (duration already fixed by centering) and
the DS-CNN (Exp 3, a stronger temporal mixer) lost badly — so this is expected
to hit the same data ceiling. Small footprint, from scratch, same recipe.

Backbone: 3x standard Conv2d (32->64->128) on (1,40,150) -> permute to
(frames, channels) -> GRU(128, 256, 2-layer, bidir) -> last-hidden concat ->
FC -> head_a(20) + head_b(19). Target ~<1.2M params.

Usage:
  CUDA_VISIBLE_DEVICES=0 python exp5_crnn.py --epochs 36 --tag crnn1
"""
import argparse, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import config as C
from train_centered import (load_noise_bank, load_audio_split,
                            CenteredAugDataset, row_weights, weighted_ce, evaluate)
from train import _class_weights as class_weights


class CRNN(nn.Module):
    def __init__(self, n_a=C.N_A, n_b=C.N_B,
                 ch=(32, 64, 128), gru=128, gru_layers=2, mid=256):
        super().__init__()
        c0, c1, c2 = ch
        self.cnn = nn.Sequential(
            nn.Conv2d(1, c0, 3, 3, padding=1), nn.BatchNorm2d(c0), nn.ReLU(),
            nn.Conv2d(c0, c1, 3, 3, padding=1), nn.BatchNorm2d(c1), nn.ReLU(),
            nn.Conv2d(c1, c2, 3, 3, padding=1), nn.BatchNorm2d(c2), nn.ReLU(),
        )
        # input to GRU: (batch, frames=150, channels=c2)
        self.gru = nn.GRU(c2, gru, num_layers=gru_layers, batch_first=True,
                          bidirectional=True)
        gdim = gru * 2  # bidirectional: 2 dirs at final layer
        self.fc = nn.Sequential(nn.Linear(gdim, mid), nn.ReLU(), nn.Dropout(0.3))
        self.head_a = nn.Linear(mid, n_a)
        self.head_b = nn.Linear(mid, n_b)

    def forward(self, x):
        h = self.cnn(x)                      # (B, c2, 40, 150)
        h = h.permute(0, 3, 1, 2)            # (B, 150, c2, 40)
        h = h.mean(dim=3)                    # (B, 150, c2) pool over mel bins
        out, _ = self.gru(h)                 # (B, 150, gru*2*layers)
        feat = out[:, -1, :]                 # last timestep
        feat = self.fc(feat)
        return self.head_a(feat), self.head_b(feat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=36)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--tag", default="crnn1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--real-boost", type=float, default=2.5)
    ap.add_argument("--weak-boost", type=float, default=1.5)
    ap.add_argument("--class-boost", type=float, default=2.0)
    ap.add_argument("--reject-boost", type=float, default=1.0,
                    help="extra multiplier on the REJECT class weight (idx 19)")
    ap.add_argument("--snr-min", type=float, default=12.0)
    ap.add_argument("--snr-max", type=float, default=28.0)
    ap.add_argument("--gain-db", type=float, default=4.0)
    ap.add_argument("--shift", type=float, default=0.10)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[crnn] device={device}", flush=True)

    weak_names = ["VOLUME_UP", "VOLUME_DOWN", "LIGHT_ON", "LIGHT_OFF", "STOP", "NEXT"]
    weak_set = np.array([C.INTENT2IDX[n] for n in weak_names])

    noise_bank = load_noise_bank()
    tr_rows = load_audio_split(["train", "fil", "neg_train", "neg_noise"])
    va_rows = load_audio_split(["test", "tune_oos"])
    kw = dict(snr_min=args.snr_min, snr_max=args.snr_max, gain_db=args.gain_db, shift=args.shift)
    va = CenteredAugDataset(va_rows, noise_bank, aug=False, seed=args.seed)
    val = torch.utils.data.DataLoader(va, batch_size=256, shuffle=False,
                                      num_workers=args.workers, pin_memory=True)
    print(f"[crnn] train={len(tr_rows)} val={len(va_rows)}", flush=True)

    ya_arr = np.array([r[1] for r in tr_rows])
    real_arr = np.array([r[3] for r in tr_rows])
    rw = row_weights(ya_arr, real_arr, args.real_boost, args.weak_boost, weak_set)
    cw = class_weights(torch.tensor(ya_arr), C.N_A, device, boost=args.class_boost)
    if args.reject_boost != 1.0:
        rej_idx = C.N_A - 1  # REJECT is the last class
        cw[rej_idx] = cw[rej_idx] * args.reject_boost
        print(f"[crnn] reject_boost={args.reject_boost} -> cw[{rej_idx}]={cw[rej_idx].item():.3f}", flush=True)

    tr = CenteredAugDataset(tr_rows, noise_bank, aug=True, seed=args.seed, weights=rw, **kw)
    trl = torch.utils.data.DataLoader(tr, batch_size=args.batch, shuffle=True,
                                      num_workers=args.workers, pin_memory=True)

    model = CRNN(n_a=C.N_A, n_b=C.N_B).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[crnn] params={n_params:,}", flush=True)

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
        print(f"[crnn] ep{ep:02d} loss={tot/n:.4f} val_intent={vi*100:.2f} "
              f"val_slot={vs*100:.2f} val_e2e={ve*100:.2f}  ({dt:.0f}s)", flush=True)
        if ve > best_e2e:
            best_e2e = ve
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 6:
                print("[crnn] early stop", flush=True)
                break

    out = os.path.join(C.OUT_DIR, f"vcm_crnn_{args.tag}.pt")
    torch.save({"model": best_state, "n_a": C.N_A, "n_b": C.N_B,
                "intents": C.INTENTS, "value_tokens": C.VALUE_TOKENS,
                "best_val": best_e2e, "n_params": n_params, "arch": "CRNN",
                "tag": args.tag, "epochs_run": ep,
                "recipe": "CRNN(CNN+GRU) + centered + weighting + runB noise"}, out)
    print(f"[crnn] saved {out}  best_val_e2e={best_e2e*100:.2f}", flush=True)


if __name__ == "__main__":
    main()
