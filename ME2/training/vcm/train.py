"""Train the two-head VCM classifier on the golden HF dataset.

Replicates the old architecture (1.62M params) from
vcm_work/model_twohead/vcm_twohead_best_ai231.pt:
  Conv2d(1,32,3,3)->BN->ReLU->MP2
  Conv2d(32,32,3,3)->BN->ReLU->MP2
  Conv2d(32,64,3,3)->BN->ReLU->MP2
  Conv2d(64,64,3,3)->BN->ReLU->MP2
  Conv2d(64,128,3,3)->BN->ReLU->MP2
  flatten (128*9*10=11520) -> Linear(11520,128)
  head_a: Linear(128,20)  intent(+reject)
  head_b: Linear(128,19)  slot-token(+reject)
Loss = CE(head_a) + CE(head_b).  head_b is only meaningful when head_a is a
slotted intent; we still supervise it (fixed/OOS -> reject) which is what the
old model did.
"""
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config as C


def _class_weights(y, n_cls, device, boost=1.0):
    """Inverse-frequency class weights, clamped to [1, boost]."""
    counts = torch.bincount(y, minlength=n_cls).clamp(min=1).float()
    w = counts.sum() / (n_cls * counts)
    w = w.clamp(max=boost)
    return w.to(device)


class TwoHeadCNN(nn.Module):
    def __init__(self, n_a=C.N_A, n_b=C.N_B):
        super().__init__()
        # 3 same 3x3 convs on (1,40,150) -> spatial (2,6) -> 128*2*6=1536
        self.conv = nn.Sequential(
            nn.Conv2d(1, 48, 3, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.Conv2d(48, 96, 3, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
            nn.Conv2d(96, 192, 3, 3, padding=1), nn.BatchNorm2d(192), nn.ReLU(),
        )
        self.bottleneck = nn.Sequential(
            nn.Linear(192 * 2 * 6, 256), nn.ReLU(), nn.Dropout(0.3),
        )
        self.head_a = nn.Linear(256, n_a)
        self.head_b = nn.Linear(256, n_b)

    def forward(self, x):
        h = self.conv(x)
        h = torch.flatten(h, 1)
        h = self.bottleneck(h)
        return self.head_a(h), self.head_b(h)


def load_npz(prefix):
    z = np.load(f"{C.OUT_DIR}/splits_{prefix}.npz")
    X = torch.from_numpy(z["X"]).float()
    ya = torch.from_numpy(z["ya"]).long()
    yb = torch.from_numpy(z["yb"]).long()
    return TensorDataset(X, ya, yb)


def run_epoch(model, loader, opt=None, device=None):
    if device is None: device = torch.device('cpu')
    model.train(opt is not None)
    tot = 0.0
    n = 0
    for X, ya, yb in loader:
        X, ya, yb = X.to(device), ya.to(device), yb.to(device)
        if opt is not None:
            opt.zero_grad()
            la, lb = model(X)
            loss = F.cross_entropy(la, ya) + F.cross_entropy(lb, yb)
            loss.backward()
            opt.step()
        else:
            with torch.no_grad():
                la, lb = model(X)
                loss = F.cross_entropy(la, ya) + F.cross_entropy(lb, yb)
        tot += loss.item() * X.size(0)
        n += X.size(0)
    return tot / n


def evaluate(model, loader, device=None):
    if device is None: device = torch.device('cpu')
    model.eval()
    ya_all, yb_all, pa_all, pb_all = [], [], [], []
    for X, ya, yb in loader:
        X = X.to(device)
        with torch.no_grad():
            la, lb = model(X)
        pa_all.append(la.argmax(1).cpu())
        pb_all.append(lb.argmax(1).cpu())
        ya_all.append(ya)
        yb_all.append(yb)
    ya = torch.cat(ya_all); yb = torch.cat(yb_all)
    pa = torch.cat(pa_all); pb = torch.cat(pb_all)
    acc_a = (pa == ya).float().mean().item()
    acc_b = (pb == yb).float().mean().item()
    # end-to-end: intent correct AND (slot correct OR no-slot expected)
    e2e = ((pa == ya) & ((pb == yb) | (yb == C.N_VALUE))).float().mean().item()
    return acc_a, acc_b, e2e


def main():
    global device, F
    F = torch.nn.functional
    torch.manual_seed(C.SEED)
    np.random.seed(C.SEED)
    torch.set_num_threads(C.THREADS)
    device = torch.device("cpu")
    print(f"[golden] device={device} threads={C.THREADS} seed={C.SEED}")

    tr = load_npz("tr")
    va = load_npz("va")
    tr_dl = DataLoader(tr, batch_size=C.BATCH, shuffle=True)
    va_dl = DataLoader(va, batch_size=256, shuffle=False)

    model = TwoHeadCNN().to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"[golden] params={nparams/1e6:.2f}M  N_A={C.N_A} N_B={C.N_B}  "
          f"train={len(tr)} val={len(va)}")

    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C.EPOCHS)

    best_va = float("inf")
    best = None
    t0 = time.time()
    log_rows = []
    for ep in range(1, C.EPOCHS + 1):
        tl = run_epoch(model, tr_dl, opt, device)
        sched.step()
        vl = run_epoch(model, va_dl, None, device)
        acc_a, acc_b, e2e = evaluate(model, va_dl, device)
        dt = time.time() - t0
        star = ""
        if vl < best_va:
            best_va = vl
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
            star = "  *"
        print(f"[golden] ep{ep:02d} loss={tl:.3f} val={vl:.3f} "
              f"accA={acc_a:.4f} accB={acc_b:.4f} e2e={e2e:.4f} ({dt:.0f}s){star}")
        log_rows.append(f"{ep}\t{tl:.4f}\t{vl:.4f}\t{acc_a:.4f}\t{acc_b:.4f}\t{e2e:.4f}")

    # restore best
    model.load_state_dict(best)
    model.eval()
    acc_a, acc_b, e2e = evaluate(model, va_dl, device)
    print(f"\n[golden] BEST val={best_va:.4f}  accA={acc_a:.4f} accB={acc_b:.4f} e2e={e2e:.4f}")

    # ---- detailed per-intent test breakdown ----
    print("\n[golden] ===== TEST (val split) PER-INTENT =====")
    Xs, yas, ybs = [], [], []
    for X, ya, yb in va_dl:
        Xs.append(X); yas.append(ya); ybs.append(yb)
    Xs = torch.cat(Xs); yas = torch.cat(yas); ybs = torch.cat(ybs)
    with torch.no_grad():
        la, lb = model(Xs.to(device))
    pa = la.argmax(1).cpu(); pb = lb.argmax(1).cpu()
    names = C.INTENTS + [C.REJECT]
    print(f"{'intent':18s} {'acc':>7s}  (n)")
    for i, nm in enumerate(names):
        mask = (yas == i)
        if mask.sum() == 0:
            continue
        acc = (pa[mask] == i).float().mean().item()
        print(f"{nm:18s} {acc*100:6.1f}%  ({int(mask.sum())})")

    # save
    ckpt = {
        "model": model.state_dict(),
        "intents": C.INTENTS,
        "value_tokens": C.VALUE_TOKENS,
        "n_a": C.N_A, "n_b": C.N_B,
        "n_flat": C.N_A, "reject": C.N_INTENT,
        "tag": "golden_hf",
        "epoch": C.EPOCHS, "val_loss": best_va,
        "accA": acc_a, "accB": acc_b, "e2e": e2e,
        "dataset": "airimonda/ai231-me2-voice-commands (train+test only)",
    }
    out = f"{C.OUT_DIR}/vcm_twohead_golden.pt"
    torch.save(ckpt, out)
    print(f"\n[golden] saved -> {out}")
    with open(f"{C.OUT_DIR}/train_log_golden.tsv", "w") as f:
        f.write("epoch\tloss\tval_loss\tacc_a\tacc_b\te2e\n")
        f.write("\n".join(log_rows) + "\n")
    print("[golden] DONE.")


if __name__ == "__main__":
    main()
