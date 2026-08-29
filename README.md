# AI 231 — Machine Exercises

Repository for the AI 231 course machine exercises.

## ME1 — 3-Layer CNN for MNIST (einops / einsum only)

A 3-layer CNN (Conv → Conv → Linear) for MNIST digit classification, trained for
5 epochs. **Every** convolutional and linear operation is implemented explicitly
with `einops` / `torch.einsum` — no `nn.Conv2d`, `nn.Linear`, `nn.MaxPool2d`,
`nn.BatchNorm2d`, `F.conv2d`, or `F.cross_entropy` are used for the network or loss.

| Item | Detail |
|------|--------|
| Architecture | Conv(32, 3×3) → MaxPool(2×2) → Conv(64, 3×3) → Flatten → Linear(10) |
| Ops | `einops` rearrange/unfold + `torch.einsum` throughout |
| Training | Adam, lr=1e-3, batch=128, 5 epochs, CPU |
| **Test accuracy** | **98.54%** |

### Files
- [`ME1/ME1_cnn_einops_executed.ipynb`](ME1/ME1_cnn_einops_executed.ipynb) — fully executed
  notebook with code, training logs, learning curves, 4×4 sample-image grid,
  requirements checklist, and action log.
- [`ME1/ME1_cnn_einops.ipynb`](ME1/ME1_cnn_einops.ipynb) — source notebook (unexecuted).

> The MNIST dataset is downloaded automatically by the notebook into
> `ME1/mnist_data/` (gitignored).
