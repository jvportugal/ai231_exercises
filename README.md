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

## ME2 — End-to-End Voice Command Model

A Raspberry Pi-oriented, hands-free voice-control pipeline. A compact local
model detects the wake word **“Destiny,”** plays an acknowledgement chime,
captures one command, and uses a two-head ONNX voice-command model to predict
one of 19 actions plus any required slot value.

| Item | Detail |
|------|--------|
| Pipeline | Destiny wake word → endpointed command → intent/slot VCM → action |
| Models | Bundled 73K-parameter wake CNN + 3.19M-parameter CRNN command model |
| Runtime | Local CPU inference with ONNX Runtime; no cloud speech recognition |
| Actions | 19 intents covering media, weather, time, lighting, timers, reminders, communication, and climate |
| Hardware | Designed for Raspberry Pi 5; optional RGB LED on BCM 17/27/22 |
| Interface | Game Boy-inspired GUI with a terminal fallback |

### Files

- [ME2/README.md](ME2/README.md) — comprehensive architecture, hardware,
  setup, usage, testing, troubleshooting, datasets, augmentation, and model
  training documentation.
- [ME2/setup.sh](ME2/setup.sh) — installs the runtime in an isolated virtual
  environment and verifies both bundled models.
- [ME2/vcm_action_runner/](ME2/vcm_action_runner/) — the complete
  wake-word-to-action application.
- [ME2/wakeword_model/](ME2/wakeword_model/) and
  [ME2/vcm_model/](ME2/vcm_model/) — deployable ONNX models and their exact
  preprocessing/label assets.

> ME2 includes every project-specific asset required to run the pipeline.
> Internet access is required during setup for standard dependencies and for
> live weather; weather falls back to an explicitly labelled simulation when
> offline.
