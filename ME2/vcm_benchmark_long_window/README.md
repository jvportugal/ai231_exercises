# Short/long VCM bundle for Raspberry Pi testing

This bundle combines the existing 2.49-second structured attention model with
its masked, centered 4.99-second continuation (seed 28101, epoch 6). Predictions
are blended 50/50. Both ONNX models run on CPU without PyTorch. The existing
0.8-second trailing-silence capture setting still applies.

From the repository's `ME2` directory, after the usual setup:

```bash
.venv/bin/python vcm_action_runner/runner.py --wakeword --term \
  --no-wake-chime --benchmark --vcm-bundle vcm_benchmark_long_window
```

Benchmark mode records decisions under `~/vcm_benchmark/` and suppresses real
actions. Remove `--benchmark` to exercise the devices. The previous bundle is
still available with `--vcm-bundle vcm_benchmark_structured_slots`.

The runtime changes shipped with this bundle are required: `vcm_infer.py` and
`vcm_long_window.py` supply the longer frontend and calibrated intent gates.
Pull the complete commit rather than copying only the ONNX files.

## Selection and limits

This is the current choice for the balance of development accuracy and rejection
of unsupported commands, not a winner on every metric. The frozen configuration
was selected on TEST FIT and tune-OOS FIT. Full TEST is development performance;
the approved dataset HOLDOUT has been examined in previous experiments.

| Offline clip metric | TEST | HOLDOUT |
| --- | ---: | ---: |
| Overall intent + slot accuracy, including OOS | 94.01% | 92.08% |
| Real-audio accuracy, including OOS | 77.31% | 85.42% |
| Native OOS rejection | 78.95% (60/76) | 68.75% (11/16) |

Additional synthetic OOS test rejection is 92.0% (230/250). Versus the earlier
five-second candidate, TEST real accuracy improved 0.97 percentage points and
HOLDOUT real accuracy fell by one net clip (1.04 points). Live room accuracy and
Raspberry Pi latency remain to be measured. These metrics are offline clip
results, not physical Pi results.

`verification.json` records the frozen policy, checkpoint hash, comparison
metrics, ONNX parity, and runtime waveform checks. `checksums.json` identifies
all bundle assets. The model pair stays within the existing 3,192,807 parameter
budget. No new training or calibration was performed for this shipment.
