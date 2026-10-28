# Selective-recovery VCM for Raspberry Pi testing

This opt-in bundle combines the shipped short/long models with a small
acoustic expert. It can recover a rejected command when the expert supports
it, and applies a class-specific scope verifier to accepted commands. The
bundle contains 3,172,055 parameters, within the existing 3,192,807 cap.
Speech, OOS examples and transcript supervision came from the approved AI231
dataset; acoustic augmentations used procedural non-speech noise.

From the repository's `ME2` directory:

```bash
.venv/bin/python vcm_action_runner/runner.py --wakeword --term \
  --no-wake-chime --benchmark --vcm-bundle vcm_benchmark_joint_recovery
```

Benchmark mode suppresses real actions and records decisions under
`~/vcm_benchmark/`. The older `vcm_benchmark_long_window` bundle remains
available for comparison. Pull the complete repository commit: the new runtime
logic in `vcm_infer.py`, `vcm_long_window.py` and `vcm_selective_scope.py` is
required along with these model files.

The candidate was fitted to previously examined TEST development partitions;
the previously examined HOLDOUT informed the design and is not independent.
On a second procedural-noise draw, average changes across seven conditions
were +2.64 percentage points command accuracy, +2.08 real-command accuracy,
+6.67 native OOS rejection and +0.80 additional synthetic OOS rejection.
Historical HOLDOUT averages were +0.77 command, +0.00 real command, +1.79
native OOS and +0.40 synthetic OOS. Some individual conditions regress: in
clean HOLDOUT one real command was newly rejected, and in the room condition
one of 16 native OOS examples was newly accepted. These offline results are
not physical Pi measurements; test latency and behavior on the device.

`verification.json` documents the runtime parity checks and evaluation limits;
`checksums.json` lists the bundle assets.
