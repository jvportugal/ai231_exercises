# Frozen attention ensemble for Raspberry Pi benchmarking

This is the TEST-selected VCM candidate. It does not replace `vcm_model/`.

- 52% KD144 probability
- 48% real-heavy epoch-26 probability
- REJECT log-probability offset: `-0.03534795343875885`
- input: `[batch, 1, 40, 250]`
- total parameters: 3,158,256 across two ONNX forward passes

Selection used TEST clean/channel and tune-OOS only. The weights and reject
offset were frozen before HOLDOUT evaluation. Do not tune them from the
physical HOLDOUT benchmark.

From the `ME2` directory, run:

```bash
python3 vcm_action_runner/runner.py --voice --benchmark \
  --vcm-bundle vcm_benchmark_attention_ensemble
```

For wake-word benchmarking, replace `--voice` with `--wakeword`. The benchmark
mode logs decisions and suppresses real device actions. Compare against the
unchanged bundled baseline using the exact same captured audio.
