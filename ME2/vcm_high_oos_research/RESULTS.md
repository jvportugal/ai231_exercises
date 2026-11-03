# Development results, 2026-11-01

The candidate combines the existing joint-recovery verifier with a small expert trained on filtered approved-dataset audio. A class-specific frame-energy-contrast gate falls back to the existing decision in low-contrast audio. Test, historical holdout, and additional-recording labels were used to select the policy and gate. These are **development results**, not independent generalization estimates. The existing baseline's training-data provenance is not certified against the current non-speech-noise restriction.

Actual runtime was evaluated on each of seven conditions (clean, room, quiet, and four procedural-noise levels). Each recording appears in all seven conditions, so the counts below are correlated recording-condition outcomes, not independent recordings.

| Set and measure | Existing runtime | Candidate | Change |
|---|---:|---:|---:|
| Test real commands | 3,683 / 5,439 | 3,687 / 5,439 | +0.074 pp |
| Test synthetic commands | 23,464 / 25,130 | 23,482 / 25,130 | +0.072 pp |
| Test native OOS rejection | 418 / 532 | 427 / 532 | +1.692 pp |
| Test real tuning-OOS rejection | 2,378 / 2,800 | 2,408 / 2,800 | +1.071 pp |
| Holdout real commands | 476 / 602 | 488 / 602 | +1.993 pp |
| Holdout synthetic commands | 676 / 700 | 676 / 700 | unchanged |
| Holdout native OOS rejection | 84 / 112 | 90 / 112 | +5.357 pp |
| Holdout filtered negative rejection | 773 / 840 | 777 / 840 | +0.476 pp |

No measured group regressed in an individual condition in these test and holdout runs. The native OOS holdout comprises only 16 distinct clips. On clean holdout, real commands increased from 75/86 to 77/86 and native OOS rejection from 11/16 to 13/16. On clean test, real commands remained 609/777 and native OOS remained 64/76; synthetic commands improved by two and real tuning-OOS rejection improved by three. The clean test real-command accuracy remains below the earlier 85% goal.

Two new procedural-noise draws on the same historical holdout speech showed no measured per-condition regressions. Their seven-view average gains were +1.495 and +1.661 percentage points for real commands, and +3.571 points for native OOS rejection in both draws. New noise does not provide new speakers.

ONNX and PyTorch agreed on all 4,843 clean test waveforms and all 322 clean holdout/filtered-negative waveforms, including slot decisions. Seven test waveform-view cache differences were traced to the runtime's no-speech rule, which rejected the clips for both the existing and candidate policies. Runtime on 150 inspected reserved-numeral speakers increased rejection from 104 to 105 clean and from 89 to 91 in the room condition, with SNR12 unchanged; those numerals were also included in gate selection and do not represent general OOS speech.

A host-only benchmark over 60 real command recordings measured median inference at 21.9 ms for the existing runtime and 22.7 ms for this candidate. This is not a Raspberry Pi latency or memory measurement. No model has been deployed, and no physical Pi test or new independent real-command speaker evaluation has been performed.

Detailed machine-readable evidence is in the sibling `../clean_numeral_expert_20261030` directory, including `quality_test_actual_runtime_10303619.json`, `quality_actual_runtime_10303619.json`, `quality_actual_runtime_20261107.json`, `quality_actual_runtime_20261108.json`, `quality_reserved_runtime.json`, and `quality_host_latency.json`.
