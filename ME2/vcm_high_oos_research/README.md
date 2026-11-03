# High-OOS research model

This experimental bundle contains the four ONNX models, frontend, and the eight-stage feature-warp policy used for the clean waveform results below. It is separate from the Raspberry Pi runner and does not replace the existing bundle. The policy may perform multiple model passes and requires PyTorch, NumPy, and ONNX Runtime. Raspberry Pi latency, memory use, and behavior have not been measured.

| Previously reused development recordings | Real commands | All valid commands | Native OOS rejected |
| --- | ---: | ---: | ---: |
| TEST | 663/777 (85.33%) | 4221/4367 (96.66%) | 70/76 (92.11%) |
| HOLDOUT | 78/86 (90.70%) | 177/186 (95.16%) | 16/16 (100%) |

Both sets and their labels were repeatedly used to select and tune the policy; these numbers are not independent generalization estimates. The holdout OOS set has only 16 recordings. The underlying historical weights have uncertified noise provenance. The clean policy has substantial regressions in simulated noise and room conditions, described in `RESULTS_20261102.md`; it should not be treated as a validated noise-robust deployment.

For research inference, from this directory with the dependencies installed:

```python
import numpy as np
from fitted_policy import FittedPolicy

model = FittedPolicy()
# waveform: mono 16 kHz float32 samples, with speech already captured
intent, confidence, probabilities, info = model.classify(waveform)
```

`original_waveform_replay.py` is the original experiment script preserved for provenance; it depends on the original experiment workspace and dataset paths. The JSON files preserve the fitted thresholds and measured decisions. `research_checksums.json` covers the original model assets, and `RESULTS_20261102.md` documents the broader experiments and limitations.
