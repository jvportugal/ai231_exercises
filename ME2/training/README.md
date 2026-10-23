# ME2 model training and export

This directory contains the source, fixed split manifests, retained PyTorch
checkpoints, evaluation programs, and exporters for both models shipped by
ME2. The third-party audio corpora are deliberately not copied into Git: the
MSWC English archive is tens of gigabytes, and the RV90 corpus has
source-specific license and consent conditions. Obtain each corpus from its
publisher and accept its terms before training.

The retained checkpoints provide the shortest reproducibility path: they let
the exporters rebuild the ONNX graphs without retraining. The training scripts
provide the longer path from the published audio. Exact floating-point weights
from a new training run can vary with PyTorch, accelerator, and library
versions even when the recorded seeds and hyperparameters are unchanged.
Verify all retained inputs with `sha256sum -c checksums.sha256` from this
directory.

## Environment

From `ME2/training`:

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
~~~

## Destiny wake-word model

The source corpus is the English portion of the
[MLCommons Multilingual Spoken Words Corpus](https://mlcommons.org/datasets/multilingual-spoken-words/),
released under CC BY 4.0. Download and unpack the official English audio, then
materialize only the clips referenced by the committed split manifests:

~~~bash
python3 wakeword/prepare_mswc.py /path/to/unpacked/mswc/english
python3 wakeword/train_robust.py
python3 wakeword/evaluate_deployment.py
python3 wakeword/export_robust.py --destination wakeword/artifacts
~~~

Files:

- `wakeword/splits/*.json` are the exact speaker-disjoint split manifests;
- `wakeword/build_splits.py` documents how those manifests were generated;
- `wakeword/train_robust.py` contains the architecture, feature pipeline,
  online augmentation, optimizer, seed, early stopping, and evaluation logic;
- `wakeword/evaluate_deployment.py` applies the runtime-style multi-placement
  scoring procedure;
- `wakeword/export_robust.py` exports ONNX opset 17, the mel bank, and metadata;
- `wakeword/checkpoints/destiny_robust_v2.pt` is the retained selected
  checkpoint; and
- the two JSON reports are the retained training and deployment evaluations.

To export without retraining, run only the final `export_robust.py` command.
The exporter verifies PyTorch/ONNX parity and numerical equivalence with the
shipped ONNX graph. ONNX file bytes can differ across PyTorch exporter versions
even when graph outputs are identical.

## RV90 command model

Download the public parquet artifacts from
[AI231 ME2 Voice Commands](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands)
after reviewing the dataset card and source licenses:

~~~bash
python3 vcm/prepare_dataset.py
~~~

The default location is `vcm/data/ai231-me2-voice-commands`. Override it with
`AI231_VCM_DATA=/absolute/path` when using an existing checkout. Training
outputs default to `vcm/checkpoints`; override that location with
`AI231_VCM_OUTPUT=/absolute/path`.

Recreate the X6 base checkpoint:

~~~bash
python3 vcm/battery.py \
  --tag X6_babble --seed 301 --epochs 44 --workers 6 \
  --gru 256 --gru-layers 3 --babble-p 0.3
~~~

Run the selected four-epoch source-balanced refinement:

~~~bash
python3 vcm/finetune_realvoice_safe.py \
  --base vcm_crnn_X6_babble.pt --tag rv_kd \
  --epochs 4 --samples-per-epoch 10000 --batch 64 --threads 8 \
  --lr 2e-5 --cnn-lr-scale 0.1 \
  --real-voice-fraction 0.30 --other-real-fraction 0.20 \
  --reject-fraction 0.15 --babble-p 0.10 --speed-p 0.10 \
  --kd-weight 0.70 --temperature 2.0 --seed 812
~~~

The selected refinement state is epoch 3. Rebuild the final 80/20 interpolated
PyTorch and ONNX bundle:

~~~bash
python3 vcm/export_rv90.py
~~~

The exporter defaults to the committed
`vcm/checkpoints/vcm_crnn_X6_babble.pt` and
`vcm/checkpoints/vcm_crnn_rv_kd_ep03.pt`, verifies the 3,192,807-parameter
architecture, checks PyTorch/ONNX numerical parity, copies the shipped mel
bank, labels, and fixed decoding policy, and writes hashes to
`vcm/artifacts/rv90/export_manifest.json`.

The four WAV files in `vcm/noise` are the exact room-noise bank used by the
primary and refinement loaders. `vcm/finetune_rv_kd.json` records the selected
refinement arguments and per-epoch validation measurements.

## What is and is not stored in Git

Included:

- all source required by the selected training and export paths;
- exact wake-word split manifests;
- exact procedural/local noise inputs;
- retained selected checkpoints and evaluation reports; and
- deployable ONNX, filterbank, label, and policy assets elsewhere in ME2.

Not included:

- third-party source audio/parquet corpora;
- virtual environments and downloaded Python wheels;
- rejected experiment checkpoints, feature caches, or temporary logs; and
- GPU/CPU caches and generated `artifacts` directories.
