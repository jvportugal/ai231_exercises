import json
import os
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import vcm_infer as V


class _FakeOnnxSession:
    def __init__(self, path, providers=None):
        self.index = 0 if path.endswith("a.onnx") else 1

    def run(self, output_names, feed):
        intent = [
            np.asarray([[2.0, 0.0, -1.0]], np.float32),
            np.asarray([[0.0, 3.0, -1.0]], np.float32),
        ][self.index]
        slot = [
            np.asarray([[1.0, 0.0]], np.float32),
            np.asarray([[0.0, 2.0]], np.float32),
        ][self.index]
        return intent, slot


class EnsembleModelTests(unittest.TestCase):
    def test_probability_ensemble_matches_weighted_softmax(self):
        fake_ort = types.SimpleNamespace(InferenceSession=_FakeOnnxSession)
        with tempfile.TemporaryDirectory() as directory:
            manifest = {
                "type": "probability_ensemble_onnx_v1",
                "members": [
                    {"name": "a", "model": "a.onnx", "weight": 0.25},
                    {"name": "b", "model": "b.onnx", "weight": 0.75},
                ],
            }
            path = os.path.join(directory, "ensemble.json")
            with open(path, "w") as handle:
                json.dump(manifest, handle)
            with mock.patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                session = V.load_model(directory)
                actual = session.run(
                    None, {"mel": np.zeros((1, 1, 40, 250), np.float32)})

        logits_a = np.asarray([[2.0, 0.0, -1.0]], np.float32)
        logits_b = np.asarray([[0.0, 3.0, -1.0]], np.float32)
        expected = (0.25 * session._softmax_batch(logits_a)
                    + 0.75 * session._softmax_batch(logits_b))
        np.testing.assert_allclose(np.exp(actual[0]), expected, rtol=1e-6)

    def test_invalid_weight_sum_is_rejected(self):
        fake_ort = types.SimpleNamespace(InferenceSession=_FakeOnnxSession)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ensemble.json")
            with open(path, "w") as handle:
                json.dump({
                    "type": "probability_ensemble_onnx_v1",
                    "members": [
                        {"name": "a", "model": "a.onnx", "weight": 0.2},
                        {"name": "b", "model": "b.onnx", "weight": 0.2},
                    ],
                }, handle)
            with mock.patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                with self.assertRaisesRegex(ValueError, "sum to 1"):
                    V.load_model(directory)


if __name__ == "__main__":
    unittest.main()
