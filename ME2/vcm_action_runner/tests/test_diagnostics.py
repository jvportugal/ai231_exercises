import json
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from vcm_action_runner.diagnostics import DiagnosticRecorder
from vcm_action_runner.voice import Voice


class DiagnosticTests(unittest.TestCase):
    def test_records_exact_audio_and_both_ensemble_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "ensemble.json").write_text(json.dumps({
                "type": "long_window_ensemble_onnx_v1",
                "long_model": "long.onnx", "short_model": "short.onnx"}))
            for name in ("long.onnx", "short.onnx", "labels.json", "policy.json"):
                (bundle / name).write_text("test")
            recorder = DiagnosticRecorder(root / "diagnostics", bundle,
                bundle / "labels.json", bundle / "policy.json", root)
            x = np.array([0, -.123456, .987654], np.float32)
            recorder.record(x, {"ok": False, "rejection_reason": "low_intent_margin"},
                            {"span": (0., 1.), "min_intent_margin": .3})
            np.testing.assert_array_equal(np.load(recorder.directory / "command_0001.npy"), x)
            with wave.open(str(recorder.directory / "command_0001.wav")) as w:
                self.assertEqual(w.getframerate(), 16000)
                self.assertEqual(w.getnframes(), len(x))
            metadata = json.loads((recorder.directory / "session.json").read_text())
            self.assertIn(str(bundle / "long.onnx"), metadata["sha256"])
            self.assertIn(str(bundle / "short.onnx"), metadata["sha256"])
            detail = json.loads((recorder.directory / "command_0001.json").read_text())
            self.assertEqual(detail["result"]["rejection_reason"], "low_intent_margin")
            # Reusing the parent must create a new session, never overwrite recordings.
            other = DiagnosticRecorder(root / "diagnostics", bundle,
                bundle / "labels.json", bundle / "policy.json", root)
            self.assertNotEqual(recorder.directory, other.directory)

    def test_diagnostics_disabled_by_default(self):
        voice = Voice()
        self.assertIsNone(voice._diagnostics_dir)
        self.assertIsNone(voice._diagnostics)


if __name__ == "__main__":
    unittest.main()
