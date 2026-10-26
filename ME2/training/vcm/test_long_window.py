import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from long_window_attention import features, runtime, LongWindowAttentionVCM, export_model
from train_temporal_attention import TemporalAttentionVCM


class LongWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def speech(self, seconds):
        t = np.arange(int(seconds*16000))/16000
        x = (.12*np.sin(2*np.pi*233*t) + .04*np.sin(2*np.pi*519*t)).astype(np.float32)
        return np.pad(x, (3200, 3200))

    def test_long_window_keeps_both_ends(self):
        x = self.speech(3.6)
        short, a = runtime.prepare_clip(x)
        long, b = runtime.prepare_clip(x, n_frames=400)
        self.assertEqual(len(short), 39841)
        self.assertEqual(len(long), 63841)
        self.assertEqual(a["active_span"], b["active_span"])
        self.assertGreater(b["centered_start_seconds"], 0.)
        self.assertGreater(np.count_nonzero(long), np.count_nonzero(short))

    def test_masked_normalization_ignores_added_padding(self):
        x = self.speech(.8)
        a, _ = features(x, 400, True)
        b, _ = features(x, 500, True)
        # 100 extra mel frames => 50 padding frames on each side.
        np.testing.assert_allclose(a, b[:, :, 50:450], atol=2e-6)
        self.assertTrue(np.all(a[0, :, a[1, 0] == 0] == 0))

    def test_silence_and_empty_inputs_are_finite(self):
        model = LongWindowAttentionVCM(width=16, auxiliary_words=8,
            center_strength=.5).eval()
        for x in (np.zeros(0, np.float32), np.zeros(1000, np.float32)):
            packed, _ = features(x)
            self.assertTrue(np.isfinite(packed).all())
            with torch.no_grad():
                result = model(torch.from_numpy(packed[None]))
            self.assertTrue(all(torch.isfinite(o).all() for o in result))

    def test_unmasked_model_preserves_baseline(self):
        base = TemporalAttentionVCM(width=16, auxiliary_words=8, structured_slots=True).eval()
        model = LongWindowAttentionVCM(width=16, auxiliary_words=8, masked=False).eval()
        model.load_state_dict(base.state_dict())
        x = torch.randn(3, 1, 40, 250)
        with torch.no_grad():
            a, b = base(x), model(torch.cat((x, torch.ones_like(x)), dim=1))
        for expected, observed in zip(a, b):
            torch.testing.assert_close(expected, observed, atol=1e-6, rtol=1e-5)

    def test_center_prior_broadens_with_support(self):
        model = LongWindowAttentionVCM(width=16, auxiliary_words=8, center_strength=.5)
        support = torch.zeros(2, 50)
        support[0, 20:30] = 1
        support[1, 5:45] = 1
        prior = model.temporal_prior(support)
        self.assertGreater(prior[1, 20].item(), prior[0, 20].item())
        self.assertGreater(prior[1, 5].exp().item(), .6)

    def test_export_parity_dynamic_batch(self):
        import onnxruntime as ort
        model = LongWindowAttentionVCM(width=16, auxiliary_words=8, center_strength=.5).eval()
        packed, _ = features(self.speech(.8))
        x = torch.from_numpy(packed[None])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "long.onnx"
            export_model(model, path, 400)
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            sess = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
            for batch in (1, 3):
                batch_x = x.repeat(batch, 1, 1, 1)
                with torch.no_grad():
                    expected = model(batch_x)
                observed = sess.run(None, {"mel": batch_x.numpy()})
                for a, b in zip(expected, observed):
                    np.testing.assert_allclose(a.numpy(), b, atol=2e-5, rtol=2e-4)


if __name__ == "__main__":
    unittest.main()
