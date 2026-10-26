"""Opt-in long-window frontend/session; no PyTorch dependency at inference."""
import json
from pathlib import Path

import numpy as np
import vcm_infer as runtime


def features(waveform, n_frames=400, masked=True):
    clip, info = runtime.prepare_clip(waveform, n_frames=n_frames)
    support = np.ones(n_frames, np.float32)
    duration = info.get("active_speech_seconds", 0.)
    if masked and info.get("active_span") is not None:
        midpoint = (len(clip) // 2) / runtime.SAMPLE_RATE
        start = midpoint - duration / 2 - .20
        end = midpoint + duration / 2 + .20
        times = np.arange(n_frames) * runtime.HOP / runtime.SAMPLE_RATE
        support = ((times + runtime.WIN / runtime.SAMPLE_RATE >= start)
                   & (times <= end)).astype(np.float32)
    mel = runtime.audio_to_mel(clip, n_frames=n_frames,
        normalization_mask=support if masked else None)[0]
    if masked:
        mel *= support[None, None]
    packed = np.concatenate((mel, np.broadcast_to(support, mel.shape)), axis=0)
    return packed.astype(np.float32), info


class LongWindowSession:
    """A shared waveform produces each member's own centered input window."""
    def __init__(self, manifest_path):
        import onnxruntime as ort
        path = Path(manifest_path)
        self.manifest = json.loads(path.read_text())
        if self.manifest.get("type") != "long_window_ensemble_onnx_v1":
            raise ValueError("unsupported long-window manifest")
        self.n_frames = int(self.manifest["n_frames"])
        self.masked = bool(self.manifest["masked"])
        self.weight_a = float(self.manifest["weight_a"])
        self.adaptive = bool(self.manifest["adaptive"])
        if self.n_frames < 250 or not 0 <= self.weight_a <= 1:
            raise ValueError("invalid long-window configuration")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.long = ort.InferenceSession(str(path.parent / self.manifest["long_model"]),
            sess_options=options, providers=["CPUExecutionProvider"])
        self.short = None
        if self.weight_a > 0 or self.adaptive:
            self.short = ort.InferenceSession(str(path.parent / self.manifest["short_model"]),
                sess_options=options, providers=["CPUExecutionProvider"])
        expected = self.long.get_inputs()[0].shape
        if expected[1:] != [2, 40, self.n_frames]:
            raise ValueError("long model/frontend shape mismatch")

    def prepare_feed(self, waveform, verbose=False):
        packed, info = features(waveform, self.n_frames, self.masked)
        feed = {"long": packed[None],
                "duration": np.asarray([info.get("active_speech_seconds", 0.)])}
        if self.short is not None:
            short, _ = runtime.prepare_clip(waveform)
            feed["short"] = runtime.audio_to_mel(short)
        info["model_window_seconds"] = (1 + (self.n_frames-1)*runtime.HOP)/runtime.SAMPLE_RATE
        info["model_windows_seconds"] = (
            [runtime.N_SAMPLES / runtime.SAMPLE_RATE] if self.short is not None else []
        ) + [info["model_window_seconds"]]
        return feed, info

    def run(self, output_names, feed):
        if output_names is not None:
            raise ValueError("named outputs are not supported")
        activations = ("softmax", "softmax", "grouped_softmax_3", "sigmoid")
        long = [runtime.ProbabilityEnsembleSession._activate(a, k) for a, k in
                zip(self.long.run(None, {"mel": feed["long"]}), activations)]
        w = self.weight_a
        if self.adaptive:
            w = (w + (.75-w)*np.clip((2.49-feed["duration"])/.8, 0, 1))[:, None]
        if self.short is not None:
            short = [runtime.ProbabilityEnsembleSession._activate(a, k) for a, k in
                zip(self.short.run(None, {"mel": feed["short"]}), activations)]
            long = [w*a + (1-w)*b for a, b in zip(short, long)]
        result = [np.log(np.maximum(a, 1e-12)).astype(np.float32) for a in long[:3]]
        p = np.clip(long[3], 1e-6, 1-1e-6)
        return result + [np.log(p/(1-p)).astype(np.float32)]
