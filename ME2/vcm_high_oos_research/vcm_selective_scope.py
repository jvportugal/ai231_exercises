"""Opt-in selective recovery and scope verifier for long-window bundles."""
from pathlib import Path
import numpy as np
import onnxruntime as ort
import vcm_infer as runtime
from vcm_long_window import LongWindowSession


class _Outputs:
    def __init__(self, outputs, info):
        self.outputs, self.info = outputs, info

    def prepare_feed(self, waveform, verbose=False):
        return {}, dict(self.info)

    def run(self, output_names, feed):
        return self.outputs


class SelectiveScopeSession(LongWindowSession):
    def __init__(self, manifest_path):
        super().__init__(manifest_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.expert = ort.InferenceSession(
            str(Path(manifest_path).parent / self.manifest["expert_model"]),
            sess_options=options, providers=["CPUExecutionProvider"])
        self.scope_policy = self.manifest["scope_policy"]
        if (self.scope_policy.get("verifier") != "scope" or
                not self.scope_policy.get("rescue_only") or
                len(self.scope_policy["thresholds"]) != 19 or
                len(self.scope_policy["rescue_thresholds"]) != 19):
            raise ValueError("unsupported selective scope policy")

    def classify_audio(self, labels, waveform, verbose=False, policy=None):
        feed, info = self.prepare_feed(waveform, verbose=verbose)
        baseline = super().run(None, feed)
        original = runtime.classify(_Outputs(baseline, info), labels, waveform,
                                    policy=policy)
        original_ok, _ = runtime.safe_decision(original[0], original[1], original[3])
        if info.get("span") is None:
            return original
        expert = self.expert.run(None, {"mel": feed["long"]})
        if len(expert) < 5:
            raise RuntimeError("scope expert must expose five outputs")
        chosen = original
        if not original_ok:
            weight = float(self.scope_policy["weight"])
            temperature = float(self.scope_policy["temperature"])
            fused = []
            for index in range(3):
                a = np.asarray(baseline[index], dtype=np.float64)
                b = np.asarray(expert[index], dtype=np.float64) / temperature
                shape = a.shape
                if index == 2:
                    a, b = a.reshape(-1, 6, 3), b.reshape(-1, 6, 3)
                a = a - np.max(a, axis=-1, keepdims=True)
                b = b - np.max(b, axis=-1, keepdims=True)
                a = np.exp(a); a /= a.sum(axis=-1, keepdims=True)
                b = np.exp(b); b /= b.sum(axis=-1, keepdims=True)
                p = np.exp((1-weight)*np.log(np.maximum(a, 1e-12)) +
                           weight*np.log(np.maximum(b, 1e-12)))
                p /= p.sum(axis=-1, keepdims=True)
                fused.append(np.log(np.maximum(p, 1e-12)).reshape(shape).astype(np.float32))
            chosen = runtime.classify(_Outputs(fused + baseline[3:], info),
                                      labels, waveform, policy=policy)
        accepted, _ = runtime.safe_decision(chosen[0], chosen[1], chosen[3])
        if not accepted:
            return chosen
        index = labels["intents"].index(chosen[0])
        thresholds = (self.scope_policy["thresholds"] if original_ok
                      else self.scope_policy["rescue_thresholds"])
        if float(expert[4][0, index]) >= thresholds[index]:
            return chosen
        rejected_info = dict(chosen[3], rejected=True, slot=None,
                             scope_verifier_rejected=True)
        return "REJECT", float(chosen[2][-1]), chosen[2], rejected_info
