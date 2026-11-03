"""Research-only waveform classifier for the fitted clean development policy."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

import vcm_infer as runtime
from numeral_runtime import NumeralResearchSession
from vcm_long_window import LongWindowSession
from vcm_selective_scope import _Outputs

HERE = Path(__file__).resolve().parent


def _warp(x, factor):
    a = torch.from_numpy(np.asarray(x))
    if factor >= 200:
        shift = factor - 200 if factor < 210 else -(factor - 210)
        y = a.clone()
        v = a[:, :1]
        n = v.shape[-2]
        v = F.pad(v, (0, 0, max(0, shift), max(0, -shift)))
        y[:, :1] = v[:, :, :n] if shift >= 0 else v[:, :, -n:]
        return y.numpy()
    n = a.shape[-1]
    length = int(round(n * factor / 100))
    y = F.interpolate(a, size=(a.shape[-2], length), mode='bilinear', align_corners=False)
    y = F.pad(y, ((n-length)//2, n-length-(n-length)//2)) if length < n else y[..., (length-n)//2:(length-n)//2+n]
    if y.shape[1] > 1:
        y[:, 1:] = (y[:, 1:] >= .5).float()
    return y.numpy()


def _softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    p = np.exp(x)
    return p / p.sum(axis=-1, keepdims=True)


class FittedPolicy:
    def __init__(self):
        bundle = HERE / 'vcm_benchmark_joint_recovery'
        self.labels = runtime.load_labels(str(bundle / 'labels.json'))
        self.policy = runtime.load_policy(str(bundle / 'policy.json'))
        self.session = NumeralResearchSession(bundle / 'ensemble.json')
        self.stages = json.loads((HERE / 'eighteen_extended_robust_development.json').read_text())['stages']

    def classify(self, waveform):
        """Return the usual (intent, confidence, probabilities, info) tuple."""
        session, labels, policy = self.session, self.labels, self.policy
        wave = np.asarray(waveform, dtype=np.float32)
        intents, values = labels['intents'], labels['value_tokens']
        def numeric(result):
            ok, _ = runtime.safe_decision(result[0], result[1], result[3])
            if not ok:
                return 19, 18
            token = result[3].get('slot_token')
            return intents.index(result[0]), values.index(token) if token in values else 18
        result = runtime.classify(session, labels, wave, policy=policy)
        oa, _ = numeric(result)
        feed, info = session.prepare_feed(wave)
        if info.get('span') is None:
            return result
        original = LongWindowSession.run(session, None, feed)
        oldp = [_softmax(x.reshape(-1, 6, 3)).reshape(x.shape) if j == 2 else _softmax(x) for j, x in enumerate(original[:3])]
        expert = session.expert.run(None, {'mel': feed['long']})
        extra = session.extra.run(None, {'mel': feed['long']})
        current = result
        current_a = oa
        cache = {}
        for stage in self.stages:
            choice = stage['choices'][current_a]
            if choice['kind'] == 'none' or choice['threshold'] >= 1e8:
                continue
            key = (choice['factor'], choice['weight'])
            if key not in cache:
                warped = dict(feed, long=_warp(feed['long'], key[0]), short=_warp(feed['short'], key[0]))
                altered = LongWindowSession.run(session, None, warped)
                newp = [_softmax(x.reshape(-1, 6, 3)).reshape(x.shape) if j == 2 else _softmax(x) for j, x in enumerate(altered[:3])]
                bp = [(1-key[1])*x + key[1]*y for x, y in zip(oldp, newp)]
                mixed = [np.log(np.maximum(x, 1e-12)).astype(np.float32) for x in bp] + original[3:]
                base = runtime.classify(_Outputs(mixed, info), labels, wave, policy=policy)
                prior = session._stage(labels, wave, info, policy, mixed, base, expert, session.scope_policy)
                proposal = session._stage(labels, wave, info, policy, mixed, prior, extra, session.extra_policy, float(session.extra_policy.get('scope_mix', 0.)), expert[4])
                pa, _ = numeric(prior)
                ca, _ = numeric(proposal)
                idx = pa if pa < 19 else ca
                if idx < 19:
                    x = np.asarray(wave, dtype=np.float64)
                    x = np.pad(x, (0, (-len(x)) % 320))
                    rms = np.sqrt(np.mean(x.reshape(-1, 320)**2, axis=1)+1e-12)
                    low, high = np.percentile(rms, [20, 95])
                    if 20*np.log10(high/low) < float(session.extra_policy['quality_thresholds_db'][idx]):
                        proposal = prior
                        ca = pa
                gain = float(bp[0][0, ca] - oldp[0][0, ca])
                margin = float(np.log(max(float(bp[0][0, ca]), 1e-10)) - np.log(max(float(bp[0][0, oa]), 1e-10)))
                cache[key] = proposal, ca, gain, margin
            proposal, ca, gain, margin = cache[key]
            if (margin if choice['kind'] == 'margin' else gain) >= choice['threshold']:
                current, current_a = proposal, ca
        return current
