"""Verify veto and rescue decisions through the public runtime entry point."""
import json
import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import vcm_infer as runtime
from vcm_selective_scope import SelectiveScopeSession
from vcm_long_window import LongWindowSession


class FixedExpert:
    def __init__(self, values): self.values = values
    def run(self, names, feed): return self.values


class SelectiveScopeTests(unittest.TestCase):
    def setUp(self):
        self.labels = runtime.load_labels(str(ROOT / 'vcm_benchmark_joint_recovery' / 'labels.json'))
        self.policy = runtime.load_policy(str(ROOT / 'vcm_benchmark_joint_recovery' / 'policy.json'))
        manifest = json.loads((ROOT / 'vcm_benchmark_joint_recovery' / 'ensemble.json').read_text())
        self.session = object.__new__(SelectiveScopeSession)
        self.session.scope_policy = manifest['scope_policy']
        self.session.prepare_feed = lambda x, verbose=False: ({'long': np.zeros((1,2,40,500),np.float32)}, {'span': (0.,1.)})
        self.previous = LongWindowSession.run
        self.addCleanup(setattr, LongWindowSession, 'run', self.previous)

    def run_case(self, baseline_intent, expert_intent, scope):
        def logits(n, ix, high=8.):
            a=np.full((1,n),-8.,np.float32);a[0,ix]=high;return a
        base=[logits(20,baseline_intent),logits(19,18),np.zeros((1,18),np.float32),np.zeros((1,96),np.float32)]
        expert=[logits(20,expert_intent,20.),logits(19,18),np.zeros((1,18),np.float32),np.zeros((1,96),np.float32),np.full((1,19),scope,np.float32)]
        self.session.expert=FixedExpert(expert)
        LongWindowSession.run=lambda *args: base
        return runtime.classify(self.session,self.labels,np.zeros(16000,np.float32),policy=self.policy)

    def test_verifier_veto(self):
        intent,_,_,info=self.run_case(0,0,-1e3)
        self.assertEqual(intent,'REJECT')
        self.assertTrue(info['scope_verifier_rejected'])

    def test_rescue_requires_support(self):
        accepted=self.run_case(19,0,1e3)
        self.assertEqual(accepted[0],self.labels['intents'][0])
        rejected=self.run_case(19,0,-1e3)
        self.assertEqual(rejected[0],'REJECT')


if __name__ == '__main__': unittest.main()
