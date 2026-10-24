import json
import os
import sys
import tempfile
import unittest


HERE = os.path.dirname(os.path.abspath(__file__))
RUNNER = os.path.dirname(HERE)
if RUNNER not in sys.path:
    sys.path.insert(0, RUNNER)

from benchmark_log import BenchmarkLogger


class BenchmarkLoggerTests(unittest.TestCase):
    def _records(self, result):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "run.log")
            logger = BenchmarkLogger(path)
            logger.wake()
            logger.decision(result)
            logger.close()
            with open(path, encoding="utf-8") as handle:
                return [json.loads(line) for line in handle]

    def test_accepted_decision_has_required_fields(self):
        records = self._records({
            "ok": True, "intent": "TIME", "slot": None,
            "latency": {"classify_ms": 31.2345}, "audio_ms": 1400.0,
        })
        self.assertEqual(records[0], {"event": "wake", "wake": True})
        self.assertEqual(records[1], {
            "intent": "TIME", "slot": "", "infer_ms": 31.235,
            "audio_ms": 1400.0,
        })

    def test_rejection_is_explicit_out_of_scope(self):
        records = self._records({
            "ok": False, "intent": None, "slot": None,
            "latency": {"classify_ms": 20.0}, "audio_ms": 1800.0,
        })
        self.assertEqual(records[1]["intent"], "OUT_OF_SCOPE")
        self.assertEqual(records[1]["slot"], "")


if __name__ == "__main__":
    unittest.main()
