import unittest

import numpy as np

import vcm_infer as V


class _Session:
    def __init__(self, intent_logits, slot_logits):
        self.outputs = (np.asarray([intent_logits], np.float32),
                        np.asarray([slot_logits], np.float32))

    def run(self, _outputs, _inputs):
        return self.outputs


class VoiceSafetyTests(unittest.TestCase):
    def test_isolated_impulse_is_not_speech(self):
        audio = np.zeros(V.SAMPLE_RATE, np.float32)
        audio[4000:4010] = 0.9
        self.assertIsNone(V._detect_speech(audio))

    def test_quiet_sustained_voice_like_signal_is_detected(self):
        audio = np.zeros(V.SAMPLE_RATE, np.float32)
        t = np.arange(4800, dtype=np.float32) / V.SAMPLE_RATE
        audio[4000:8800] = 0.008 * np.sin(2 * np.pi * 180 * t)
        self.assertIsNotNone(V._detect_speech(audio))

    def test_slot_decoder_only_compares_values_valid_for_intent(self):
        intent_logits = np.full(len(V.INTENT_NAMES), -8.0, np.float32)
        intent_logits[V.INTENT_NAMES.index("COLOR")] = 8.0
        slot_logits = np.full(len(V.VALUE_TOKENS), -8.0, np.float32)
        slot_logits[V.VALUE_TOKENS.index("TIMER_10_SECONDS")] = 7.0
        slot_logits[V.VALUE_TOKENS.index("COLOR_RED")] = 6.0
        slot_logits[V.VALUE_TOKENS.index("COLOR_BLUE")] = 1.0
        slot_logits[V.VALUE_TOKENS.index("COLOR_GREEN")] = 0.0
        audio = np.zeros(V.SAMPLE_RATE, np.float32)
        audio[2000:10000] = 0.02 * np.sin(
            2 * np.pi * 180 * np.arange(8000) / V.SAMPLE_RATE)
        label, _, _, info = V.classify(
            _Session(intent_logits, slot_logits),
            {"intents": V.INTENT_NAMES, "value_tokens": V.VALUE_TOKENS}, audio)
        self.assertEqual(label, "COLOR")
        self.assertEqual(info["slot"], "Red")

    def test_safe_policy_rejects_ambiguous_slot(self):
        info = {"span": (0.1, 0.7), "rejected": False,
                "intent_margin": 0.8, "slot": "Red",
                "slot_conf": 0.8, "slot_margin": 0.03}
        self.assertEqual(V.safe_decision("COLOR", 0.9, info),
                         (False, "low_slot_margin"))

    def test_safe_policy_accepts_clear_fixed_command(self):
        info = {"span": (0.1, 0.7), "rejected": False,
                "intent_margin": 0.8, "slot": None}
        self.assertEqual(V.safe_decision("WEATHER", 0.9, info), (True, None))

    def test_wake_short_policy_accepts_clear_short_fixed_command(self):
        info = {"span": (0.1, 0.9), "rejected": False,
                "intent_margin": 0.40, "slot": None}
        self.assertEqual(V.safe_decision("PAUSE", 0.72, info),
                         (False, "low_intent_confidence"))
        self.assertEqual(
            V.safe_decision("PAUSE", 0.72, info, allow_wake_short=True),
            (True, None))

    def test_wake_short_policy_does_not_relax_long_or_slotted_commands(self):
        long_info = {"span": (0.1, 1.8), "rejected": False,
                     "intent_margin": 0.40, "slot": None}
        self.assertFalse(V.safe_decision(
            "CALL", 0.72, long_info, allow_wake_short=True)[0])
        slot_info = {"span": (0.1, 0.9), "rejected": False,
                     "intent_margin": 0.40, "slot": "Red",
                     "slot_conf": 0.9, "slot_margin": 0.8}
        self.assertFalse(V.safe_decision(
            "COLOR", 0.72, slot_info, allow_wake_short=True)[0])

    def test_wake_short_policy_excludes_poorly_calibrated_fixed_intent(self):
        info = {"span": (0.1, 0.9), "rejected": False,
                "intent_margin": 0.40, "slot": None}
        self.assertFalse(V.safe_decision(
            "LIGHT_ON", 0.72, info, allow_wake_short=True)[0])

    def test_wake_short_policy_still_rejects_ambiguous_audio(self):
        info = {"span": (0.1, 0.8), "rejected": False,
                "intent_margin": 0.12, "slot": None}
        self.assertFalse(V.safe_decision(
            "CALL", 0.70, info, allow_wake_short=True)[0])


if __name__ == "__main__":
    unittest.main()
