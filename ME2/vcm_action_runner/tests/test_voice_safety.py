import unittest

import numpy as np

import vcm_infer as V
from vcm_action_runner.voice import Voice, describe


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

    def test_long_capture_centers_active_speech_in_model_window(self):
        audio = np.zeros(6 * V.SAMPLE_RATE, np.float32)
        speech_samples = int(1.9 * V.SAMPLE_RATE)
        t = np.arange(speech_samples, dtype=np.float32) / V.SAMPLE_RATE
        start = 2 * V.SAMPLE_RATE
        audio[start:start + speech_samples] = 0.02 * np.sin(2 * np.pi * 180 * t)

        clip, info = V.prepare_clip(audio)

        self.assertEqual(len(clip), V.N_SAMPLES)
        self.assertAlmostEqual(info["active_speech_seconds"], 1.9, places=2)
        self.assertAlmostEqual(info["centered_start_seconds"], 0.3, places=2)
        nonzero = np.flatnonzero(np.abs(clip) > 1e-5)
        midpoint = 0.5 * (nonzero[0] + nonzero[-1]) / V.SAMPLE_RATE
        self.assertAlmostEqual(midpoint, V.WINDOW_SEC / 2.0, places=2)
        self.assertAlmostEqual(float(np.max(np.abs(clip))), V.PEAK_TARGET,
                               places=4)

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

    def test_wake_reject_rescue_accepts_one_clear_conditional_command(self):
        info = {"span": (0.1, 1.2), "command_intent": "PAUSE",
                "command_conf": 0.14, "command_conditional_conf": 0.78,
                "command_conditional_margin": 0.61, "reject_conf": 0.82}
        self.assertEqual(V.wake_reject_rescue(info), "PAUSE")

    def test_wake_reject_rescue_rejects_ambiguous_or_extreme_reject(self):
        ambiguous = {"span": (0.1, 1.2), "command_intent": "CALL",
                     "command_conf": 0.12, "command_conditional_conf": 0.55,
                     "command_conditional_margin": 0.20, "reject_conf": 0.84}
        self.assertIsNone(V.wake_reject_rescue(ambiguous))
        extreme = dict(ambiguous, command_conditional_conf=0.90,
                       command_conditional_margin=0.80, reject_conf=0.96)
        self.assertIsNone(V.wake_reject_rescue(extreme))

    def test_voice_bridge_dispatches_calibrated_wake_reject_rescue(self):
        probabilities = np.full(len(V.INTENT_NAMES), 0.04 / 18.0, np.float32)
        probabilities[V.INTENT_NAMES.index("PAUSE")] = 0.14
        probabilities[V.REJECT_IDX_A] = 0.82
        intent_logits = np.log(probabilities)
        slot_logits = np.zeros(len(V.VALUE_TOKENS), np.float32)
        audio = np.zeros(V.SAMPLE_RATE, np.float32)
        t = np.arange(6000, dtype=np.float32) / V.SAMPLE_RATE
        audio[3000:9000] = 0.02 * np.sin(2 * np.pi * 180 * t)
        bridge = Voice()
        bridge._sess = _Session(intent_logits, slot_logits)
        bridge._labels = {"intents": V.INTENT_NAMES,
                          "value_tokens": V.VALUE_TOKENS}

        strict = bridge.classify_audio(audio)
        rescued = bridge.classify_audio(audio, allow_wake_short=True)

        self.assertFalse(strict["ok"])
        self.assertEqual(strict["rejection_reason"], "model_reject")
        self.assertTrue(rescued["ok"])
        self.assertEqual(rescued["intent"], "PAUSE")
        self.assertEqual(rescued["acceptance_policy"], "wake_reject_rescue")

    def test_legacy_e8_accepts_original_non_reject_top_prediction(self):
        probabilities = np.full(len(V.INTENT_NAMES), 0.15 / 17.0, np.float32)
        probabilities[V.INTENT_NAMES.index("WEATHER")] = 0.55
        probabilities[V.INTENT_NAMES.index("TIME")] = 0.25
        probabilities[V.REJECT_IDX_A] = 0.05
        bridge = Voice()
        bridge._sess = _Session(np.log(probabilities),
                                np.zeros(len(V.VALUE_TOKENS), np.float32))
        bridge._labels = {"intents": V.INTENT_NAMES,
                          "value_tokens": V.VALUE_TOKENS}
        audio = np.zeros(V.SAMPLE_RATE, np.float32)
        t = np.arange(6000, dtype=np.float32) / V.SAMPLE_RATE
        audio[3000:9000] = 0.02 * np.sin(2 * np.pi * 180 * t)

        safer = bridge.classify_audio(audio)
        legacy = bridge.classify_audio(audio, legacy_e8=True)

        self.assertFalse(safer["ok"])
        self.assertEqual(safer["rejection_reason"], "low_intent_confidence")
        self.assertTrue(legacy["ok"])
        self.assertEqual(legacy["intent"], "WEATHER")
        self.assertEqual(legacy["acceptance_policy"], "legacy_e8")
        self.assertIn("legacy E8 policy", describe(legacy))

    def test_every_rejection_reports_top_three_model_predictions(self):
        probabilities = np.full(len(V.INTENT_NAMES), 0.01 / 17.0, np.float32)
        probabilities[V.INTENT_NAMES.index("PAUSE")] = 0.10
        probabilities[V.INTENT_NAMES.index("CALL")] = 0.07
        probabilities[V.REJECT_IDX_A] = 0.82
        bridge = Voice()
        bridge._sess = _Session(np.log(probabilities),
                                np.zeros(len(V.VALUE_TOKENS), np.float32))
        bridge._labels = {"intents": V.INTENT_NAMES,
                          "value_tokens": V.VALUE_TOKENS}
        audio = np.zeros(V.SAMPLE_RATE, np.float32)
        t = np.arange(6000, dtype=np.float32) / V.SAMPLE_RATE
        audio[3000:9000] = 0.02 * np.sin(2 * np.pi * 180 * t)

        result = bridge.classify_audio(audio)
        description = describe(result)

        self.assertFalse(result["ok"])
        self.assertEqual([name for name, _ in result["top_predictions"]],
                         ["REJECT", "PAUSE", "CALL"])
        self.assertIn("top 3 model predictions: REJECT=0.82, PAUSE=0.10, CALL=0.07",
                      description)


if __name__ == "__main__":
    unittest.main()
