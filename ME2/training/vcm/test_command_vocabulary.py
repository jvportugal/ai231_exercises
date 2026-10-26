import unittest

import config as C
from train_temporal_attention import make_vocabulary, word_targets


class CommandVocabularyTests(unittest.TestCase):
    def test_observed_contrasts_survive_frequency_cutoff(self):
        texts = ["room bedroom"] * 20 + ["turn lights on", "turn lights off"]
        rows = [(None, 0, 0, True)] * len(texts)
        vocabulary = make_vocabulary(texts, rows, 2)
        self.assertEqual(set(vocabulary), {"on", "off"})
        targets = word_targets(texts[-2:], vocabulary)
        self.assertEqual(targets[0, vocabulary.index("on")], 1)
        self.assertEqual(targets[0, vocabulary.index("off")], 0)

    def test_only_real_training_commands_define_vocabulary(self):
        texts = ["on", "off", "pause", "temperature"]
        rows = [(None, 0, 0, True), (None, 0, 0, False),
                (None, C.N_INTENT, 0, True), (None, 0, 0, True)]
        self.assertEqual(set(make_vocabulary(texts, rows, 96)), {"on", "temperature"})


if __name__ == "__main__":
    unittest.main()
