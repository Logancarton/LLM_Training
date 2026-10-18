"""Safety and learning-curve smoke tests; run with python -m unittest."""
import tempfile
import unittest
import torch
from sonara_story_learning import (Example, TemporalLearner, encode, load_examples,
                                   metrics, semantic_diagnostic, time_features,
                                   train_and_measure, vocabulary)


class StoryLearningTests(unittest.TestCase):
    def test_strict_no_answer_leakage(self):
        with tempfile.TemporaryDirectory() as folder:
            data = load_examples(folder, seed=0)
        self.assertEqual(sum(e.split == "train" for e in data), 100)
        self.assertEqual(sum(e.split != "train" for e in data), 60)
        self.assertTrue(all(t < e.query_epoch for e in data for t, _ in e.events))
        words = vocabulary([e for e in data if e.split == "train"])
        for term in ("maya", "paperclip", "bike"):
            self.assertNotIn(term, words)

    def test_counterfactual_time_pair(self):
        with tempfile.TemporaryDirectory() as folder:
            data = load_examples(folder, seed=0)
        timing = [e for e in data if e.split == "time"]
        for near, far in zip(timing[::2], timing[1::2]):
            self.assertEqual(near.events, far.events)
            self.assertNotEqual(near.label, far.label)
            self.assertNotEqual(time_features(near.events[-1][0], near.query_epoch),
                                time_features(far.events[-1][0], far.query_epoch))

    def test_no_time_model_cannot_distinguish_same_events(self):
        now = Example("near", "time", 20_000, 1, [(10_000, "Jake lost his ball.")])
        far = Example("far", "time", 300_000, 0, now.events)
        words = vocabulary([now])
        inputs_a = [x.unsqueeze(0) for x in encode(now, words)]
        inputs_b = [x.unsqueeze(0) for x in encode(far, words)]
        model = TemporalLearner(len(words), use_time=False)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(*inputs_a), model(*inputs_b)))

    def test_five_checkpoints_two_arms(self):
        with tempfile.TemporaryDirectory() as folder:
            data = load_examples(folder, seed=1)
        rows = train_and_measure(data, seed=1, steps=1)
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(0 <= row["overall"] <= 1 for row in rows))

    def test_oracle_control_explicitly_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            data = load_examples(folder, seed=1)
        rows = semantic_diagnostic(data, seed=1, steps=1)
        self.assertEqual(len(rows), 10)
        self.assertEqual({r["arm"] for r in rows}, {"oracle_time", "oracle_no_time"})


if __name__ == "__main__":
    unittest.main()
