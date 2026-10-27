import json
from pathlib import Path
import tempfile
import unittest

from sonara_memory import Episode, LinkLearner, Memory, ParagraphEncoder
from train import build_examples, association_metrics, benchmark


class MemoryBehaviorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        samples, _ = build_examples(n=40)
        cls.learner = LinkLearner()
        cls.learner.fit(samples, epochs=60)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "memory.db"
        self.memory = Memory(self.db, self.learner)

    def tearDown(self):
        self.memory.close()
        self.temp.cleanup()

    def test_paragraph_to_time_order_and_novelty(self):
        entries = self.memory.ingest(
            "On Monday morning, Sarah put her keys on the kitchen table. "
            "Later that afternoon, she moved them into her bedroom drawer. "
            "Usually, Sarah keeps her keys by the front door.")
        self.assertEqual([e.kind for e in entries], ["location", "location", "habit"])
        self.assertEqual(entries[0].identity, entries[1].identity)
        self.assertTrue(entries[1].novelty)
        self.assertEqual(self.memory.recall("Where are Sarah's keys now?")["answer"], "bedroom drawer")
        self.assertEqual(self.memory.recall("Where were Sarah's keys before?")["answer"], "kitchen table")
        self.assertIn("front door", self.memory.recall("What is unusual about Sarah's keys?")["answer"])
        self.assertGreater(self.memory.counts()["links"], 0)

    def test_no_invented_causation(self):
        self.memory.ingest("At noon, Sarah moved the keys to the drawer. Sarah has been forgetting things.")
        result = self.memory.recall("Why did Sarah's keys?")
        self.assertIsNone(result["answer"])
        self.assertEqual(self.memory.counts()["events"], 2)  # source note preserved
        self.assertEqual(self.memory.conn.execute("SELECT kind FROM events WHERE id=2").fetchone()[0], "note")

    def test_isolated_identical_objects_and_bounded_retrieval(self):
        for i in range(50):
            self.memory.ingest(f"At dawn, Maya moved the notebook to the shelf.")
        self.memory.ingest("At dawn, Sarah moved the notebook to the desk.")
        result = self.memory.recall("Where are Sarah's notebook now?")
        self.assertEqual(result["answer"], "desk")
        self.assertLessEqual(len(result["activated"]), 3)
        self.assertLessEqual(result["examined"], 4)
        self.assertEqual(self.memory.counts()["events"], 51)

    def test_restart_persists_episode_edges_and_trained_parameters(self):
        self.memory.ingest("At dawn, Maya moved the compass to the cellar. "
                           "At noon, Maya moved the compass to the attic.")
        start_count = self.memory.counts()
        model = Path(self.temp.name) / "model.json"
        self.learner.save(model)
        self.memory.close()
        self.memory = Memory(self.db, LinkLearner.load(model))
        self.assertEqual(self.memory.counts(), start_count)
        self.assertEqual(self.memory.recall("Where were Maya's compass before?")["answer"], "cellar")

    def test_ambiguous_pronoun_is_not_guessed(self):
        entries = ParagraphEncoder().encode("She moved them into the drawer.")
        self.assertEqual(entries[0].kind, "note")
        self.assertFalse(entries[0].identity)

    def test_unseen_identity_relation_learning(self):
        heldout, stories = build_examples(seed=99, n=30, test=True)
        baseline = association_metrics(LinkLearner(), heldout)
        trained = association_metrics(self.learner, heldout)
        self.assertGreater(trained["accuracy"], baseline["accuracy"])
        self.assertGreater(trained["accuracy"], 0.90)
        results = benchmark(self.learner, stories, n=30)
        self.assertEqual(results["answer_accuracy"], 1.0)
        self.assertLess(results["mean_activated"], 3.0)


if __name__ == "__main__":
    unittest.main()
