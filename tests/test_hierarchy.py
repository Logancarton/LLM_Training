"""End-to-end proof of memory hierarchy and genuinely link-dependent recall."""
from pathlib import Path
import tempfile
import unittest

from sonara_memory import LinkLearner, Memory
from train import build_context_examples, build_examples


class HierarchicalRecallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        identity, _ = build_examples(n=50)
        cls.learner = LinkLearner()
        cls.learner.fit(identity, epochs=50)
        cls.learner.fit_context(build_context_examples(n=100), epochs=60)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "memory.db"
        self.memory = Memory(self.db, self.learner)

    def tearDown(self):
        self.memory.close()
        self.temp.cleanup()

    def test_associative_answer_requires_learned_edges_ablation(self):
        self.memory.ingest("At dawn, Sarah moved the keys to the shelf.")
        self.memory.ingest("At noon, Sarah placed the map on the shelf.")
        query = "What is associated with Sarah's keys?"
        result = self.memory.recall(query)
        self.assertIn("Sarah's map", result["answer"])
        self.assertEqual(result["association_relation"], "shared-actor")
        self.assertTrue(any(level["level"] == 2 for level in result["activated_levels"]))
        self.assertTrue(any(level["level"] == 1 for level in result["activated_levels"]))
        self.assertTrue(any(level["level"] == 0 for level in result["activated_levels"]))
        self.assertLessEqual(len(result["activated"]), 3)
        # Ablation: unchanged evidence, identity, parser, and database lookup.
        with self.memory.conn:
            self.memory.conn.execute("DELETE FROM links")
        ablated = self.memory.recall(query)
        self.assertIsNone(ablated["answer"])
        self.assertEqual(ablated["reason"], "No learned contextual link")
        # Factual location queries are intentionally link-independent.
        self.assertEqual(self.memory.recall("Where are Sarah's keys now?")["answer"], "shelf")

    def test_levels_have_evidence_counts_and_restart_survival(self):
        self.memory.ingest("At dawn, Sarah moved the keys to the shelf.")
        self.memory.ingest("At noon, Sarah placed the keys on the shelf.")
        self.memory.ingest("At dusk, Sarah moved the keys to the drawer.")
        hierarchy = self.memory.hierarchy("Sarah", "keys")
        self.assertEqual(hierarchy["levels"][0]["level"], 2)
        self.assertEqual(hierarchy["levels"][0]["evidence_count"], 3)
        shelf = next(x for x in hierarchy["levels"] if x["label"] == "shelf")
        self.assertEqual(shelf["level"], 1)
        self.assertEqual(shelf["evidence_count"], 2)
        self.memory.close()
        self.memory = Memory(self.db, self.learner)
        self.assertEqual(self.memory.hierarchy("Sarah", "keys"), hierarchy)
        self.assertEqual(self.memory.recall("Where are Sarah's keys now?")["answer"], "drawer")

    def test_pattern_guided_seed_selection_not_just_last_three_events(self):
        for _ in range(4):
            self.memory.ingest("At dawn, Sarah moved the keys to the shelf.")
        self.memory.ingest("At noon, Sarah moved the keys to the drawer.")
        self.memory.ingest("At dusk, Sarah moved the keys to the counter.")
        related = self.memory.associate("Sarah", "keys", limit=3)
        # Event 4 is no longer among the 3 most recent events, but the
        # repeated shelf motif elects it as one of the bounded seeds.
        seeded = [n["id"] for n in related["activated_levels"]
                  if n["kind"] == "episode"]
        self.assertIn(4, seeded)
        self.assertIn(6, seeded)
        self.assertLessEqual(len(seeded), 3)
        self.assertLessEqual(len(related["related"]), 3)

    def test_abstains_on_unlinked_context_and_migrates_old_hierarchy(self):
        self.memory.ingest("At dawn, Sarah moved the keys to the shelf.")
        self.memory.ingest("At noon, Noah moved the keys to the drawer.")
        self.assertIsNone(self.memory.recall(
            "What is connected to Sarah's keys?")["answer"])
        self.memory.close()
        import sqlite3
        with sqlite3.connect(self.db) as conn:
            conn.execute("DROP TABLE memory_nodes")
        self.memory = Memory(self.db, self.learner)
        self.assertEqual(self.memory.hierarchy("Sarah", "keys")["levels"][0]["evidence_count"], 1)
        self.assertEqual(self.memory.hierarchy("Noah", "keys")["levels"][0]["evidence_count"], 1)


if __name__ == "__main__":
    unittest.main()
