"""End-to-end sparse cue -> learned context -> detailed reinstatement tests."""
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

from sonara_memory import LinkLearner, Memory
from train import build_context_examples, build_examples


class HippocampalInspiredRecallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        identity_samples, _ = build_examples(n=50)
        cls.model = LinkLearner()
        cls.model.fit(identity_samples, epochs=60)
        cls.model.fit_context(build_context_examples(n=120), epochs=75)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = Path(self.temp.name) / "events.db"
        self.memory = Memory(self.db, self.model)

    def tearDown(self):
        self.memory.close()
        self.temp.cleanup()

    def _connected_story(self):
        self.memory.ingest("At dawn, Sarah moved the keys to the shelf.")
        self.memory.ingest("At noon, Sarah placed the map on the drawer.")
        self.memory.ingest("At dusk, Ruby moved the notebook to the drawer.")

    def test_two_hop_existing_links_reinstate_detailed_evidence(self):
        self._connected_story()
        rows = self.memory.conn.execute(
            "SELECT source,target,relation FROM links ORDER BY source,target").fetchall()
        edges = [(r["source"], r["target"], r["relation"]) for r in rows]
        self.assertTrue(any(a == 2 and b == 1 and rel == "shared-actor"
                            for a, b, rel in edges), edges)
        self.assertTrue(any(a == 3 and b == 2 and rel == "shared-place"
                            for a, b, rel in edges), edges)
        self.assertFalse(any({a, b} == {1, 3} for a, b, _ in edges), edges)

        traced = self.memory.reactivate("Sarah", "keys", limit=10, hops=2)
        self.assertEqual(traced["seed"]["id"], 1)
        recovered = next((x for x in traced["related"] if x["id"] == 3), None)
        self.assertIsNotNone(recovered)
        self.assertEqual(recovered["path"], [1, 2, 3])
        self.assertEqual(recovered["hops"], 2)
        self.assertEqual(recovered["source"],
                         "At dusk, Ruby moved the notebook to the drawer.")
        self.assertEqual(len(recovered["supporting_links"]), 2)
        self.assertTrue(traced["phases"]["detail_cue"])
        contexts = traced["phases"]["broader_context"]
        self.assertTrue(any(x["kind"] == "actor" and x["value"] == "Sarah"
                            for x in contexts))
        self.assertTrue(any(x["kind"] == "location" and x["value"] == "drawer"
                            for x in contexts))
        self.assertTrue(traced["phases"]["detail_reinstatement"])

        question = "How are Sarah's keys connected to Ruby's notebook?"
        self.assertIsNotNone(self.memory.recall(question)["answer"])
        self.assertEqual(self.memory.recall(question)["activated"][0]["path"],
                         [1, 2, 3])

    def test_edge_ablation_removes_associative_answer_but_preserves_facts(self):
        self._connected_story()
        question = "How are Sarah's keys connected to Ruby's notebook?"
        self.assertIsNotNone(self.memory.recall(question)["answer"])
        with self.memory.conn:
            self.memory.conn.execute("DELETE FROM links WHERE source=3 AND target=2")
        after = self.memory.recall(question)
        self.assertIsNone(after["answer"])
        self.assertEqual(after["reason"], "No stored associative path")
        self.assertEqual(self.memory.recall("Where are Sarah's keys now?")["answer"],
                         "shelf")

    def test_sparse_budgets_and_no_cross_identity_factual_merging(self):
        self._connected_story()
        for _ in range(80):
            self.memory.ingest("At noon, Noah moved the compass to the hallway.")
        recall = self.memory.reactivate("Sarah", "keys", hops=2, beam=4, limit=3)
        self.assertLessEqual(recall["examined"], 4 + 4 * 4)
        self.assertLessEqual(len(recall["related"]), 3)
        self.assertEqual(self.memory.recall("Where are Ruby's notebook now?")["answer"],
                         "drawer")
        self.assertEqual(self.memory.recall("Where are Sarah's keys now?")["answer"],
                         "shelf")

    def test_persistent_index_and_no_synthetic_hierarchy_memories(self):
        self._connected_story()
        before = self.memory.reactivate("Sarah", "keys", limit=10)
        self.memory.close()
        self.memory = Memory(self.db, self.model)
        self.assertEqual(self.memory.reactivate("Sarah", "keys", limit=10), before)
        tables = {r[0] for r in self.memory.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertNotIn("memory_nodes", tables)
        self.assertEqual(self.memory.counts()["events"], 3)

    def test_live_cli_enters_same_sparse_memory_path(self):
        model_path = Path(self.temp.name) / "trained.json"
        self.model.save(model_path)
        args = [sys.executable, "sonara.py", "--db", str(self.db),
                "--model", str(model_path)]

        def invoke(*parameters):
            return json.loads(subprocess.run(
                args + list(parameters), check=True, capture_output=True,
                text=True, cwd=Path(__file__).resolve().parents[1]).stdout)

        invoke("store", "At dawn, Sarah moved the keys to the shelf.")
        invoke("store", "At noon, Sarah placed the map on the drawer.")
        invoke("store", "At dusk, Ruby moved the notebook to the drawer.")
        trace = invoke("trace", "Sarah", "keys", "--hops", "2", "--limit", "10")
        target = next(r for r in trace["related"] if r["identity"] == "ruby::notebook")
        self.assertEqual(target["path"], [1, 2, 3])
        self.assertTrue(any(c["kind"] == "actor" and c["value"] == "Sarah"
                            for c in trace["phases"]["broader_context"]))
        self.assertEqual(invoke("ask", "How are Sarah's keys connected to Ruby's notebook?")[
            "activated"][0]["path"], [1, 2, 3])

    def test_disconnected_episode_and_unsupported_cause_abstain(self):
        self.memory.ingest("At dawn, Sarah moved the keys to the shelf.")
        self.memory.ingest("At noon, Noah moved the compass to the drawer.")
        answer = self.memory.recall(
            "How are Sarah's keys connected to Noah's compass?")
        self.assertIsNone(answer["answer"])
        self.assertIsNone(self.memory.recall("Why did Sarah's keys?")["answer"])


if __name__ == "__main__":
    unittest.main()
