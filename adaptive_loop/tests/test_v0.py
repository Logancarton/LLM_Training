"""Tests for each v0 build step, in spec order. Run from the project folder:

    python3 -m unittest discover tests -v

They run fully offline (mock sensor and generator, hashing embedder).
"""
import json
import os
import sys
import tempfile
import time
import unittest
import warnings
from pathlib import Path

warnings.simplefilter("ignore", ResourceWarning)   # temp databases are left to the OS

os.environ["SONARA_SENSOR_PROVIDER"] = "mock"
os.environ["SONARA_GENERATOR_PROVIDER"] = "mock"
os.environ["SONARA_EMBEDDER"] = "hash"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import action as act          # noqa: E402
import brief as brief_mod      # noqa: E402
import config                 # noqa: E402
import db                     # noqa: E402
import llm                    # noqa: E402
import modulators as mod      # noqa: E402
import retrieval              # noqa: E402
from brain import Brain       # noqa: E402
from embed import HashEmbedder  # noqa: E402
from evidence import Evidence  # noqa: E402
from runtime import Control, Runtime  # noqa: E402
from sensor import heuristic  # noqa: E402


def fresh_brain(tmp=None):
    tmp = Path(tmp or tempfile.mkdtemp())
    return Brain(tmp / "t.db", HashEmbedder(), Evidence(tmp / "mem")), tmp


class Step1Schema(unittest.TestCase):
    def test_tables_and_general_location(self):
        brain, _ = fresh_brain()
        names = {r["name"] for r in brain.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertTrue(set(db.TABLES) <= names)
        self.assertEqual(db.llm_general_id(brain.conn), db.llm_general_id(brain.conn))


class Step2Sensor(unittest.TestCase):
    MESSAGES = ["How do I sort a list in Python?", "thanks, that works!", "No, that's wrong",
                "I'm so stressed and overwhelmed today", "Should I raise my prices?",
                "what about it?", "My patient's lithium dose needs adjusting", "ok",
                "Tell me about the hippocampus", "I meant the other file", "hi",
                "Can you explain DA again?", "This is great, love it", "why?",
                "urgent: payroll is due tomorrow", "I feel lost", "and then what?",
                "Actually, the meeting is Tuesday", "What is 2+2?", "That helped a lot"]

    def test_valid_shape_every_time(self):
        prev = None
        for text in self.MESSAGES:
            s = heuristic(text, prev)
            self.assertTrue(-1 <= s["affect"]["valence"] <= 1)
            for key in ("stakes", "ambiguity"):
                self.assertTrue(0 <= s[key] <= 1)
            self.assertIn(s["intent"], ("ask", "tell", "correct", "vent", "decide", "other"))
            json.dumps(s)
            prev = text

    def test_sensible_tags(self):
        self.assertTrue(heuristic("No, that's wrong", "x")["is_correction"])
        self.assertGreaterEqual(heuristic("urgent: payroll is due tomorrow")["stakes"], 0.7)
        self.assertGreaterEqual(heuristic("what about it?")["ambiguity"], 0.5)
        self.assertLess(heuristic("I'm so stressed and overwhelmed")["affect"]["valence"], 0)


class Step3Retrieval(unittest.TestCase):
    def test_reworded_matches_unrelated_is_novel(self):
        brain, _ = fresh_brain()
        topics = ["sort a list in python", "bake sourdough bread", "plant tomatoes in spring",
                  "fix a flat bike tire", "learn spanish verbs", "train a puppy to sit"]
        for i in range(30):
            brain.handle_turn(f"How do I {topics[i % len(topics)]} step {i}?")
        mods = mod.load(brain.conn)
        vec = brain.embedder.embed("what's the best way to sort a python list")
        r = retrieval.retrieve(brain.conn, "hash", vec, mods)
        self.assertFalse(r["is_novel"])
        self.assertIn("python", r["matched"][0]["text"])
        vec = brain.embedder.embed("quantum chromodynamics lecture notes")
        self.assertTrue(retrieval.retrieve(brain.conn, "hash", vec, mods)["is_novel"])


class Step4Modulators(unittest.TestCase):
    def test_scripted_sequences(self):
        base = dict(config.MOD_BASELINE)
        s = mod.update_turn(base, {"novelty": 1.0, "stakes": 0.1, "confidence": 0.65})
        self.assertGreater(s["ne"], base["ne"])
        self.assertLess(s["ht"], config.SAFETY_HT)
        s = mod.update_turn(base, {"novelty": 0.2, "stakes": 0.8, "confidence": 0.65})
        self.assertGreaterEqual(s["ht"], config.SAFETY_HT)
        s = mod.update_turn(base, {"is_correction": True, "confidence": 1.0})
        self.assertGreater(s["ach"], base["ach"] + 0.2)

    def test_returns_to_baseline_without_oscillation(self):
        s = {"ne": 0.9, "ht": 0.9, "ach": 0.8}
        history = [s]
        for _ in range(20):
            s = mod.update_turn(s, {"novelty": 0.0, "confidence": 1.0})
            history.append(s)
        for ch in mod.CHANNELS:
            self.assertAlmostEqual(s[ch], config.MOD_BASELINE[ch], delta=0.1)
            steps = [b[ch] - a[ch] for a, b in zip(history, history[1:])]
            self.assertTrue(all(st <= 1e-9 for st in steps), f"{ch} went back up")
            self.assertTrue(all(abs(st) <= config.MOD_MAX_STEP + 1e-9 for st in steps))


class Step5Action(unittest.TestCase):
    def test_choices(self):
        calm = dict(config.MOD_BASELINE)
        tense = {"ne": 0.3, "ht": 0.8, "ach": 0.2}
        self.assertEqual(act.select(0.65, 0.9, calm, False)[0], act.CLARIFY)
        self.assertEqual(act.select(0.3, 0.2, tense, False)[0], act.HEDGE)
        self.assertEqual(act.select(0.9, 0.1, calm, False)[0], act.ANSWER)
        self.assertNotIn(act.RETRIEVE_MORE, act.select(0.2, 0.1, tense, False)[1])


class Step6Brief(unittest.TestCase):
    def test_cap_and_corrections_kept(self):
        corr = [{"right_text": "Use 41810 N Venture Dr " * 3, "wrong_text": "x" * 70,
                 "confirmations": 2}] * 3
        hot = [{"text": "topic " * 20}] * 3
        text = brief_mod.build("HEDGE", {"ne": 0.3, "ht": 0.9, "ach": 0.2}, 0.9,
                               {"valence": -0.5, "valence_base": 0.1, "arousal": 0.8,
                                "arousal_base": 0.3}, corr, hot,
                               [{"kind": "llm_general", "summary": ""}])
        self.assertEqual(text.count("correction:"), 3)
        optional = [l for l in text.splitlines() if l.startswith(("user_now", "hot", "location"))]
        if optional:   # optional lines are only added while the brief fits the cap
            self.assertLessEqual(len(text), config.BRIEF_MAX_CHARS)

    def test_typical_brief_fits(self):
        brain, _ = fresh_brain()
        r = brain.handle_turn("How do I plant tomatoes in spring?")
        self.assertLessEqual(len(r["brief"]), config.BRIEF_MAX_CHARS)
        self.assertTrue(r["brief"].startswith("[STATE]"))


class Step7PostLoop(unittest.TestCase):
    def test_correction_holds_and_learning_happens(self):
        brain, _ = fresh_brain()
        brain.handle_turn("What day is the team meeting?")
        brain.handle_turn("No, that's wrong. The team meeting is on Tuesday.")
        held = 0
        for q in ["What day is the team meeting again?", "When is the team meeting?",
                  "Remind me what day the team meeting is"]:
            r = brain.handle_turn(q)
            held += "Tuesday" in r["brief"]
            brain.handle_turn("thanks, got it")
        self.assertEqual(held, 3)
        conf = brain.conn.execute("SELECT confirmations FROM corrections").fetchone()[0]
        self.assertGreater(conf, 0)

    def test_dopamine_moves_edges(self):
        brain, _ = fresh_brain()
        brain.handle_turn("How do I bake sourdough bread?")
        before = brain.conn.execute("SELECT MAX(weight) FROM edges").fetchone()[0]
        brain.handle_turn("thanks, that works great")
        after = brain.conn.execute("SELECT MAX(weight) FROM edges").fetchone()[0]
        self.assertGreater(after, before)

    def test_self_confirmation_guard(self):
        brain, _ = fresh_brain()
        r = brain.handle_turn("Tell me about the hippocampus")
        brain.handle_turn("No, I meant the hippocampal index theory")
        marker = "[mock reply"
        self.assertIn(marker, r["response"])
        for table, col in (("cues", "text"), ("corrections", "right_text"),
                           ("episodes", "summary"), ("locations", "summary"),
                           ("window_items", "text")):
            for row in brain.conn.execute(f"SELECT {col} FROM {table}"):
                self.assertNotIn(marker, row[0] or "", f"model output leaked into {table}")


class Regressions(unittest.TestCase):
    """Bugs found by tracing real conversations; each must stay fixed."""

    def test_thanks_is_not_ambiguous(self):
        self.assertLess(heuristic("thanks, got it", "previous")["ambiguity"], 0.5)

    def test_second_look_does_not_match_itself(self):
        brain, _ = fresh_brain()
        brain.handle_turn("urgent: payroll is due tomorrow")   # raises caution
        r = brain.handle_turn("How do I bake sourdough bread?")
        self.assertGreater(r["novelty"], 0.5)

    def test_correction_raises_confidence(self):
        brain, _ = fresh_brain()
        brain.handle_turn("What day is the team meeting?")
        brain.handle_turn("No, that's wrong. The team meeting is on Tuesday.")
        r = brain.handle_turn("When is the team meeting?")
        self.assertGreaterEqual(r["confidence"], config.CORRECTION_EDGE_WEIGHT - 0.1)
        self.assertEqual(r["action"], act.ANSWER)

    def test_neutral_turns_do_not_erode_confidence(self):
        brain, _ = fresh_brain()
        for t in ["How do I plant tomatoes?", "How do I fix a bike tire?",
                  "How do I train a puppy?", "How do I learn spanish verbs?"]:
            brain.handle_turn(t)
        weights = [r[0] for r in brain.conn.execute(
            "SELECT weight FROM edges WHERE uses > 0")]
        self.assertTrue(weights and all(abs(w - config.LLM_GENERAL_PRIOR) < 1e-6
                                        for w in weights))


class EvidenceBridge(unittest.TestCase):
    """The repo's sonara_memory.py store: user words in, quoted back with provenance."""

    def test_only_user_words_are_written(self):
        brain, _ = fresh_brain()
        texts = ["My dog Jake loves the pond behind the house.", "thanks, that works"]
        for t in texts:
            brain.handle_turn(t)
        stored = list(brain.evidence.store.memories.values())
        self.assertEqual([m.content for m in stored], texts)
        self.assertTrue(all(m.source == "conversation" for m in stored))
        self.assertFalse(any("[mock reply" in m.content for m in stored))

    def test_recalled_across_sessions_without_self_match(self):
        brain, tmp = fresh_brain()
        first = brain.handle_turn("My dog Jake loves the pond behind the house.")
        self.assertEqual(first["evidence"], [])
        later = Brain(tmp / "t2.db", HashEmbedder(), Evidence(tmp / "mem"))   # new session
        r = later.handle_turn("Where does Jake like to go?")
        self.assertIn("evidence: user said", r["brief"])
        self.assertIn("pond", r["brief"])

    def test_loop_runs_without_the_store(self):
        tmp = Path(tempfile.mkdtemp())
        brain = Brain(tmp / "t.db", HashEmbedder(), Evidence(enabled=False))
        self.assertIsNone(brain.handle_turn("hello there")["evidence_id"])


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.brain, tmp = fresh_brain()
        self.control = Control(tmp / "control.json")
        self.rt = Runtime(self.brain, self.control)

    def test_off_blocks_turns_and_flag_survives(self):
        self.assertTrue(self.control.is_on())
        self.rt.toggle()
        self.assertFalse(Control(self.control.path).is_on())   # a fresh read: survives restart
        self.assertIn("error", self.rt.chat("hello?"))
        self.assertIsNone(self.rt.heartbeat_once())
        self.rt.toggle()
        self.assertNotIn("error", self.rt.chat("hello?"))

    def test_resume_applies_full_gap_once(self):
        now = time.time()
        mod.save(self.brain.conn, {"ne": 0.9, "ht": 0.3, "ach": 0.2}, now)
        db.set_meta(self.brain.conn, "last_decay", now - 3600)   # an hour switched off
        minutes = self.brain.tick(now)
        self.assertAlmostEqual(minutes, 60, delta=0.01)
        expected = 0.3 + 0.6 * 0.5 ** (60 / config.MOD_HALF_LIFE_MINUTES["ne"])
        self.assertAlmostEqual(mod.load(self.brain.conn)["ne"], expected, places=5)
        self.assertEqual(self.brain.tick(now), 0.0)   # applied once, not twice

    def test_no_llm_calls_in_quiet_wake(self):
        before = llm.CALL_COUNT
        start = time.time()
        for i in range(1, 30):
            self.rt.heartbeat_once(start + i * 60)
        self.assertEqual(llm.CALL_COUNT, before)

    def test_off_mid_turn_leaves_db_consistent(self):
        import threading
        t = threading.Thread(target=self.rt.chat, args=("How do I train a puppy?",))
        t.start()
        self.rt.toggle()   # waits for the turn's write to finish
        t.join()
        turns = self.brain.conn.execute("SELECT COUNT(*) FROM turns").fetchone()[0]
        logged = self.brain.conn.execute("SELECT COUNT(*) FROM modulators").fetchone()[0]
        self.assertEqual(turns, logged)
        self.assertFalse(self.control.is_on())

    def test_states(self):
        self.assertEqual(self.rt.state(), "quiet")
        self.rt.chat("hello there")
        self.assertEqual(self.rt.state(), "awake")
        self.assertEqual(self.rt.state(now=time.time() + 30 * 60), "quiet")
        self.assertEqual(self.rt.state(now=time.time() + 3 * 3600), "sleep")
        self.rt.toggle()
        self.assertEqual(self.rt.state(), "off")


if __name__ == "__main__":
    unittest.main()
