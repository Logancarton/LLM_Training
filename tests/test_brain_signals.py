"""Behavioral tests for fifty trained connection cues, not a neural simulation."""
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from association_signals import brain_signals
from sonara_memory import Episode, LinkLearner, Memory
from train import association_metrics, build_context_examples


class BrainSignalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = LinkLearner()
        cls.model.fit_context(build_context_examples(n=80), epochs=60)

    def test_exactly_fifty_signals_all_have_live_evidence_cases(self):
        episodes = [
            Episode(actor='Sarah', item='keys', action='moved', location='garden shelf',
                    stated_time='Monday morning at 8 AM', kind='location',
                    source='Maybe Sarah moved the keys from garden shelf.', novelty=True),
            Episode(actor='Sarah', item='keys', action='moved', location='garden shelf',
                    stated_time='Monday morning at 8 AM', kind='location',
                    source='Maybe Sarah moved the keys from garden shelf.', novelty=True),
            Episode(actor='Sarah', item='keys', action='placed', location='garden bench',
                    stated_time='Tuesday evening at 9 PM', kind='location',
                    source='Sarah placed keys elsewhere.'),
            Episode(actor='Noah', item='keys', action='usually', location='office shelf',
                    stated_time='Monday morning at 8 AM', kind='habit',
                    source='Noah did not move a key.'),
            Episode(actor='Sarah', item='gloves', action='usually', location='garden shelf',
                    kind='habit', source='Perhaps Sarah usually keeps gloves.'),
            Episode(actor='Ruby', item='map', kind='note', source='Unknown world language.'),
            Episode(actor='Sarah', item='keys', action='moved', location='garden shelf',
                    stated_time='Wednesday afternoon 11 AM', kind='location',
                    source='She moved those keys.'),
            Episode(actor='Zara', item='camera', action='moved', location='garden drawer',
                    kind='location', source='Zara moved an object.'),
            Episode(actor='Sarah', item='keys', action='usually', location='office shelf',
                    kind='habit', source='Usually Sarah keeps the keys at the office shelf.'),
        ]
        names = set(brain_signals(episodes[0], episodes[1]))
        self.assertEqual(len(names), 50)
        covered = set()
        for a in episodes:
            for b in episodes:
                if a is b:
                    continue
                actual = brain_signals(a, b)
                self.assertEqual(set(actual), names)
                self.assertTrue(all(isinstance(x, float) and 0 <= x <= 1 for x in actual.values()))
                covered.update(k for k, value in actual.items() if value > 0)
        self.assertEqual(covered, names, f'Unreachable signals: {names - covered}')

    def test_context_learning_with_unseen_people_objects_places(self):
        samples = build_context_examples(seed=91, n=40, test=True)
        untrained = association_metrics(LinkLearner(), samples, context=True)
        trained = association_metrics(self.model, samples, context=True)
        self.assertGreater(trained['accuracy'], untrained['accuracy'])
        self.assertGreater(trained['accuracy'], 0.9)

    def test_context_assoc_is_not_factual_identity_merging(self):
        with tempfile.TemporaryDirectory() as folder:
            with Memory(Path(folder) / 'memory.db', self.model) as memory:
                memory.ingest('At dawn, Sarah moved the keys to the shelf.')
                memory.ingest('At noon, Noah moved the keys to the drawer.')
                self.assertEqual(memory.recall("Where are Sarah's keys now?")['answer'], 'shelf')
                self.assertEqual(memory.recall("Where are Noah's keys now?")['answer'], 'drawer')
                self.assertEqual(memory.associate('Noah', 'keys')['related'], [])
                # Shared place is contextual association, not same-object identity.
                memory.ingest('At dusk, Ruby placed the map on the drawer.')
                related = memory.associate('Ruby', 'map')['related']
                self.assertTrue(any(x['relation'] == 'shared-place' for x in related))
                self.assertTrue(all('association_score' in x and x['evidence_cues'] for x in related))
                self.assertEqual(memory.recall("Where are Noah's keys now?")['answer'], 'drawer')

    def test_sparse_growth_stays_linear_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            db = Path(folder) / 'memory.db'
            model_path = Path(folder) / 'model.json'
            self.model.save(model_path)
            with Memory(db, self.model) as memory:
                for _ in range(75):
                    memory.ingest('At dawn, Sarah moved the keys to the shelf.')
                    memory.ingest('At noon, Sarah placed the map into the drawer.')
                counts = memory.counts()
                self.assertLessEqual(counts['links'], 5 * counts['events'])
                self.assertGreater(counts['links'], 0)
                related = memory.associate('Sarah', 'map', 4)
                self.assertLessEqual(len(related['related']), 4)
            with Memory(db, LinkLearner.load(model_path)) as memory:
                self.assertEqual(memory.counts(), counts)
                self.assertEqual(memory.recall("Where are Sarah's keys now?")['answer'], 'shelf')
                self.assertLessEqual(len(memory.associate('Sarah', 'map', 4)['related']), 4)

    def test_old_database_schema_migrates_non_destructively(self):
        with tempfile.TemporaryDirectory() as folder:
            db = Path(folder) / 'old.db'
            conn = sqlite3.connect(db)
            conn.execute('''CREATE TABLE links (
                source INTEGER NOT NULL, target INTEGER NOT NULL,
                strength REAL NOT NULL, relation TEXT NOT NULL,
                PRIMARY KEY(source,target))''')
            conn.execute('INSERT INTO links VALUES (1,2,0.8,"temporal-association")')
            conn.commit()
            conn.close()
            with Memory(db, self.model) as memory:
                row = memory.conn.execute('SELECT relation,evidence FROM links').fetchone()
                self.assertEqual(tuple(row), ('temporal-association', '[]'))

    def test_notes_cannot_create_speculative_context_or_causation(self):
        with tempfile.TemporaryDirectory() as folder:
            with Memory(Path(folder) / 'safe.db', self.model) as memory:
                memory.ingest('Sarah might be forgetting because of stress.')
                memory.ingest('Sarah may have moved her keys.')
                self.assertEqual(memory.counts()['links'], 0)
                self.assertIsNone(memory.recall("Why did Sarah's keys?")['answer'])


if __name__ == '__main__':
    unittest.main()
