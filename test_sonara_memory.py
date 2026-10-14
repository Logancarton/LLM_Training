"""
test_sonara_memory.py — Automated verification of Sonara's memory system
"""

import time
import shutil
from pathlib import Path
from sonara_memory import SonaraMemorySystem, format_relative_time


def test_memory_system():
    test_dir = Path("test_sonara_memories")
    if test_dir.exists():
        shutil.rmtree(test_dir)

    print("=== TEST 1: Initializing Sonara Memory ===")
    mem_sys = SonaraMemorySystem(str(test_dir))
    assert len(mem_sys.memories) == 0
    print("✓ Memory initialized successfully")

    print("\n=== TEST 2: Recording Memories with Timestamps ===")
    m1 = mem_sys.remember(
        content="Logan's old dog Jake liked tennis balls.",
        category="episodic",
        importance=8.5,
    )
    print(f"  Memory 1 ID:        {m1.id}")
    print(f"  ISO-8601 Timestamp: {m1.timestamp_iso}")
    print(f"  Human Timestamp:    {m1.timestamp_human}")
    print(f"  Relative Time:      {m1.relative_time()}")
    print(f"  Entities Extracted: {m1.entities}")
    print(f"  Semantic Frame:     {m1.semantic_frame}")

    assert m1.timestamp_iso is not None
    assert m1.timestamp_epoch > 0
    assert "Logan" in m1.entities or "logan" in m1.entities
    assert "Jake" in m1.entities or "jake" in m1.entities

    # Add second related memory
    m2 = mem_sys.remember(
        content="Jake caught the tennis ball in the yard.",
        category="episodic",
        importance=6.0,
    )

    # Add a user preference
    m3 = mem_sys.remember(
        content="User prefers concise explanations and active code verification.",
        category="preference",
    )
    assert m3.category == "preference"
    assert m3.importance >= 8.5

    print("✓ Recorded episodic and preference memories with timestamps")

    print("\n=== TEST 3: Cognitive Recall & Search ===")
    results = mem_sys.recall("Who was Jake?")
    assert len(results) > 0
    best_mem, score = results[0]
    print(f"  Query: 'Who was Jake?'")
    print(f"  Top Match (Score {score:.2f}): \"{best_mem.content}\"")
    print(f"  Timestamp: {best_mem.timestamp_human} ({best_mem.relative_time()})")
    assert "Jake" in best_mem.content

    pref_results = mem_sys.recall("What does user prefer?")
    assert len(pref_results) > 0
    print(f"  Preference Query Top Match: \"{pref_results[0][0].content}\"")

    print("\n=== TEST 4: Memory Reflection / Consolidation ===")
    reflections = mem_sys.reflect()
    print(f"  Generated {len(reflections)} reflection(s):")
    for r in reflections:
        print(f"  • [{r.timestamp_human}] {r.content}")
    assert len(reflections) >= 1

    print("\n=== TEST 5: Timeline & Journal Generation ===")
    timeline = mem_sys.get_timeline(limit=10)
    print(f"  Timeline has {len(timeline)} chronological entries:")
    for item in timeline:
        print(f"    - [{item['timestamp_human']}] ({item['relative_time']}) [{item['category'].upper()}] {item['content']}")

    journal_path = mem_sys.export_journal()
    assert journal_path.exists()
    journal_content = journal_path.read_text(encoding="utf-8")
    assert "Sonara's Memory Journal" in journal_content
    print(f"✓ Markdown journal successfully generated at: {journal_path}")

    # Cleanup test dir
    if test_dir.exists():
        shutil.rmtree(test_dir)

    print("\n🎉 ALL TESTS PASSED! Sonara's memory system is fully operational.")


if __name__ == "__main__":
    test_memory_system()
