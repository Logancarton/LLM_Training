# Lab log

One entry per run: what changed, result, what the wrong answers showed.

## 2026-10-17 · Run 0: pipeline smoke test (fixed baseline)

- **Changed:** first run. Baseline = cosine(Qwen question vector, memory vector) + 0.1 × recency (7-day half-life), untuned.
- **Data:** 12 hand-written notes (one per day), 9 questions with hand-labeled correct notes. Not saved; it tests the code only.
- **Result:** recall@5 = 9/9. Eight questions had the correct note at rank 1.
- **Wrong answers:** none. With only 12 memories, the top 5 covers 42% of everything, so recall@5 can't tell methods apart yet. Experiment 1 needs hundreds of memories before this metric means anything.

## 2026-10-17 · Run 1: 20-question pipeline test, with Qwen follow-ups stored

- **Changed:** follow-ups on; each question, answer and follow-up is stored as it goes, so later questions can recall them. Baseline settings unchanged (untuned).
- **Data:** 40 synthetic notes (one per day, with later notes updating earlier ones: dentist, rent, longest run, knee), 20 reworded questions with hand-labeled correct notes. Not saved; it tests the code only.
- **Result:** recall@5 = 16/20. With the Qwen-generated memories removed from the ranking, it's 20/20. Qwen-generated memories took 2.65 of the 5 top slots on average. Answers: 12/20 right, 2 partly right, 6 wrong (my judgment).
- **Wrong answers:** all 4 recall misses happened because a correct note was pushed out by past questions, answers and follow-ups. Those are minutes old and get the full recency bonus; the notes are days old. They are also worded like questions, so they score high on similarity. Wrong answers carry forward. "Knee ligament tear" was made up, stored, and recalled for the next question. "Dentist Oct 9" was a stale answer, and it came back as the answer to "when is my sibling visiting?". Three answers picked an old fact over the newer one (dentist, longest run, current book). That last result is partly a flaw in this test: the notes' shown dates were all today, because only their stored epoch was back-dated.
