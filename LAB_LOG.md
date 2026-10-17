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

## 2026-10-17 · Run 2: tuned, source-aware baseline on Run 1 data

- **Changed:** the baseline now adds a bias for each memory source. Its recency weight, half-life and source biases are tuned on train questions by coordinate ascent; ties keep the current value. Same memories and questions as Run 1, re-scored with no new Qwen calls.
- **Data:** Run 1's 20 questions, split by id hash into 14 train and 6 eval.
- **Result:** train recall@5 went from 11/14 (default) to 14/14 (tuned). Eval stayed at 5/6 either way. Tuning set the recency weight to 0 and left every source bias at 0. Removing recency alone fixed all train misses, so the source biases were never needed.
- **Wrong answers:** the one eval miss is "When is my sibling coming to visit?" The top 5 is Qwen's earlier sister/Denver question, answer and follow-up, plus the dentist question. This is a similarity problem, not recency, and no source bias was learned to fix it. The recency result partly comes from the test itself: the notes were back-dated by days while the chat memories were minutes old. In real use, notes and chats are interleaved in time.

## 2026-10-17 · Run 3: Sonara v1 sanity run on Run 1 data

- **Changed:** first Sonara v1 run (linear scorer, 9 inputs, 5 seeds), with the feedback-rule reference. The pools were rebuilt with `label_pool`, because Run 1 didn't record them.
- **Data:** Run 1's 20 questions: 14 train, 6 eval, no time split. This checks the code only; with this little data the numbers say nothing.
- **Result:** at every training size, the baseline got 5/6, the rule got 5/6 (bonus tuned to 0), and Sonara got 6/6 (sd 0). The gap is one question.
- **Wrong answers / why:** Sonara's win comes from source weights (note +4.4, generated sources about −4), not from experience. Its experience weights are negative (neighbor vote −4.6, useful count −4.8), because every question here is on a new topic, so a memory's past usefulness doesn't predict future usefulness. The baseline can apply source biases too, but its tuner maximizes hit counts and stops on ties: recency weight 0 already got train to 14/14, so it never tried the biases. Sonara's gradient on the full pool did find them. **That is a difference in optimizer strength, not in experience.** The next change should be one control that isolates experience: the same learner with the three experience inputs zeroed.
