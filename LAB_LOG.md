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

## 2026-10-17 · Run 4: no-experience control on Run 1 data

- **Changed:** added the control: the same learner, trained and ranked with an empty history, so its three experience inputs are always 0. Data is the same as Run 3.
- **Result:** at every training size, the control got 6/6, the same as Sonara. Sonara minus control = +0.0. The +16.7 over the baseline is all optimizer strength (source weights), with nothing from experience. This confirms Run 3's reading.
- **Test check:** in the synthetic test, the control first reached 100% by learning a negative cosine weight, a shortcut that test happened to allow. Adding questions where cosine is the right signal closes it. Now experience is the only way to find the target: control 46%, Sonara 69%, baseline 54%.

## 2026-10-17 · Run 5: day-clock inputs on Run 1 data

- **Changed:** one change, the clock. Age in days was replaced with six clock inputs: gap in log-seconds; same time of day, same day of week and same time of year (cosine of the cycle difference); log sessions between memory and now (a quiet hour or more starts a session); and similarity of the most similar newer memory. Sonara and the no-experience control both get them; the fixed baseline doesn't.
- **Result:** at every training size, Sonara got 6/6 (sd 0), except at 7 train questions, where one seed missed once (93.3%, sd 8.2). The control got 6/6 and the baseline 5/6. The clock changed nothing measurable here.
- **Weights:** newer similar memory −3.3, in the expected direction (a replaced memory is less useful). The time-of-day weight (−1.8) is an artifact of this test, not something learned about time: the notes were back-dated to the same time of day, and the chat memories were minutes old.
- **Wrong answers:** none at full training. recall@5 can't show whether the clock picks the current fact (dentist, longest run, current book), because old and new versions both land in the top 5.

## 2026-10-17 · Run 6: online learning in time order

- **Changed:** one change. Sonara and the control now learn one train question at a time, in ask order (Adam lr 0.05, 10 steps per question), and carry their weights forward. No shuffling, no retraining from scratch, no seeds. Every eval question is scored as of the second it was asked. The baseline and rule are re-tuned every 10 labeled train questions. The 3 time splits are replaced by 4 consecutive periods.
- **Data:** Run 1's 20 questions (14 train, 6 eval). This checks the code only.
- **Result by period** (2 eval questions each; only 3 periods with 6 questions): Sonara 50 / 50 / 100%, baseline 100 / 50 / 100%. The control and rule match the baseline. In the synthetic test's later half, Sonara got 86%, the control 43% and the baseline 57%.
- **Wrong answers:** both of Sonara's misses ("why wouldn't my vehicle start", "longest run") have top 5s padded with the oldest notes ("Started the Sonara project", the first dentist booking). After a few big steps on few questions, the weights favor older memories (log gap +0.34, time of year +2.0) and rank below its own cosine-only start. This is online-learning noise: lr 0.05 × 10 steps moves the weights a lot on each question. A slower learning rate would be the next single change, chosen before seeing new data, not tuned on this set.

## 2026-10-17 · Run 7: absolute timeline position

- **Changed:** one change. A new clock input, `log_life_seconds`: seconds from Sonara's first memory to this memory, log scale. Same online learner and data as Run 6.
- **Result by period:** Sonara 50 / 50 / 100% (same as Run 6). The control dropped to 50% in period 1 (it was 100%). The baseline got 100 / 50 / 100%.
- **Weights:** life position +1.1, so later memories are favored. Time of year fell from +2.0 to +0.6 and newer-similar from −2.1 to −1.8. The new input took over part of what the cycles were absorbing.
- **Wrong answers:** the same two misses. "Longest run" now ranks running notes in its top 5 ("Ran 10 km", the half marathon, the shoes) instead of the Sonara-start note, but still not the 18 km note. "Vehicle start" still ranks the brake notes above the battery notes. The early-period drop is still the online step size (Run 6), not the new input.

## 2026-10-17 · Run 8: slower online learning rate

- **Changed:** one change. Online Adam learning rate 0.05 → 0.01. The value was chosen before running, not tuned on this data.
- **Result by period:** Sonara 100 / 50 / 100%, the same as the baseline, rule and control in every period. The early drop from Runs 6–7 (50% in period 1) is gone; Sonara no longer falls below its starting point. In the synthetic test's later half: Sonara 71% (it was 86%), control 57%, baseline 57%. Slower learning still learns, just less in the same number of questions.
- **Weights** are smaller everywhere (cosine 4.3 → 1.8, newer-similar −1.8 → −0.6), so they stay nearer the cosine-only start.
- **Wrong answers:** one miss, "longest run". The 18 km note is still missing from the top 5, which holds the running shoes, two project notes and the book. The baseline and control miss it too in that period, so this is a cosine problem the experience hasn't fixed yet, not a learning problem.
