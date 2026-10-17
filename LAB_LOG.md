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

## 2026-10-17 · Run 9: top-1 becomes the target metric

- **Changed:** one change, the metric (decided with the user). `sonara.py curve` now reports top-1 as the target, with recall@5 and `answer_ok` alongside. The learner is unchanged. Real data so far: 1 memory, 0 questions, so this is again synthetic only.
- **Result (synthetic test, later half, 7 eval questions):** top-1: Sonara 43%, control 57%, baseline 57%. recall@5: Sonara 71%, control 57%, baseline 57% (as in Run 8). The test still asserts recall@5, the metric it was built for.
- **Wrong answers:** all 4 of Sonara's top-1 misses are decoy questions ("where did I leave the book / car / wallet / charger"), and in each the spare-key note is ranked first. Sonara wins every "where did I leave it" question (4/4; baseline and control 0/4). The cause is `log_useful` (+0.70): it counts how often a memory was useful for *any* question, so the most-used memory gets a boost everywhere. Neighbor vote (+0.80) is question-specific but doesn't separate here, because "where did I leave it" is close to every decoy question. Cosine fell to +0.47. Recall@5 hid this: the decoy was still in the top 5, just behind the popular memory. A question-conditional replacement for the popularity count would be a later, separate change.

## 2026-10-17 · Run 10: shadow-mode change 1 (persistence, incremental newer-similar, bounded candidates)

- **Changed:** three pieces, each its own commit and test, none meant to change results. (a) `newer_similar` is now a saved running-max record per memory, updated once per new memory instead of an N x N comparison on every question. (b) Sonara's weights and optimizer state are saved with the feature list and a hash of the code; when either changes, or the saved questions are no longer a prefix of today's, all train feedback is replayed in time order. (c) Every method ranks from one bounded candidate set: the top 50 memories by cosine plus the 10 newest (sizes chosen before any data). `curve` reports how often a correct memory is in that set. Session counts and life position still read the whole timeline, not just the candidates.
- **Checks:** the incremental index equals the N x N version at three different seconds; saved-then-continued state equals a full replay bit for bit, and equals `run`'s final weights; a stale code hash forces a replay.
- **Result:** synthetic test unchanged (later half recall@5: Sonara 71%, control 57%, baseline 57%; same final weights), as expected, because it has 7 memories and the bounded set only binds above 60. Correct memory in the candidate set: 13/13.
- **Wrong answers:** same as Run 9; nothing in this change touches ranking below 60 memories. Whether the bounded set loses memories only experience would find is a question for real data; the coverage line will show it.

## 2026-10-17 · Run 11: shadow-mode change 2 (Sonara ranks alongside the baseline in chat)

- **Changed:** one change. In `chat` the baseline still picks the memories Qwen answers from; Sonara ranks the same bounded candidate set alongside it. Each question row now stores both full rankings (`rankings.baseline`, `rankings.sonara`). After a train-split question is logged, Sonara learns it at once (`load_learner`) and saves; eval-split questions are never learned. The labeling prompt now says a memory is marked useful for the question, true or not.
- **Checks:** a scripted 8-question chat (Qwen stubbed) logs both rankings on every row; the saved state lists exactly the train question ids; a labeled eval question afterwards leaves the weights and state unchanged.
- **Result:** no ranking change, so no new numbers. Real data is still 1 memory, 0 questions. Sonara's live ranking is not shown to the user and not used for answers; change 3 (union labeling pool) is what lets memories only Sonara finds be labeled.

## 2026-10-17 · Run 12: shadow-mode change 3 (union labeling pool)

- **Changed:** one change. The labeling pool in `chat` was the top 10 by cosine plus the 5 newest. It is now the baseline's top 5, Sonara's top 5 and the 5 newest (up to 15), shuffled and shown without scores. Each question row stores `proposed_by`: which systems put each candidate in the pool.
- **Checks:** a memory ranked last by the baseline but first by Sonara lands in the pool credited to Sonara only; every candidate has a proposer; in the scripted chat, Sonara's top 5 is always inside the pool.
- **Result:** no numbers; real data is still 1 memory, 0 questions. Shadow mode is now complete: the baseline answers, Sonara ranks and learns from train labels, and both can be credited. Note for later analysis: the pool now depends on Sonara's weights, so Sonara's labels are no longer drawn only from cosine and recency. The baseline is still compared on the same labels, and `proposed_by` lets the per-system credit be checked.

## 2026-10-17 · Run 13: first end-to-end shadow session with real Qwen (scripted, trial store)

- **Changed:** nothing in the code. First run of the full shadow `chat` against real Ollama (qwen2.5:1.5b answers, qwen3-embedding vectors), driven by a script, in a separate store (`learning_data/sonara_trial`) so the real eval set stays clean. **These are questions I wrote, not your experience, so they are not evidence.**
- **Data:** 11 notes, 10 questions, then 7 updated notes (dentist moved, rent raised, new longest run, new book, sister's dates, new gym, standup time) plus 1 knee note, then 20 reworded questions. Each question was labeled with the note that was current when asked. `answer_ok` came from a keyword check. All of it happened within about 2 minutes, so the clock inputs carry nothing. 30 questions: 24 train, 6 eval.
- **Live rankings (logged at ask time):** before the updates, both systems put the correct note first on all 10 questions. After the updates, the correct note was first for the baseline on 1 of 20 questions and for Sonara on 12 of 20. On the first three questions after the updates, Sonara was *worse* (correct note at 27, 17, 12 vs. the baseline's 4), then it caught up. The baseline in `chat` uses the default settings (recency on, no source bias), so Qwen's recent questions, answers and follow-ups filled its top slots. In 5 questions only Sonara proposed the correct note (`proposed_by`), so the union pool did its job.
- **Curve (6 eval questions, 3 periods):** top-1 baseline 50 / 0 / 0%, Sonara 50 / 100 / 100%, control 50 / 100 / 100%. **Sonara minus control is 0.** The gain comes from inputs that don't use experience: source weights (note +2.1, generated memories about −2.1) and `newer_similar` −1.2, which pushes down a note once a similar, newer note exists. The experience weights all ended negative (useful count −0.65, vote −0.44), as in Run 3: past usefulness pointed at the old versions. Correct memory in the candidate set: 6/6.
- **Wrong answers (all 12 `answer_ok` = n read):** 2 are false negatives from the keyword check ("8th to the 10th of November", "October 3"). The other 10 are real, and every one is a stale fact: Friday dentist → "Tuesday", $1525 → "$1450", Nov 15 → "Nov 8", Hail Mary → "Dune", Crunch $55 → "Planet Fitness $25", 9:30 → "10:00". Stale answers spread: once "$1450" was stored as a Qwen answer, it came back for the next rent question. In 5 of the 10, Sonara's shadow ranking had the current note first while Qwen answered from the baseline's top 5.
- **Reading:** this backs up the earlier guess (Runs 1, 5, 9) that the bottleneck is choosing the current version, and `newer_similar` plus source weights handle much of it without any experience. Two things here should not be read as Sonara beating the baseline: the `chat` baseline is untuned, and I designed the data.

## 2026-10-17 · Run 14: chat answers from the tuned baseline

- **Changed:** one change. `chat` used the baseline's default settings (recency on, no source bias). Now it uses `live_params`: tuned on labeled train questions at start, then re-tuned every 10 labeled train questions, the same schedule `curve` uses. So the system Qwen answers from is the same fair baseline that `curve` scores, and `answer_ok` measures that system. `TUNE_EVERY` moved into `sonara_recall.py` so both share it.
- **Checks:** in a scripted 20-question chat, the tuner is called at start and after the 10th labeled train question, and the next question's answering ranking uses the new settings. The test stubs the tuner and fixes the split, because a first version that relied on real tuning failed on 3 of 8 random splits.
- **Found while testing:** coordinate ascent stays on the defaults when question and answer memories crowd a note *together*: no single bias change gains a hit, so ties keep the current value (the same weakness as Run 3). On those splits the "tuned" baseline is the default one. This is not fixed here; it's a candidate for its own change (e.g. a joint search over source biases), decided before looking at real data.
- **Result:** no numbers. The goal stays slow improvement over time, measured on real timestamped chats, with no verdict before about 150 eval questions.

## 2026-10-17 · Run 15: the baseline tuner searches all combinations

- **Changed:** one change. The baseline tuner was coordinate ascent (one setting at a time, ties keep the current value). It got stuck when two generated sources crowd a note together: no single bias gains a hit (Run 14). It now tries every combination (5 recency weights × 3 half-lives × 5 biases per source). Each question's cosines, ages and sources are computed once, and all combinations are scored together. The objective is unchanged (recall@5 hits on train), and ties keep the defaults.
- **Checks:** a new test where user-question and follow-up memories crowd the note together. The old tuner, run on the same data, stays at the defaults and misses; the new one sets both biases negative and hits. All tests pass. `curve` on the trial store takes 1.3 s.
- **Result (trial store, my data, not evidence):** baseline recall@5 rose to 100 / 100 / 100% (it was 100 / 50 / 100%). Baseline top-1 stayed 50 / 0 / 0%. Sonara and the control are unchanged at 50 / 100 / 100%. After 20 train questions the tuned settings are recency off, generated sources −0.4, notes −0.2.
- **Wrong answers (all 5 baseline top-1 misses on eval):** 3 have the *old* version of the fact first ("Tuesday Oct 21" for the dentist twice, "Currently reading Dune", "Planet Fitness $25"). 2 have an earlier question first ("When is my dentist appointment?", "What time is standup?"), from before the tuner had 10 labels. Ranked by these settings, the baseline can't tell which of two similar notes is newer; that's what Sonara's `newer_similar` input does. Two possible later changes, each separate: tune the baseline for top-1 (the target) instead of recall@5, and decide whether the baseline should also get `newer_similar`. That second one is a fairness question to settle before real data, not after.

## 2026-10-17 · Run 16: the baseline gets newer_similar, tuned for top-1

- **Changed:** the baseline now adds `newer × newer_similar` (Sonara's input: how similar the most similar newer memory is), with `newer` tuned from (−0.4, −0.2, −0.1, 0), default 0. Decided with the user so the baseline has the same information as Sonara apart from experience. A second change was required for that: under a recall@5 objective, `newer` can't be tuned (the current note is already in the top 5, so pushing the old one down gains nothing, and the new test showed it staying at 0). So the tuner now ranks settings by top-1 hits first (the target metric), then recall@5. The two effects are reported separately below.
- **Checks:** a new test with an updated note: the tuner picks `newer` < 0 and the current note moves to first. All tests pass. `curve` on the trial store takes 1.4 s.
- **Result (trial store, my data, not evidence), baseline top-1 by period:** before 50 / 0 / 0%. With only the top-1 objective (`newer` held at 0): 50 / 0 / 0%, no change. With `newer` as well: 50 / 50 / 0%. Sonara and the control are unchanged at 50 / 100 / 100%.
- **Wrong answers (4 baseline top-1 misses on eval):** 1 is from before the tuner had 10 labels (an earlier question ranks first). In the other 3 the old version still comes first, by a small margin: the current note is second ("Currently reading Dune", "Planet Fitness $25", "Tuesday Oct 21"). Tuning picked `newer` −0.4, the edge of the grid, so a wider grid might do better. Sonara's ratio is similar (newer −1.21 against cosine 2.89, about −0.42 per unit of cosine), but it separates sources much more (note +2.1 vs generated −2.1). The bias grid is capped at −0.4, and that cap may be what limits the baseline. Widening either grid is a separate change, to be decided on reasoning, not on this data.

## 2026-10-17 · Run 17: wider baseline grids, with a margin tie-break

- **Changed:** decided with the user, so the baseline isn't capped below what Sonara can learn. `newer` grid widened to (−1.6, −0.8, −0.4, −0.2, −0.1, 0). Source biases go from −1.6 to +0.4 and are now *relative to notes*: notes stay at 0, because only differences between source biases change a ranking. That keeps the search to 46k combinations with 4 sources, scored in chunks. Widening exposed a tie-break problem, so two more rules were needed (results below):
  1. Ties used to go to the first combination in grid order, which is now the extreme end (−1.6).
  2. "Closest to the defaults" picked the smallest settings that just barely fit train (10/10, 19/20). On eval those failed by a hair: Qwen follow-ups ranked first, and baseline top-1 fell to 50 / 0 / 0%.
  3. Final rule: most top-1 hits, then most recall@5 hits, then the widest margin between the best correct memory and the best other one (capped at ±1 per question; Sonara's cross-entropy rewards margin the same way), then closest to the defaults.
- **Checks:** all tests pass, including the updated-note test (`newer` < 0, current note first). The test's starting point was fixed so it doesn't depend on the tie-break.
- **Result (trial store, my data, not evidence), top-1 by period:** baseline 50 / 100 / 50% (Run 16: 50 / 50 / 0%). Sonara and the control: 50 / 100 / 100%. After 20 train questions the tuned settings are recency 0.4 with a 1-day half-life, newer −0.8, and generated sources −0.8.
- **Wrong answers (2 baseline top-1 misses on eval):** "What day do I see the dentist?" was asked before the tuner had 10 labels, so the default settings rank an earlier question first. In "What's my gym membership cost?", "Planet Fitness $25" still beats "Crunch $55" (the correct note is second). The gap between baseline and Sonara is now one question. Sonara minus control is still 0, so nothing so far comes from experience.
- **Cost:** with 5 sources (your real store has one legacy `user_observation` memory), a tune takes about 16 s at 300 train questions. `chat` re-tunes every 10 labels, so that's an occasional pause.

## 2026-10-17 · Run 18: mock life, 15 simulated days of timestamped use (real Qwen, trial store)

- **Changed:** nothing in the code. I played the user through `chat` with real Qwen, in `learning_data/sonara_mock_life`. A simulated clock gave every memory and question a real timeline: morning and evening sessions from Oct 20 to Nov 3, a few minutes between actions. Each day had diary notes and fact notes, and 7 of 11 facts changed on set days (dentist twice, run twice, book, garage code, standup, gym, Maya). Questions came in two kinds. *Direct* ones are worded like the note. *Shorthand* ones use words the note lacks ("When do I see Dr. Lee?", "What's the usual?", "What's my PR?", "Where's the spare?"). Labels were the note current when asked. The planned 21 days stopped at day 15, when an Ollama call hung for 600 s. **Mock data I wrote, not evidence.**
- **Data:** 96 questions (62 train, 34 eval), 330+ memories. `answer_ok` 27/34.
- **Curve, top-1 by period:** baseline 67 / 89 / 33 / 14%, control 22 / 78 / 44 / 14%, Sonara 22 / 67 / 56 / 29%. By kind:

  | eval questions | n | baseline | control | Sonara | correct in candidates |
  |---|---|---|---|---|---|
  | direct | 17 | 82% | 59% | 59% | 17/17 |
  | shorthand | 17 | 24% | 24% | 29% | 13/17 |
  | shorthand, second half | 11 | 9% | 18% | 36% | |
  | after a fact changed | 10 | 10% | 20% | 30% | 8/10 |

- **First sign of experience:** for the first time the experience weights came out positive (neighbor vote +1.24, useful count +0.46, shown-not-useful −1.08). In the second half, Sonara got shorthand questions the control missed ("What's my distance record?" → the run note). That's only 2 questions out of 11, so it is a direction, not a result.
- **Wrong answers (all 22 eval misses read), three structural problems:**
  1. **Age bias (9 misses).** Sonara and the control often put the *first* memory ever written ("Dentist appointment is Tuesday the 21st") first for unrelated questions (rent, garage code, coffee, standup, Maya). Their clock inputs learned "older notes are useful" (gap +0.38, life position +0.31, time of year +0.55): in this mock most labeled facts were written early and the diary notes came later. That correlation is in the data, not about time. This is why they trail the baseline on direct questions (59% vs 82%).
  2. **`newer_similar` no longer means "replaced" (5 misses).** The old version still wins after updates (rent $1450, dentist Oct 31, Maya Nov 8). Sonara's `newer_similar` weight went *positive* (+0.18). Every useful fact gets restated by later Qwen questions, answers and follow-ups, so a high `newer_similar` now marks a memory that gets talked about, not one that was replaced. The baseline inherits the same input.
  3. **The candidate set hides what only experience finds (4 misses).** For "What's my PR?", "When's the car due?", "When's my thing with Dr. Lee?" and "What's the usual?", the correct note wasn't in the top 50 by cosine or the 10 newest, so no method could rank it. Experience can't help with a memory it never sees.
- **Possible next changes, one at a time:** (2) compute `newer_similar` only against newer *notes*, so Qwen's restatements don't count; (3) add memories marked useful for similar past train questions to the shared candidate set; (1) look at the age bias again after those two, because part of it may come from them.

## 2026-10-17 · Run 19: newer_similar counts only newer notes

- **Changed:** one change, decided with the user after Run 18. `newer_similar` used to compare a memory against every newer memory. Now it compares only against newer *notes* (`user_note`, plus the legacy `user_observation`). Qwen's questions, answers and follow-ups restate useful facts, so a newer Qwen memory no longer counts as replacing anything. The saved index records which sources count and rebuilds when that changes. Sonara and the baseline both use the new input.
- **Checks:** the incremental index still matches the all-pairs version when only notes count. A note that Qwen restated word for word keeps a low `newer_similar`, and it rises once a newer, similar note is written.
- **Result (mock-life store from Run 18, same data, re-scored), top-1 on 34 eval:**

  | | baseline | control | Sonara |
  |---|---|---|---|
  | all | 53 → 56% | 41 → 47% | 44 → 47% |
  | direct (17) | 82 → 82% | 59 → 71% | 59 → 59% |
  | shorthand (17) | 24 → 29% | 24 → 24% | 29 → 35% |
  | after an update (10) | 10 → 20% | 20 → 40% | 30 → 30% |

  Sonara's `newer_similar` weight went from +0.18 to −1.65, now pointing the right way. Experience weights are about the same (vote +1.24, useful +0.46).
- **Wrong answers (all 18 Sonara misses read):**
  - *Experience holds onto the old version (2):* "What day is my dentist cleaning?" twice ranks the Oct 31 note first, while the control gets the Nov 20 one. The Oct 31 note had been marked useful several times before the update, and the useful count and neighbor vote still point at it. Learning from past usefulness and noticing updates pull in opposite directions. The same thing likely explains rent ($1450 over $1525) and Maya (Nov 8 over Nov 15).
  - *Age bias (7):* still the first memory ever written ("Dentist appointment is Tuesday the 21st") ranked first for unrelated questions (code, coffee, rent, Maya, standup). The log-seconds inputs aren't scaled, so a gap of 1.2M s vs 100k s outweighs cosine differences. Gap +0.43 and life position +0.28 move together and the first memory is an outlier on both.
  - *Outside the candidate set (4):* PR, car due, Dr. Lee, the usual, as in Run 18.
  - *Wrong but nearby (5):* "car made a weird noise" over "oil change due" (3), "Planet Fitness" over rent (1), diary notes (2).
- **Reading:** this fix was needed: the input now means "replaced". On its own, Sonara is still level with the control and below the baseline overall. Next, as planned: the candidate set (so experience can reach shorthand memories), then the age bias. The pull between experience and updates is a new finding to keep watching. These are my mock data, not evidence.

## 2026-10-17 · Run 20: the candidate set includes what experience links to the question

- **Changed:** one change, decided with the user. The shared candidate set (top 50 by cosine + 10 newest) now also includes the memories marked useful for the 5 most similar earlier *train* questions. Eval labels are never used, and only questions asked before count. Every method ranks from this set, the baseline included. Before this, memories that only experience links to a question ("What's my PR?" → the run note) couldn't be ranked at all.
- **Checks:** a memory that cosine misses joins the set once it's marked useful for a similar earlier train question. The same label on an eval question doesn't pull it in, and neither does a question asked later.
- **Result (mock-life store, same data, re-scored), top-1 on 34 eval:**

  | | baseline | control | Sonara |
  |---|---|---|---|
  | all | 56 → 56% | 47 → 38% | 47 → 47% |
  | direct (17) | 82 → 82% | 71 → 53% | 59 → 47% |
  | shorthand (17) | 29 → 29% | 24 → 24% | 35 → 47% |
  | shorthand, 2nd half (11) | 18 → 18% | 18 → 18% | 36 → 55% |
  | after an update (10) | 20 → 20% | 30 → 30% | 30 → 40% |

  Correct memory in the candidate set: 30/34 → 33/34. Experience weights grew (vote +1.24 → +1.58, useful +0.46 → +0.75).
- **The experience part:** on shorthand questions Sonara now beats the control by 23 points overall and by 36 points in the second half: 4 more correct out of 11, all from experience inputs, with labels only from earlier train questions. This is the first clear Sonara-minus-control gap. It is still mock data I wrote, with 11 questions.
- **Wrong answers (all 18 Sonara misses read):** in 10, "Dentist appointment is Tuesday the 21st" (the very first memory) ranks first for unrelated questions: code, coffee, rent, car, Maya, standup. The extra candidates are old notes that were often useful, and the unscaled log-seconds inputs (gap +0.49) favor old memories, so the age bias from Runs 18–19 now costs more. That's why direct questions dropped for both learners while the baseline, which ranks by cosine, held at 82%. The other misses: the Oct 31 dentist note over the Nov 20 one (3: experience holding onto the old version, as in Run 19), Planet Fitness over rent (2), the spare key or standup notes over coffee and car (3).
- **Next, as planned:** the age bias, by scaling the log-seconds clock inputs so they can't outweigh cosine. It is now the largest single cause of Sonara misses.
