"""Controlled synthetic experience-learning investigation of Sonara's actual reader/writer.

Run: python experiment_experience_learning.py --seeds 3 --steps 100
Synthetic benchmark labels are researcher-created supervision, NOT Sonara's inferences.
Outputs are local; this module never reads personal sessions or modifies production checkpoints.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import random
import tempfile
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from LLM_Training import (
    PAD, START, UNK, IdeaReader, TinyTransformer, answer_with_weights,
    build_examples, encode_data, extend_vocabulary, load_checkpoint,
    personal_loss, personal_tokens, validation,
)
from sonara_memory import SonaraMemorySystem

AGENTS = ("jake", "lucy", "max", "piper")
OBJECTS = ("ball", "stick", "toy")
PLACES = ("yard", "pond", "shed")
BASE_TIME = "2026-10-13"


def scenario(actor, helper, obj, start, middle, end, corrected, scene):
    # Times and evidence are independent observations, never concatenated as a
    # pre-interpreted story when they enter Sonara's memory.
    events = [
        f"12:15 {actor} carried the {obj} into the {start}.",
        f"12:16 the {obj} fell into the {middle}.",
        f"12:17 {helper} picked up a rake.",
        f"12:18 {helper} pulled the {obj} into the {end}.",
        (f"12:20 correction: the {obj} never reached the {end}; it stayed in the {middle}."
         if corrected else f"12:20 {actor} collected the {obj} from {helper}."),
    ]
    final_place = middle if corrected else end
    # Explicitly separate observation-supported questions from unknown motives.
    questions = [
        (f"where was the {obj} at 12:16?", middle, "time"),
        (f"where was the {obj} at 12:18?", middle if corrected else end, "correction" if corrected else "time"),
        ("who picked up a rake?", helper, "entity"),
        (f"who carried the {obj} into the {start}?", actor, "entity"),
        (f"what did {actor} chase?", "unknown", "uncertainty"),
        (f"where was the {obj} at the end?", final_place, "integration"),
        ("did 12:16 happen before 12:18?", "yes", "time"),
        (f"what fell into the {middle}?", obj, "integration"),
        (f"why did {helper} use a rake?", "unknown", "uncertainty"),
        (f"was the {obj} finally in the {middle}?", "yes" if corrected else "no", "correction"),
    ]
    return {"id": scene, "events": events, "questions": questions, "corrected": corrected,
            "actor": actor, "helper": helper, "object": obj}


def dataset(seed):
    """Scenario-level holdout: held-out (actor, object, middle) tuples never trained."""
    rng = random.Random(seed)
    rows = []
    for ai, actor in enumerate(AGENTS):
        helper = AGENTS[(ai + 1) % len(AGENTS)]
        for oi, obj in enumerate(OBJECTS):
            for mi, middle in enumerate(PLACES):
                start, end = PLACES[(mi + 1) % 3], PLACES[(mi + 2) % 3]
                # Paired correction variants to prevent a one-class shortcut.
                for corrected in (False, True):
                    rows.append(scenario(actor, helper, obj, start, middle, end, corrected,
                                         f"scene_{ai}{oi}{mi}{int(corrected)}"))
    rng.shuffle(rows)
    held_keys = set(rng.sample(sorted({(r['actor'], r['object']) for r in rows}), 3))
    train = [r for r in rows if (r["actor"], r["object"]) not in held_keys]
    held = [r for r in rows if (r["actor"], r["object"]) in held_keys]
    assert len(train) == 54 and len(held) == 18
    assert not {(r['actor'], r['object']) for r in train} & {(r['actor'], r['object']) for r in held}
    return train, held


def question_only(row, q):
    return f"{row['id']} {q}"


def with_experience(row, q):
    return f"observations {' '.join(row['events'])} question {q}"


def load_original(checkpoint):
    if not checkpoint.exists():
        raise FileNotFoundError(f"Existing baseline checkpoint required, not found: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = state.get("run_config", {})
    examples, seed = config.get("examples") or 8000, config.get("seed") or 7
    training, valid, *_ = build_examples(examples, seed, config.get("ideas_file") or "", config.get("details_file") or "")
    data = encode_data(training, valid)
    width, depth, context = state["architecture"]
    class Args:
        pass
    args = Args()
    args.width, args.layers = width, depth
    word = TinyTransformer(len(data['words']), context, width, depth)
    reader = IdeaReader(len(data['words']), width, data['label_names'], data['word_to_id'][PAD])
    writer = TinyTransformer(len(data['words']), context, width, depth, conditioned=True)
    load_checkpoint(str(checkpoint), data, args, word, reader, writer, torch.device('cpu'))
    return data, word, reader, writer


def first_word(reader, writer, data, question):
    words = personal_tokens(question)
    ids = data['word_to_id']
    t = torch.tensor([[ids.get(w, ids[UNK]) for w in words]], dtype=torch.long)
    with torch.no_grad():
        vector, _ = reader(t, torch.tensor([len(words)]))
        start = torch.tensor([[ids[START]]], dtype=torch.long)
        logits = writer(start, vector)[0, 0]
    return data['words'][int(logits.argmax())], logits


def accuracy(reader, writer, data, rows, context=False, shuffled=False, prompts_override=None):
    reader.eval(); writer.eval()
    correct, total, loss_sum, groups = 0, 0, 0., {}
    prompts = [with_experience(r, q) if context else question_only(r, q)
               for r in rows for q, _, _ in r['questions']]
    if prompts_override is not None:
        prompts = prompts_override
    if shuffled:
        prompts = prompts[len(prompts)//2:] + prompts[:len(prompts)//2]
    with torch.no_grad():
        for i, row in enumerate(rows):
            for j, (q, target, category) in enumerate(row['questions']):
                prompt = prompts[i * len(row['questions']) + j]
                predicted, logits = first_word(reader, writer, data, prompt)
                expected = data['word_to_id'].get(target, data['word_to_id'][UNK])
                loss_sum += float(F.cross_entropy(logits.unsqueeze(0), torch.tensor([expected])))
                matched = predicted == target
                correct += matched
                total += 1
                hit, count = groups.get(category, (0, 0))
                groups[category] = (hit + int(matched), count + 1)
    return {"accuracy": round(correct/total, 4), "n": total,
            "target_loss": round(loss_sum/total, 4),
            "by_category": {k: round(a/b, 4) for k, (a, b) in sorted(groups.items())}}


def measure_retrieval(rows):
    """Measures evidence coverage, NOT answered reasoning accuracy."""
    with tempfile.TemporaryDirectory(prefix='sonara-synthetic-memory-') as work:
        memory = SonaraMemorySystem(work)
        for row in rows:
            for event in row['events']:
                memory.remember(f"{row['id']} {event}", source='synthetic_experiment')
        hit, total, wrong_scene, subset, prompts = 0, 0, 0, {}, []
        for row in rows:
            for q, target, category in row['questions']:
                results = memory.recall(question_only(row, q), limit=5)
                texts = [m.content.lower() for m, _ in results]
                # Exact retrieved evidence, no gold answer or answer-specific filter.
                prompts.append(f"observations {' '.join(texts)} question {q}")
                wrong_scene += any(not s.startswith(row['id'] + ' ') for s in texts)
                if target in ('unknown','yes','no'):
                    continue  # An absent fact or logical relation cannot be validated by token presence.
                # Upper bound: answer appears somewhere in retrieved evidence.
                found = any(target in personal_tokens(s) for s in texts)
                hit += found
                total += 1
                a,b = subset.get(category, (0,0))
                subset[category]=(a+int(found),b+1)
    metrics = {"evidence_coverage": round(hit/max(1,total),4), "n": total,
            "wrong_scene_retrieval_count": wrong_scene,
            "by_category": {k:round(a/b,4) for k,(a,b) in sorted(subset.items())},
            "warning":"Retrieval evidence coverage is not answer accuracy."}
    return metrics, prompts


def train_on_synthetic(reader, writer, data, rows, steps, seed, lr):
    rng = random.Random(seed)
    pairs = [(question_only(row, q), a) for row in rows for q,a,_ in row['questions']]
    # Also train on observations supplied in CURRENT INPUT, for transfer to
    # novel combinations. No held-out events or answers enter this pool.
    pairs += [(with_experience(row, q), a) for row in rows for q,a,_ in row['questions']]
    optimizer = torch.optim.AdamW(list(reader.parameters()) + list(writer.parameters()),lr=lr)
    before = {k: p.detach().clone() for k,p in reader.named_parameters()}
    reader.train(); writer.train()
    first, final = None, None
    for step in range(steps):
        q,a = pairs[rng.randrange(len(pairs))]
        loss = personal_loss(reader, writer, q, a, data, torch.device('cpu'))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(reader.parameters()) + list(writer.parameters()), 1.0)
        optimizer.step()
        first = float(loss.detach()) if first is None else first
        final = float(loss.detach())
    change = sum(float((p.detach()-before[k]).abs().sum()) for k,p in reader.named_parameters())
    return {"first_batch_loss":round(first,4),"last_batch_loss":round(final,4),
            "reader_abs_parameter_delta":round(change,4),"steps":steps}


def run_seed(seed, steps, lr, checkpoint):
    torch.manual_seed(seed)
    start = time.monotonic()
    training, held = dataset(seed)
    data, word, reader, writer = load_original(checkpoint)
    # New vocab is permitted only from TRAIN experiences and supervised targets.
    train_text = [question_only(r,q)+' '+with_experience(r,q)+' '+ans
                  for r in training for q, ans, _ in r['questions']]
    lexicon = sorted({t for content in train_text for t in personal_tokens(content)})
    vocabulary_added = extend_vocabulary(lexicon,data,word,reader,writer)
    # Do not allow heldout vocabulary to be silently created by test examples.
    missing = sorted({w for r in held for q,_,_ in r['questions']
                      for w in personal_tokens(with_experience(r,q)) if w not in data['word_to_id']})
    retrieval, retrieved_prompts = measure_retrieval(held)
    valid_sample = {k: v[:64] for k,v in data['valid'].items()}
    def general_retention():
        return round(validation(3,writer,reader,valid_sample,data['word_to_id'][PAD],torch.device('cpu'),16)[0],4)
    before = {
        'general_detail_loss_sample':general_retention(),
        'weights_only_train_episodes':accuracy(reader,writer,data,training),
        'weights_only_heldout_episodes':accuracy(reader,writer,data,held),
        'novel_with_observations':accuracy(reader,writer,data,held,context=True),
        'integrated_retrieval_prompt':accuracy(reader,writer,data,held,prompts_override=retrieved_prompts),
    }
    gradients = train_on_synthetic(reader,writer,data,training,steps,seed,lr)
    after = {
        'general_detail_loss_sample':general_retention(),
        'weights_only_train_episodes':accuracy(reader,writer,data,training),
        'weights_only_heldout_episodes':accuracy(reader,writer,data,held),
        'novel_with_observations':accuracy(reader,writer,data,held,context=True),
        'novel_with_shuffled_observations':accuracy(reader,writer,data,held,context=True,shuffled=True),
        'integrated_retrieval_prompt':accuracy(reader,writer,data,held,prompts_override=retrieved_prompts),
    }
    # Only neural input changes here; no tool/runtime memory access allowed.
    # Existing public-data retention proxy uses an unchanged, predetermined batch.
    # Report question-answer separation explicitly: this harness was not called
    # through the interactive approval path and C4 is not a live integration test.
    return {"seed":seed,"train_scenarios":len(training),"heldout_scenarios":len(held),
            "heldout_questions":len(held)*10,"vocab_extended":vocabulary_added,
            "heldout_oov":missing,"original_checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "baseline":before,"memory_only":retrieval,"adaptation":gradients,
            "post_train":after,"runtime_seconds":round(time.monotonic()-start,2),
            "limitations":["Training Q/A targets are generated by a synthetic experimental oracle, not inferred by Sonara.",
                           "Novel-episode weights-only questions cannot reveal unseen episode-specific facts.",
                           "Memory-only scores token evidence coverage for direct entity/place answers; it does not score yes/no or unknown.",
                           "Integrated condition feeds retrieved text into the neural reader in the experiment; live session integration is not proved."]}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',default='LLM_Training_checkpoint_v2.pt')
    parser.add_argument('--seeds',type=int,default=1)
    parser.add_argument('--steps',type=int,default=60)
    parser.add_argument('--lr',type=float,default=0.002)
    parser.add_argument('--output',default='artifacts/experience_learning_report.json')
    args=parser.parse_args()
    if args.seeds not in (1,2,3) or not 1<=args.steps<=500: parser.error('seeds 1..3, steps 1..500')
    torch.set_num_threads(2)
    results=[run_seed(17+i,args.steps,args.lr,Path(args.checkpoint)) for i in range(args.seeds)]
    report={'experiment':'Sonara controlled experience learning - baseline investigation',
            'not_proof_of':'semantic understanding or autonomous temporal reasoning',
            'runs':results}
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    for r in results:
        print(f"Seed {r['seed']} | baseline weights-only {r['baseline']['weights_only_train_episodes']['accuracy']}"
              f" -> {r['post_train']['weights_only_train_episodes']['accuracy']}"
              f" | novel context {r['baseline']['novel_with_observations']['accuracy']}"
              f" -> {r['post_train']['novel_with_observations']['accuracy']}"
              f" | memory evidence {r['memory_only']['evidence_coverage']}")
    print('Saved synthetic-only report:',path)


if __name__=='__main__': main()
