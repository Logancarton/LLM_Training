"""
test_deliberation.py — DELIBERATIVE REASONING: ASSESS -> CREATE -> VERIFY -> RE-ASSESS
========================================================================================

PURPOSE
    Demonstrates deliberative generation (System 2 / Process-Supervision / Test-Time Search).
    Rather than blind autoregressive dumping, at EACH token step the system:
      1. ASSESS STATE: Evaluates what has been written vs what semantic goals remain.
      2. CREATE: Proposes candidate next words with model confidence probabilities.
      3. ASSESS STEP (Verify / Critic): Evaluates candidate against target goals.
      4. RE-ASSESS: Checks if the situation meaning is complete and decides next action.

RUN
    python3 test_deliberation.py
    python3 test_deliberation.py --idea "the dog sees the cat"
    python3 test_deliberation.py --idea "the cat sleeps by the window"
"""

import argparse
import torch

from LLM_Training import (
    DEFAULT_IDEAS_FILE,
    DEFAULT_DETAILS_FILE,
    DEFAULT_LAYERS,
    PAD,
    build_data,
    TinyTransformer,
    IdeaReader,
    train_idea_reader,
    train_detail_writer,
    generate_with_assessment,
)


def main():
    parser = argparse.ArgumentParser(description="Test deliberate assess-create-reassess generation loop")
    parser.add_argument("--ideas-file", default=DEFAULT_IDEAS_FILE, help="Path to ideas training file")
    parser.add_argument("--details-file", default=DEFAULT_DETAILS_FILE, help="Path to details training file")
    parser.add_argument("--steps", type=int, default=300, help="Training updates (default: 300)")
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS, help=f"Number of stacked Transformer layers (default: {DEFAULT_LAYERS})")
    parser.add_argument("--idea", default="the cat chases the ball", help="Situation to deliberately generate")
    args = parser.parse_args()

    torch.manual_seed(7)
    if torch.get_num_threads() < 4:
        try:
            torch.set_num_threads(4)
        except Exception:
            pass

    print("=" * 70)
    print("      DELIBERATIVE GENERATION LOOP (ASSESS -> CREATE -> RE-ASSESS)")
    print("=" * 70)

    data = build_data(args.ideas_file, args.details_file)
    vsize = len(data["word_to_id"])
    ctx = data["context_size"]

    print(f"\n[1/2] Training Stage 2 Idea Reader ({args.steps} steps)...")
    reader = IdeaReader(vsize, data["label_names"], data["word_to_id"][PAD])
    train_idea_reader(reader, data, args.steps)

    print(f"[2/2] Training Stage 3 Generator ({args.layers} layers, {args.steps} steps)...")
    writer = TinyTransformer(vsize, context_size=ctx, conditioned=True, num_layers=args.layers)
    train_detail_writer(writer, reader, data, args.steps)

    reader.eval()
    writer.eval()

    # Run the interactive Assess -> Create -> Verify -> Re-assess loop
    generate_with_assessment(writer, reader, args.idea, data)


if __name__ == "__main__":
    main()
