"""Thin, executable entrypoint for the Sonara memory experiment."""
import argparse
import json
from sonara_memory import LinkLearner, Memory, encode_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="sonara.db")
    parser.add_argument("--model", default="model.json")
    sub = parser.add_subparsers(dest="command", required=True)
    save = sub.add_parser("store")
    save.add_argument("paragraph")
    ask = sub.add_parser("ask")
    ask.add_argument("question")
    related = sub.add_parser("related")
    related.add_argument("actor")
    related.add_argument("item")
    related.add_argument("--limit", type=int, default=5)
    hierarchy = sub.add_parser("hierarchy")
    hierarchy.add_argument("actor")
    hierarchy.add_argument("item")
    hierarchy.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    learner = LinkLearner.load(args.model)
    with Memory(args.db, learner) as memory:
        if args.command == "store":
            print(encode_json(memory.ingest(args.paragraph)))
        elif args.command == "ask":
            print(json.dumps(memory.recall(args.question), indent=2))
        elif args.command == "related":
            print(json.dumps(memory.associate(args.actor, args.item, args.limit), indent=2))
        else:
            print(json.dumps(memory.hierarchy(args.actor, args.item, args.limit), indent=2))


if __name__ == "__main__":
    main()
