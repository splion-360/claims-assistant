"""Score a claim_packet.json against a ground-truth file and print the results table.

Usage:
    .venv/bin/python scripts/score_packet.py claim_packet.json ground_truth/my_room.yml [--out results.md]
"""
import argparse
import json
import sys
from pathlib import Path

# Run from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from library_claim.evaluation import load_truth, results_markdown, score  # noqa: E402


def main() -> None:
    """Parse arguments, score the packet and write the Markdown results."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('packet', type=Path, help='claim_packet.json from a sweep')
    parser.add_argument('truth', type=Path, help='ground-truth YAML (see ground_truth/template.yml)')
    parser.add_argument('--out', type=Path, help='also write the Markdown results to this file')
    args = parser.parse_args()
    text = results_markdown(score(json.loads(args.packet.read_text()), load_truth(args.truth)))
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == '__main__':
    main()
