"""Compare Drex and the vision model on the judgement that decides the book count.

Two crops of a shelf: are they the same physical copy, or two different books?
Getting this wrong inflates the inventory, so it is the decision the claim's book
count rests on. Ground truth comes from the spine text each crop was read as,
which is independent of the association decision being tested.
"""
import argparse
import asyncio
import base64
import json
import random
import sqlite3
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from library_claim.config import CONFIG
from library_claim.decisions.same_book import SameBookCandidate, crop_jpeg, same_book_probabilities
from library_claim.domain.models import Box, BookObservation, Frame
from library_claim.storage.evidence import EvidenceStore
from library_claim.storage.repository import Repository
from provider import VISION_MODEL, structured

QUESTION = ('Image 1 and image 2 are each a crop of one book from a shelf. Decide whether they show '
            'the same single physical copy photographed twice, or two different books. Two copies of '
            'the same title standing side by side are different books. Judge the book itself, not the '
            'background. Answer with a probability between 0 and 1 that they are the same physical copy.')


class SameAnswer(BaseModel):
    """The same judgement, asked of the vision model."""
    same_physical_book: float


def crops(data: Path) -> list[dict]:
    """Collect every book crop whose spine was read as a title.

    Args:
        data: Capture data directory.

    Returns:
        One record per readable sighting, with its title and crop bytes.
    """
    database = data / 'claims.sqlite3'
    store = EvidenceStore(data / 'frames')
    found = []
    with Repository(database) as repo:
        for (sweep_id,) in sqlite3.connect(database).execute('SELECT id FROM sweeps'):
            frames = {frame.id: frame for frame in repo.list_for_sweep(Frame, sweep_id)}
            for row in repo.list_for_sweep(BookObservation, sweep_id):
                if row.provenance != 'visual' or not row.title or row.frame_id not in frames:
                    continue
                try:
                    image = cv2.imdecode(np.frombuffer(store.read(frames[row.frame_id].evidence_key),
                                                       np.uint8), cv2.IMREAD_COLOR)
                    if image is None:
                        continue
                    found.append({'id': row.id, 'title': row.title.strip().upper(),
                                  'view': row.view, 'author': row.author,
                                  'jpeg': crop_jpeg(image, row.bbox)})
                except (OSError, ValueError, KeyError, cv2.error):
                    continue
    return found


def pairs(records: list[dict], count: int, seed: int) -> list[tuple[dict, dict, bool]]:
    """Build a balanced set of same-book and different-book pairs.

    Args:
        records: Readable sightings.
        count: Pairs of each kind.
        seed: Sampling seed.

    Returns:
        Pairs with the label taken from their titles.
    """
    rng = random.Random(seed)
    same, different = [], []
    for first in range(len(records)):
        for second in range(first + 1, len(records)):
            a, b = records[first], records[second]
            (same if a['title'] == b['title'] else different).append((a, b))
    rng.shuffle(same)
    rng.shuffle(different)
    return ([(a, b, True) for a, b in same[:count]] + [(a, b, False) for a, b in different[:count]])


async def ask_vision(first: dict, second: dict) -> tuple[float, float]:
    """Ask the vision model whether two crops are the same physical copy.

    Args:
        first: One sighting.
        second: The other sighting.

    Returns:
        Its probability and the wall-clock latency in milliseconds.
    """
    def url(jpeg):
        """Encode a crop for the request.

        Args:
            jpeg: Crop bytes.

        Returns:
            A data URL.
        """
        return {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,'
                                                   + base64.b64encode(jpeg).decode()}}

    started = time.monotonic()
    answer = await structured(QUESTION, [url(first['jpeg']), url(second['jpeg'])],
                              SameAnswer, model=VISION_MODEL)
    return answer.same_physical_book, (time.monotonic() - started) * 1000


def drex_same(first: dict, second: dict) -> tuple[float, float, int]:
    """Ask Drex the same question through the pipeline's own decision.

    Args:
        first: One sighting.
        second: The other sighting.

    Returns:
        Its probability, latency in milliseconds and input tokens.
    """
    class Reading:
        """The readings the decision carries alongside the crops."""
        title, author, view = first['title'], first['author'], first['view']

    candidate = SameBookCandidate(book_id='other', title=second['title'], author=second['author'],
                                  view=second['view'], crop_jpeg=second['jpeg'], method='benchmark',
                                  score=0.0)
    started = time.monotonic()
    probabilities = same_book_probabilities(first['jpeg'], Reading(), [candidate])
    return probabilities[0], (time.monotonic() - started) * 1000, 0


async def main() -> None:
    """Run both deciders over the labelled pairs and report accuracy."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs', type=int, default=10, help='pairs of each kind')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--data', type=Path, default=Path('.claim-data'))
    parser.add_argument('--out', type=Path, default=Path('docs/same-book-benchmark.json'))
    args = parser.parse_args()

    records = crops(args.data)
    sample = pairs(records, args.pairs, args.seed)
    print(f'{len(records)} readable sightings -> {len(sample)} labelled pairs\n')
    rows = []
    for index, (first, second, label) in enumerate(sample, 1):
        row = {'truth': label, 'titles': [first['title'][:28], second['title'][:28]]}
        try:
            probability, latency, _ = await asyncio.to_thread(drex_same, first, second)
            row['drex'] = {'p': probability, 'ms': latency}
        except Exception as error:  # noqa: BLE001 - a failed call is a result
            row['drex'] = {'error': type(error).__name__}
        try:
            probability, latency = await ask_vision(first, second)
            row['vision'] = {'p': probability, 'ms': latency}
        except Exception as error:  # noqa: BLE001
            row['vision'] = {'error': type(error).__name__}
        rows.append(row)
        print(f'  {index:>3}/{len(sample)}  truth={"same" if label else "diff"}  '
              f'drex={row["drex"].get("p", "err")!s:<8} vision={row["vision"].get("p", "err")}')

    def score(key, threshold):
        """Accuracy and latency for one decider.

        Args:
            key: Which decider's rows to score.
            threshold: Probability at or above which it says same book.

        Returns:
            A summary dict.
        """
        usable = [row for row in rows if 'error' not in row[key]]
        if not usable:
            return {'answered': 0}
        right = [row for row in usable if (row[key]['p'] >= threshold) == row['truth']]
        same_right = [r for r in usable if r['truth'] and r[key]['p'] >= threshold]
        diff_right = [r for r in usable if not r['truth'] and r[key]['p'] < threshold]
        return {'answered': len(usable), 'accuracy': round(len(right) / len(usable), 3),
                'same_book_recall': round(len(same_right) / max(sum(1 for r in usable if r['truth']), 1), 3),
                'different_book_recall': round(len(diff_right) / max(sum(1 for r in usable if not r['truth']), 1), 3),
                'median_ms': round(sorted(row[key]['ms'] for row in usable)[len(usable) // 2])}

    summary = {'pairs': len(rows),
               'drex': {**score('drex', CONFIG.drex.same_book_min), 'model': CONFIG.drex.model},
               'vision': {**score('vision', 0.5), 'model': VISION_MODEL}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'summary': summary, 'rows': rows}, indent=2))
    print('\n' + json.dumps(summary, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
