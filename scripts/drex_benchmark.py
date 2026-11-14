"""Compare Drex frame admission against the frontier vision model on the same saved frames.

Drex replaces a judgement a vision model would otherwise make: does this frame
show a book worth reading? The vision model's own detection on the same frame is
the reference, so agreement, latency and cost are measured on identical inputs.
"""
import argparse
import asyncio
import json
import random
import sqlite3
import sys
import time
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from library_claim.decisions.frame_check import check_frame
from library_claim.config import CONFIG
from library_claim.domain.models import Frame
from library_claim.storage.evidence import EvidenceStore
from library_claim.storage.repository import Repository
from provider import VISION_MODEL, structured

BASELINE_INSTRUCTION = (
    'Look at this camera frame from a home library sweep. Decide whether it shows at least one '
    'physical book whose spine or cover is worth reading for a title, and whether any title in it '
    'is legible. Judge only what is visible. Answer with probabilities between 0 and 1.'
)


class BaselineCheck(BaseModel):
    """The same judgement, asked of the frontier vision model."""
    book_visible: float
    spine_legible: float


def saved_frames(database: Path, data_dir: Path, limit: int, seed: int) -> list[tuple[str, bytes]]:
    """Collect saved frames to replay through both deciders.

    Args:
        database: Capture SQLite path.
        data_dir: Capture data directory holding the frame store.
        limit: Maximum frames to sample.
        seed: Sampling seed, so a run can be repeated.

    Returns:
        Frame IDs paired with their JPEG bytes.
    """
    store = EvidenceStore(data_dir / 'frames')
    rows = []
    with Repository(database) as repo:
        connection = sqlite3.connect(database)
        for (sweep_id,) in connection.execute('SELECT id FROM sweeps'):
            for frame in repo.list_for_sweep(Frame, sweep_id):
                try:
                    rows.append((frame.id, store.read(frame.evidence_key)))
                except (OSError, ValueError, KeyError):
                    continue
    random.Random(seed).shuffle(rows)
    return rows[:limit]


async def baseline(jpeg: bytes) -> tuple[BaselineCheck, float]:
    """Ask the frontier vision model the admission question.

    Args:
        jpeg: Frame bytes.

    Returns:
        Its answer and the wall-clock latency in milliseconds.
    """
    import base64
    url = 'data:image/jpeg;base64,' + base64.b64encode(jpeg).decode()
    started = time.monotonic()
    answer = await structured(BASELINE_INSTRUCTION,
                              [{'type': 'image_url', 'image_url': {'url': url}}],
                              BaselineCheck, model=VISION_MODEL)
    return answer, (time.monotonic() - started) * 1000


async def main() -> None:
    """Run both deciders over the sample and report the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=25)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--data', type=Path, default=Path('.claim-data'))
    parser.add_argument('--out', type=Path, default=Path('docs/drex-benchmark.json'))
    args = parser.parse_args()

    frames = saved_frames(args.data / 'claims.sqlite3', args.data, args.limit, args.seed)
    print(f'replaying {len(frames)} saved frames through Drex and {VISION_MODEL}\n')
    rows = []
    for index, (frame_id, jpeg) in enumerate(frames, 1):
        record = {'frame_id': frame_id}
        try:
            started = time.monotonic()
            drex = await asyncio.to_thread(check_frame, jpeg)
            record['drex'] = {'book_visible': drex.book_visible, 'spine_legible': drex.spine_legible,
                              'issue': drex.issue, 'latency_ms': drex.latency_ms,
                              'wall_ms': (time.monotonic() - started) * 1000,
                              'input_tokens': drex.input_tokens}
        except Exception as error:  # noqa: BLE001 - a failed call is a result, not a crash
            record['drex'] = {'error': type(error).__name__}
        try:
            answer, latency = await baseline(jpeg)
            record['baseline'] = {'book_visible': answer.book_visible,
                                  'spine_legible': answer.spine_legible, 'wall_ms': latency}
        except Exception as error:  # noqa: BLE001
            record['baseline'] = {'error': type(error).__name__}
        rows.append(record)
        print(f'  {index:>3}/{len(frames)}  drex={record["drex"].get("book_visible", "err")!s:<8}'
              f'baseline={record["baseline"].get("book_visible", "err")}')

    paired = [row for row in rows if 'error' not in row['drex'] and 'error' not in row['baseline']]
    threshold = CONFIG.drex.book_visible_min
    agree = [row for row in paired
             if (row['drex']['book_visible'] >= threshold) == (row['baseline']['book_visible'] >= threshold)]
    drex_ms = [row['drex']['wall_ms'] for row in paired]
    base_ms = [row['baseline']['wall_ms'] for row in paired]
    summary = {
        'frames_compared': len(paired),
        'agreement': round(len(agree) / len(paired), 3) if paired else None,
        'drex_mean_ms': round(sum(drex_ms) / len(drex_ms)) if drex_ms else None,
        'baseline_mean_ms': round(sum(base_ms) / len(base_ms)) if base_ms else None,
        'speedup': round(sum(base_ms) / sum(drex_ms), 2) if drex_ms and sum(drex_ms) else None,
        'drex_input_tokens': sum(row['drex']['input_tokens'] for row in paired),
        'drex_model': CONFIG.drex.model, 'baseline_model': VISION_MODEL,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({'summary': summary, 'rows': rows}, indent=2))
    print('\n' + json.dumps(summary, indent=2))
    print(f'\nwritten to {args.out}')


if __name__ == '__main__':
    asyncio.run(main())
