"""Set up pinned local weights or evaluate relevance on saved camera frames."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import httpx
import numpy as np

from .classifier import BookRelevanceClassifier, MODEL_DIR, MODEL_ID, REVISION

from library_claim.config import CONFIG

ARTIFACTS = {value['remote']: value['sha256'] for value in CONFIG.relevance.artifacts.model_dump().values()}


def setup(directory: Path) -> None:
    """Download and verify fixed model artifacts without uploading images.

    Args:
        directory: Local destination for model files.

    Raises:
        ValueError: An artifact fails its expected SHA-256 checksum.
        httpx.HTTPError: The model host cannot supply an artifact.
        OSError: Model files cannot be written.
    """
    directory.mkdir(parents=True, exist_ok=True)
    with httpx.Client(follow_redirects=True, timeout=CONFIG.relevance.download_timeout_s) as client:
        for remote, expected in ARTIFACTS.items():
            target = directory / Path(remote).name
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == expected:
                print(f'Verified {target.name}')
                continue
            temporary = target.with_suffix('.download')
            try:
                digest = hashlib.sha256()
                with client.stream('GET', f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{remote}') as response:
                    response.raise_for_status()
                    with temporary.open('wb') as handle:
                        for chunk in response.iter_bytes():
                            handle.write(chunk)
                            digest.update(chunk)
                if digest.hexdigest() != expected:
                    raise ValueError(f'Checksum failed for {remote}')
                temporary.replace(target)
                print(f'Downloaded {target.name}')
            finally:
                temporary.unlink(missing_ok=True)


def evaluate(directory: Path, images: Path, output: Path) -> None:
    """Write per-image local predictions and latency statistics for inspection.

    Args:
        directory: Verified local model directory.
        images: Directory containing JPEG or PNG images.
        output: JSON report destination; no source images are copied.

    Raises:
        ValueError: There are no decodable evaluation images.
    """
    started = perf_counter()
    classifier = BookRelevanceClassifier(directory)
    load_ms = (perf_counter() - started) * 1000
    results = []
    for path in sorted(images.iterdir()):
        if path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        results.append({'path': str(path), **classifier.assess(image).model_dump(mode='json')})
    if not results:
        raise ValueError('No decodable evaluation images')
    timings = [result['elapsed_ms'] for result in results]
    summary = {'frames': len(results), 'labels': dict(Counter(r['label'] for r in results)),
               'load_ms': load_ms, 'p50_ms': float(np.percentile(timings, 50)),
               'p95_ms': float(np.percentile(timings, 95)),
               'accuracy': None, 'note': 'Unlabelled sample: predictions and latency do not establish accuracy.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'summary': summary, 'results': results}, indent=2))
    print(json.dumps(summary, indent=2))
    print(f'Report: {output}')


def main() -> None:
    """Dispatch explicit setup or offline evaluation from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=MODEL_DIR)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('setup')
    assessment = commands.add_parser('evaluate')
    assessment.add_argument('images', type=Path)
    assessment.add_argument('--output', type=Path, default=Path('.claim-data/relevance-evaluation.json'))
    args = parser.parse_args()
    if args.command == 'setup':
        setup(args.model_dir)
    else:
        evaluate(args.model_dir, args.images, args.output)


if __name__ == '__main__':
    main()
