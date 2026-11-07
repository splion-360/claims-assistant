"""Compare full-frame, center, and overlapping-crop book relevance offline."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from tokenizers import Tokenizer

from library_claim.config import CONFIG, CONFIG_DIGEST
from .audit import center_crop, session, summarize, unit
from .classifier import BookRelevanceClassifier, MODEL_DIR, PROMPTS, decide, prepare_image, crop_boxes, grouped_scores, select_crop


def run(manifest: Path, full_precision_text: Path, output: Path) -> dict:
    """Evaluate a labelled capture set using unchanged prompts and thresholds.

    Args:
        manifest: JSON containing image paths, boolean labels, and optional groups.
        full_precision_text: Pinned FP32 CLIP text encoder file.
        output: JSON destination for detailed, reproducible comparison results.

    Returns:
        Report with variant totals, group counts, per-crop book scores, and runtime settings.

    Raises:
        ValueError: Labels are invalid, either class is missing, or images cannot decode.
    """
    source = json.loads(manifest.read_text())
    rows = source['images']
    if any(type(row['book_present']) is not bool for row in rows):
        raise ValueError('book_present must be a JSON boolean')
    labels = np.array([row['book_present'] for row in rows], dtype=bool)
    if not labels.any() or labels.all():
        raise ValueError('Include both book and no-book images')
    started = perf_counter()
    live = BookRelevanceClassifier()
    live_load_ms = (perf_counter() - started) * 1000
    started = perf_counter()
    tokenizer = Tokenizer.from_file(str(MODEL_DIR / Path(CONFIG.relevance.artifacts.tokenizer.remote).name))
    tokenizer.enable_truncation(max_length=CONFIG.relevance.token_length)
    tokenizer.enable_padding(length=CONFIG.relevance.token_length, pad_id=CONFIG.relevance.pad_id,
                             pad_token=CONFIG.relevance.pad_token)
    prompts = [prompt for group in PROMPTS.values() for prompt in group]
    ids = np.array([encoding.ids for encoding in tokenizer.encode_batch(prompts)], dtype=np.int64)
    text_model = session(full_precision_text)
    text_vectors = unit(text_model.run(['text_embeds'], {'input_ids': ids})[0])
    text_load_ms = (perf_counter() - started) * 1000
    del text_model
    vision_path = MODEL_DIR / Path(CONFIG.relevance.artifacts.vision.remote).name
    vision = session(vision_path)
    variants = {}
    for variant in ('live', 'fp32_text_padding', 'fp32_text_center', 'fp32_text_overlap'):
        results, times, scores = [], [], []
        for row in rows:
            image = cv2.imread(row['path'])
            if image is None:
                raise ValueError(f'Cannot decode {row["path"]}')
            started = perf_counter()
            crop_results = []
            if variant == 'live':
                result = live.assess(image)
                pair = np.array([result.scores[label] for label in PROMPTS])
                selected = None
            else:
                if variant == 'fp32_text_overlap':
                    boxes = crop_boxes(image.shape[1], image.shape[0], CONFIG.experiments.edge_crop_count)
                    tensors = [center_crop(image[top:bottom, left:right]) for left, top, right, bottom in boxes]
                else:
                    boxes = [(0, 0, image.shape[1], image.shape[0])]
                    tensors = [prepare_image(image) if variant.endswith('padding') else center_crop(image)]
                # Run each crop separately to avoid batch-dependent INT8 scale changes.
                vectors = np.concatenate([unit(vision.run(['image_embeds'], {'pixel_values': tensor})[0]) for tensor in tensors])
                pairs = grouped_scores(vectors @ text_vectors.T)
                selected = select_crop(pairs)
                pair = pairs[selected]
                crop_results = [{'box': box, 'scores': dict(zip(PROMPTS, map(float, values)))} for box, values in zip(boxes, pairs)]
                result = decide(dict(zip(PROMPTS, map(float, pair))), 0)
            elapsed = (perf_counter() - started) * 1000
            times.append(elapsed)
            scores.append(pair)
            results.append({**row, 'label': result.label, 'scores': dict(zip(PROMPTS, map(float, pair))),
                            'elapsed_ms': elapsed, 'selected_crop': selected, 'crops': crop_results})
        summary = summarize(np.array(scores), labels, times)
        groups = {}
        for group in sorted({row.get('group', 'unspecified') for row in rows}):
            members = [row for row in results if row.get('group', 'unspecified') == group]
            groups[group] = {'n': len(members), 'positive_n': sum(row['book_present'] for row in members),
                             'books_on_positives': sum(row['book_present'] and row['label'] == 'books' for row in members),
                             'books_on_negatives': sum(not row['book_present'] and row['label'] == 'books' for row in members),
                             'uncertain': sum(row['label'] == 'uncertain' for row in members)}
        variants[variant] = {'summary': summary, 'groups': groups, 'frames': results}
        print(variant, json.dumps(summary), flush=True)
    report = {'manifest': source, 'config': CONFIG.model_dump(mode='json'), 'config_digest': CONFIG_DIGEST,
              'live_load_ms': live_load_ms, 'fp32_text_load_ms': text_load_ms,
              'fp32_text_sha256': hashlib.sha256(full_precision_text.read_bytes()).hexdigest(),
              'vision_sha256': hashlib.sha256(vision_path.read_bytes()).hexdigest(), 'variants': variants}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    """Run a local edge-coverage experiment on an explicitly labelled manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--full-precision-text', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.manifest, args.full_precision_text, args.output)


if __name__ == '__main__':
    main()
