"""Compare relevance preprocessing, prompt wording, and encoder precision offline."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from library_claim.config import CONFIG, CONFIG_DIGEST
from .classifier import center_crop, MODEL_DIR, PROMPTS, prepare_image, decide


def session(path: Path) -> ort.InferenceSession:
    """Load a local encoder with the live CPU thread budget.

    Args:
        path: ONNX model file to inspect.

    Returns:
        CPU-only inference session.
    """
    options = ort.SessionOptions()
    options.intra_op_num_threads = CONFIG.relevance.intra_op_threads
    options.inter_op_num_threads = CONFIG.relevance.inter_op_threads
    return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])


def unit(vectors: np.ndarray) -> np.ndarray:
    """Normalize encoder outputs for cosine comparison.

    Args:
        vectors: Matrix of image or text embeddings.

    Returns:
        Unit-length row vectors.
    """
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def summarize(scores: np.ndarray, labels: np.ndarray, timings: list[float]) -> dict:
    """Describe score separation and routing outcomes on the labelled sample.

    Args:
        scores: Per-frame book and background scores, in that order.
        labels: Boolean book-presence annotations.
        timings: Per-frame preprocessing and image-encoder durations.

    Returns:
        Descriptive statistics and current-threshold decisions; no fitted threshold.
    """
    positive, negative = scores[labels, 0], scores[~labels, 0]
    decisions = [decide(dict(zip(PROMPTS, row)), 0).label for row in scores]
    predicted = np.array([value == 'books' for value in decisions])
    pairwise = positive[:, None] - negative[None, :]
    return {
        'positive_n': len(positive), 'negative_n': len(negative),
        'book_mean': float(positive.mean()), 'no_book_mean': float(negative.mean()),
        'mean_gap': float(positive.mean() - negative.mean()),
        'book_score_ranges_overlap': bool(positive.min() <= negative.max() and negative.min() <= positive.max()),
        'book_range': [float(positive.min()), float(positive.max())],
        'no_book_range': [float(negative.min()), float(negative.max())],
        'book_score_auc': float(((pairwise > 0) + .5 * (pairwise == 0)).mean()),
        'book_predictions_on_books': int(predicted[labels].sum()),
        'book_predictions_on_negatives': int(predicted[~labels].sum()),
        'uncertain_count': decisions.count('uncertain'),
        'image_p50_ms': float(np.percentile(timings, 50)),
        'image_p95_ms': float(np.percentile(timings, 95)),
    }


def audit(manifest: Path, full_precision_dir: Path, output: Path) -> dict:
    """Run controlled comparisons while leaving live settings and images untouched.

    Args:
        manifest: JSON with images, book_present labels, and optional candidate prompts.
        full_precision_dir: Pinned vision_model.onnx and text_model.onnx directory.
        output: Destination for per-image results, prompt diagnostics, and provenance.

    Returns:
        Experiment report with statistics for each precision/preprocessing variant.

    Raises:
        ValueError: Labels omit either class or an image cannot be decoded.
    """
    source = json.loads(manifest.read_text())
    records = source['images']
    if any(type(row['book_present']) is not bool for row in records):
        raise ValueError('book_present must be a JSON boolean')
    labels = np.array([row['book_present'] for row in records], dtype=bool)
    if not labels.any() or labels.all():
        raise ValueError('Include both book and no-book images')
    base_prompts = [prompt for group in PROMPTS.values() for prompt in group]
    prompts = base_prompts + source.get('candidate_book_prompts', [])
    tokenizer = Tokenizer.from_file(str(MODEL_DIR / Path(CONFIG.relevance.artifacts.tokenizer.remote).name))
    tokenizer.enable_truncation(max_length=CONFIG.relevance.token_length)
    tokenizer.enable_padding(length=CONFIG.relevance.token_length, pad_id=CONFIG.relevance.pad_id,
                             pad_token=CONFIG.relevance.pad_token)
    tokens = np.array([item.ids for item in tokenizer.encode_batch(prompts)], dtype=np.int64)
    models = {
        'int8': (MODEL_DIR / Path(CONFIG.relevance.artifacts.vision.remote).name,
                 MODEL_DIR / 'text_model_quantized.onnx'),
        'fp32': (full_precision_dir / 'vision_model.onnx', full_precision_dir / 'text_model.onnx'),
    }
    text_vectors = {}
    for precision, (_, text_path) in models.items():
        text_model = session(text_path)
        # Quantization can depend on batch statistics; keep the live prompt batch intact.
        base = text_model.run(['text_embeds'], {'input_ids': tokens[:len(base_prompts)]})[0]
        if len(prompts) > len(base_prompts):
            extra = text_model.run(['text_embeds'], {'input_ids': tokens[len(base_prompts):]})[0]
            base = np.concatenate((base, extra))
        text_vectors[precision] = unit(base)
        del text_model
    # Cache only small vectors so comparisons reuse identical encoder outputs.
    variants = {}
    prompt_results = {}
    for vision_precision, (vision_path, _) in models.items():
        vision = session(vision_path)
        for preprocessing, prepare in [('padding', prepare_image), ('center_crop', center_crop)]:
            vectors, timings = [], []
            for record in records:
                image = cv2.imread(record['path'])
                if image is None:
                    raise ValueError(f'Cannot decode {record["path"]}')
                started = perf_counter()
                vectors.append(unit(vision.run(['image_embeds'], {'pixel_values': prepare(image)})[0])[0])
                timings.append((perf_counter() - started) * 1000)
            for text_precision, text in text_vectors.items():
                name = f'vision_{vision_precision}_text_{text_precision}_{preprocessing}'
                similarities = np.array(vectors) @ text.T
                offset, grouped = 0, []
                for group in PROMPTS.values():
                    grouped.append(similarities[:, offset:offset + len(group)].max(axis=1))
                    offset += len(group)
                scores = np.stack(grouped, axis=1)
                variants[name] = {'summary': summarize(scores, labels, timings), 'frames': [
                    {**record, 'scores': dict(zip(PROMPTS, map(float, row))),
                     'prompt_scores': dict(zip(prompts, map(float, prompt_row)))}
                    for record, row, prompt_row in zip(records, scores, similarities)]}
                prompt_results[name] = {prompt: {
                    'book_mean': float(similarities[labels, i].mean()),
                    'no_book_mean': float(similarities[~labels, i].mean()),
                    'mean_gap': float(similarities[labels, i].mean() - similarities[~labels, i].mean())}
                    for i, prompt in enumerate(prompts)}
                print(name, json.dumps(variants[name]['summary']), flush=True)
        del vision
    model_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                    for paths in models.values() for path in paths}
    report = {'config_digest': CONFIG_DIGEST, 'config': CONFIG.model_dump(mode='json'),
              'manifest': source, 'model_sha256': model_hashes, 'variants': variants,
              'prompt_diagnostics': prompt_results}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    """Run the offline audit with explicit labelled data and comparison weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--full-precision-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.manifest, args.full_precision_dir, args.output)


if __name__ == '__main__':
    main()
