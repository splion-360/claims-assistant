"""Score overlapping camera crops with local CLIP image and text embeddings."""
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

from library_claim.config import CONFIG, CONFIG_DIGEST, ROOT

SETTINGS = CONFIG.relevance
MODEL_ID = SETTINGS.model_id
REVISION = SETTINGS.revision
MODEL_DIR = ROOT / SETTINGS.model_dir
PROMPTS = SETTINGS.prompts.model_dump()
MIN_SIMILARITY = SETTINGS.min_similarity
MIN_MARGIN = SETTINGS.min_margin
METHOD = SETTINGS.method
_SHARED = None
_LOAD_LOCK = Lock()


class RelevanceAssessment(BaseModel):
    """Report relative cosine scores and an explicitly provisional decision."""

    model_config = ConfigDict(extra='forbid', frozen=True)
    label: Literal['books', 'empty_background', 'uncertain']
    status: Literal['assessed', 'quality_blocked', 'unavailable'] = 'assessed'
    scores: dict[str, float] = Field(default_factory=dict)
    margin: float | None = None
    reason: str
    elapsed_ms: float = 0
    model: str = MODEL_ID
    revision: str = REVISION
    method: str = METHOD
    config_digest: str = CONFIG_DIGEST
    min_similarity: float = MIN_SIMILARITY
    min_margin: float = MIN_MARGIN
    crop_boxes: list[tuple[int, int, int, int]] = Field(default_factory=list)
    crop_scores: list[dict[str, float]] = Field(default_factory=list)
    selected_crop: int | None = None


def decide(scores: dict[str, float], elapsed_ms: float) -> RelevanceAssessment:
    """Abstain when cosine similarities are weak or too close to distinguish.

    Args:
        scores: Maximum prompt cosine similarity for each configured class.
        elapsed_ms: Preprocessing and inference duration.

    Returns:
        Provisional label with raw scores, not probability estimates.

    Raises:
        ValueError: Scores are missing, nonfinite, or outside cosine bounds.
    """
    if set(scores) != set(PROMPTS) or any(not np.isfinite(v) or not -1 <= v <= 1 for v in scores.values()):
        raise ValueError('Expected one finite cosine score per relevance class')
    ranked = sorted(scores, key=scores.get, reverse=True)
    top = ranked[0]
    margin = scores[top] - scores[ranked[1]]
    reason = 'separated_match'
    if scores[top] < MIN_SIMILARITY:
        reason = 'weak_match'
    elif margin < MIN_MARGIN:
        reason = 'ambiguous_match'
    return RelevanceAssessment(label=top if reason == 'separated_match' else 'uncertain',
                               scores=scores, margin=margin, reason=reason, elapsed_ms=elapsed_ms)


def prepare_image(bgr: np.ndarray) -> np.ndarray:
    """Preserve the full camera view with square padding and CLIP normalization.

    Args:
        bgr: Nonempty uint8 OpenCV image with three BGR channels.

    Returns:
        Float32 RGB tensor at the configured square model resolution.

    Raises:
        ValueError: The input does not satisfy the decoded-image contract.
    """
    if bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3 or min(bgr.shape[:2]) < 1:
        raise ValueError('Expected nonempty uint8 BGR image')
    rgb = Image.fromarray(bgr[:, :, ::-1])
    size = SETTINGS.image_size
    rgb.thumbnail((size, size), Image.Resampling.BICUBIC)
    # Center cropping can discard books near the camera edges; keep the full view.
    square = Image.new('RGB', (size, size), SETTINGS.padding_rgb)
    square.paste(rgb, ((size - rgb.width) // 2, (size - rgb.height) // 2))
    pixels = np.asarray(square, dtype=np.float32) / 255
    mean = np.array(SETTINGS.image_mean, dtype=np.float32)
    std = np.array(SETTINGS.image_std, dtype=np.float32)
    return np.ascontiguousarray(((pixels - mean) / std).transpose(2, 0, 1)[None])


def center_crop(image: np.ndarray) -> np.ndarray:
    """Apply CLIP's reference shortest-edge resize and center crop.

    Args:
        image: Original uint8 BGR camera frame.

    Returns:
        Normalized RGB model tensor using the configured CLIP normalization.
    """
    size = CONFIG.relevance.image_size
    rgb = Image.fromarray(image[:, :, ::-1])
    width, height = rgb.size
    scale = size / min(width, height)
    rgb = rgb.resize((int(width * scale), int(height * scale)), Image.Resampling.BICUBIC)
    left, top = (rgb.width - size) // 2, (rgb.height - size) // 2
    rgb = rgb.crop((left, top, left + size, top + size))
    pixels = np.asarray(rgb, dtype=np.float32) / 255
    pixels = (pixels - np.float32(CONFIG.relevance.image_mean)) / np.float32(CONFIG.relevance.image_std)
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None])


def crop_boxes(width: int, height: int, count: int) -> list[tuple[int, int, int, int]]:
    """Cover the long image axis with evenly spaced overlapping square views.

    Args:
        width: Original image width.
        height: Original image height.
        count: Requested number of crops, including both image edges.

    Returns:
        Unique left, top, right, bottom pixel boxes covering the full image.

    Raises:
        ValueError: Dimensions or count are invalid, or crops would leave gaps.
    """
    if min(width, height) < 1 or count < 2:
        raise ValueError('Use positive dimensions and at least two crops')
    side = min(width, height)
    offsets = sorted(set(round(value) for value in np.linspace(0, max(width, height) - side, count)))
    if any(b - a > side for a, b in zip(offsets, offsets[1:])):
        raise ValueError('Crop count is too small to cover this aspect ratio')
    return [(x, 0, x + side, side) if width >= height else (0, x, side, x + side) for x in offsets]


def select_crop(scores: np.ndarray) -> int:
    """Choose book evidence without comparing scores from unrelated crops.

    Args:
        scores: Per-crop book and background score pairs.

    Returns:
        Index of the strongest qualifying book margin, or strongest margin if none qualify.
    """
    eligible = [i for i, row in enumerate(scores) if decide(dict(zip(PROMPTS, row)), 0).label == 'books']
    candidates = eligible or list(range(len(scores)))
    return max(candidates, key=lambda i: scores[i, 0] - scores[i, 1])


def grouped_scores(similarities: np.ndarray) -> np.ndarray:
    """Reduce per-prompt similarities using the live classifier's prompt grouping.

    Args:
        similarities: One row per crop and one column per configured prompt.

    Returns:
        Book and background scores for each crop.
    """
    offset, columns = 0, []
    for prompts in PROMPTS.values():
        columns.append(similarities[:, offset:offset + len(prompts)].max(axis=1))
        offset += len(prompts)
    return np.stack(columns, axis=1)


class BookRelevanceClassifier:
    """Reuse a CPU image encoder and cached text vectors without network access."""

    def __init__(self, directory: Path = MODEL_DIR):
        """Load local weights and encode the fixed classification prompts once.

        Args:
            directory: Directory populated by the explicit setup command.

        Raises:
            FileNotFoundError: Setup has not downloaded the required artifacts.
            RuntimeError: A local model cannot be initialized.
        """
        import onnxruntime as ort
        from tokenizers import Tokenizer

        files = {key: Path(value['remote']).name for key, value in SETTINGS.artifacts.model_dump().items()}
        for name in files.values():
            if not (directory / name).is_file():
                raise FileNotFoundError('Run python -m library_claim.capture.relevance setup')
        options = ort.SessionOptions()
        options.intra_op_num_threads = SETTINGS.intra_op_threads
        options.inter_op_num_threads = SETTINGS.inter_op_threads
        self._vision = ort.InferenceSession(str(directory / files['vision']),
                                           sess_options=options, providers=['CPUExecutionProvider'])
        text = ort.InferenceSession(str(directory / files['text']),
                                    sess_options=options, providers=['CPUExecutionProvider'])
        tokenizer = Tokenizer.from_file(str(directory / files['tokenizer']))
        tokenizer.enable_truncation(max_length=SETTINGS.token_length)
        tokenizer.enable_padding(length=SETTINGS.token_length, pad_id=SETTINGS.pad_id, pad_token=SETTINGS.pad_token)
        prompts = [prompt for group in PROMPTS.values() for prompt in group]
        tokens = np.array([e.ids for e in tokenizer.encode_batch(prompts)], dtype=np.int64)
        vectors = text.run(['text_embeds'], {'input_ids': tokens})[0]
        self._text = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def assess(self, image: np.ndarray) -> RelevanceAssessment:
        """Compare one image with the cached prompt embeddings.

        Args:
            image: Decoded uint8 BGR camera frame.

        Returns:
            Label, cosine scores, margin, provenance, and warm inference timing.
        """
        started = perf_counter()
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
            raise ValueError('Expected nonempty uint8 BGR image')
        boxes = crop_boxes(image.shape[1], image.shape[0], SETTINGS.crop_count)
        vectors = []
        # Sequential inference preserves the INT8 scaling used in the comparison.
        for left, top, right, bottom in boxes:
            tensor = center_crop(image[top:bottom, left:right])
            vector = self._vision.run(['image_embeds'], {'pixel_values': tensor})[0]
            vectors.append(vector / np.linalg.norm(vector, axis=1, keepdims=True))
        pairs = np.clip(grouped_scores(np.concatenate(vectors) @ self._text.T), -1, 1)
        selected = select_crop(pairs)
        scores = [dict(zip(PROMPTS, map(float, pair))) for pair in pairs]
        result = decide(scores[selected], (perf_counter() - started) * 1000)
        return result.model_copy(update={'crop_boxes': boxes, 'crop_scores': scores,
                                         'selected_crop': selected})



def shared_classifier() -> BookRelevanceClassifier:
    """Lazily load one model per process under a lock.

    Returns:
        Shared classifier whose immutable prompt vectors are reused by sessions.
    """
    global _SHARED
    with _LOAD_LOCK:
        if _SHARED is None:
            _SHARED = BookRelevanceClassifier()
    return _SHARED
