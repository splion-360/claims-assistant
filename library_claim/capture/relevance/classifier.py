"""Score book relevance locally using quantized CLIP image and text embeddings."""
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
        vector = self._vision.run(['image_embeds'], {'pixel_values': prepare_image(image)})[0]
        vector /= np.linalg.norm(vector, axis=1, keepdims=True)
        similarities = (vector @ self._text.T)[0]
        scores, offset = {}, 0
        for label, prompts in PROMPTS.items():
            scores[label] = float(np.clip(similarities[offset:offset + len(prompts)].max(), -1, 1))
            offset += len(prompts)
        return decide(scores, (perf_counter() - started) * 1000)


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
