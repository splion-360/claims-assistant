"""Apply conservative quality checks before local relevance diagnostics."""
from time import perf_counter

import numpy as np

from library_claim.capture.quality import QualityAssessment
from library_claim.capture.relevance.classifier import RelevanceAssessment, shared_classifier


class CameraRelevance:
    """Keep model failures explicit while preserving the running capture pipeline."""

    def __init__(self):
        """Initialize activation-local failure state without loading model weights."""
        self._failure = None

    def assess(self, image: np.ndarray, quality: QualityAssessment) -> RelevanceAssessment:
        """Classify a candidate frame unless its measured quality is unusable.

        Args:
            image: Decoded BGR image of the current candidate.
            quality: Measurements from this exact frame.

        Returns:
            Relevance diagnostics, or an explicit blocked/unavailable result.
        """
        signals = quality.signals
        # These broad thresholds reject unusable captures, not subtle OCR defects.
        if (signals.dark_fraction or 0) > .95 or (signals.bright_fraction or 0) > .95:
            return RelevanceAssessment(label='uncertain', status='quality_blocked', reason='extreme_exposure')
        if signals.sharpness is not None and signals.sharpness < 10:
            return RelevanceAssessment(label='uncertain', status='quality_blocked', reason='insufficient_texture_or_sharpness')
        if self._failure is None:
            try:
                started = perf_counter()
                result = shared_classifier().assess(image)
                return result.model_copy(update={'elapsed_ms': (perf_counter() - started) * 1000})
            except Exception as error:
                # Avoid repeated loads on every frame when setup or inference fails.
                self._failure = type(error).__name__
        return RelevanceAssessment(label='uncertain', status='unavailable',
                                   reason=f'{self._failure}: check model setup and restart camera')
