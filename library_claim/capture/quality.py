"""Measure image quality using OpenCV diagnostics and pretrained BRISQUE."""
import math
from time import perf_counter
from typing import Annotated

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from library_claim.domain.models import Box, Quality, Score
from library_claim.config import CONFIG, CONFIG_DIGEST, ROOT

SETTINGS = CONFIG.quality


class QualityAssessment(BaseModel):
    """Describe measured quality, processing provenance, and unavailable signals."""

    model_config = ConfigDict(extra='forbid', frozen=True)
    signals: Quality
    brisque_quality_score: Score | None = None
    assessed_width_px: int
    assessed_height_px: int
    region: Box | None = None
    unavailable_reasons: list[str] = Field(default_factory=list)
    elapsed_ms: Annotated[float, Field(ge=0, allow_inf_nan=False)]
    dark_cutoff: int
    bright_cutoff: int
    opencv_version: str
    method: str = SETTINGS.method
    config_digest: str = CONFIG_DIGEST


class QualityAssessor:
    """Reuse a pretrained BRISQUE model for synchronous image assessment."""

    def __init__(self, *, max_dimension: int = SETTINGS.max_dimension, dark_cutoff: int = SETTINGS.dark_cutoff,
                 bright_cutoff: int = SETTINGS.bright_cutoff):
        """Load the bundled model and configure reproducible preprocessing.

        Args:
            max_dimension: Maximum assessed image edge length, in pixels.
            dark_cutoff: Inclusive grayscale cutoff for near-black pixels.
            bright_cutoff: Inclusive grayscale cutoff for near-white pixels.

        Raises:
            ValueError: Size or intensity limits are invalid.
            RuntimeError: The pretrained model cannot be loaded.
        """
        if not isinstance(max_dimension, int) or max_dimension < 32:
            raise ValueError('max_dimension must be an integer of at least 32')
        if not 0 <= dark_cutoff < bright_cutoff <= 255:
            raise ValueError('Intensity cutoffs must satisfy 0 <= dark < bright <= 255')
        self.max_dimension = max_dimension
        self.dark_cutoff = dark_cutoff
        self.bright_cutoff = bright_cutoff
        try:
            self._brisque = cv2.quality.QualityBRISQUE_create(
                str(ROOT / SETTINGS.brisque_model),
                str(ROOT / SETTINGS.brisque_range),
            )
        except (cv2.error, AttributeError) as error:
            raise RuntimeError('Could not load bundled BRISQUE model; install the contrib dependencies') from error

    @staticmethod
    def _validate(image: np.ndarray) -> None:
        """Validate an explicitly decoded grayscale or BGR image.

        Args:
            image: Uint8 array with two dimensions or three BGR channels.

        Raises:
            ValueError: The array type, channels, or dimensions are unsupported.
        """
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
            raise ValueError('Expected a uint8 image array')
        if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
            raise ValueError('Expected grayscale or three-channel BGR image')
        if min(image.shape[:2]) < 32:
            raise ValueError('Image dimensions must be at least 32 pixels')

    def _prepare(self, image: np.ndarray, region: Box | None) -> np.ndarray:
        """Crop and resize an image before measuring its grayscale signals.

        Args:
            image: Valid decoded image.
            region: Optional normalized region in the input frame.

        Returns:
            Grayscale uint8 image at the configured analysis resolution.

        Raises:
            ValueError: The selected region is too small for assessment.
        """
        self._validate(image)
        if region is not None:
            height, width = image.shape[:2]
            image = image[math.floor(region.y_min * height):math.ceil(region.y_max * height),
                          math.floor(region.x_min * width):math.ceil(region.x_max * width)]
        height, width = image.shape[:2]
        if min(height, width) < 32:
            raise ValueError('Selected region must be at least 32 pixels on each edge')
        # Never upscale a small crop: interpolation cannot create book detail.
        scale = min(1.0, self.max_dimension / max(height, width))
        if scale < 1:
            size = (round(width * scale), round(height * scale))
            if min(size) < 32:
                raise ValueError('Aspect ratio leaves insufficient pixels for assessment')
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    def assess(self, image: np.ndarray, *, previous: np.ndarray | None = None,
               region: Box | None = None) -> QualityAssessment:
        """Assess one frame or crop without deciding whether inference should run.

        Args:
            image: Decoded uint8 grayscale or BGR frame.
            previous: Optional preceding frame of the same shape and channel layout.
            region: Optional normalized book or shelf region; applies to both frames.

        Returns:
            Raw diagnostics, a BRISQUE-derived ranking score, and elapsed time.
            Lower raw BRISQUE is better; its clipped ranking score is higher-better.

        Raises:
            ValueError: Inputs or the selected region are invalid or incompatible.
        """
        start = perf_counter()
        gray = self._prepare(image, region)
        reasons = []
        temporal_change = None
        if previous is not None:
            self._validate(previous)
            if previous.shape != image.shape:
                raise ValueError('Previous frame must have the same shape as the current frame')
            old_gray = self._prepare(previous, region)
            # This is image change, not camera speed or an object association.
            temporal_change = float(cv2.absdiff(gray, old_gray).mean() / 255)
        else:
            reasons.append('temporal_change_requires_previous_frame')

        raw_score = None
        if float(gray.std()) < 1e-6:
            reasons.append('brisque_undefined_for_constant_image')
        else:
            try:
                score = float(self._brisque.compute(gray)[0])
                if math.isfinite(score):
                    raw_score = score
                else:
                    reasons.append('brisque_nonfinite_result')
            except cv2.error:
                reasons.append('brisque_computation_failed')

        signals = Quality(
            sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
            dark_fraction=float(np.mean(gray <= self.dark_cutoff)),
            bright_fraction=float(np.mean(gray >= self.bright_cutoff)),
            temporal_change=temporal_change,
            brisque=raw_score,
        )
        # Preserve raw scores: the bounded score is a ranking aid, not confidence.
        ranking = None if raw_score is None else float(np.clip(1 - raw_score / SETTINGS.brisque_ranking_scale, 0, 1))
        return QualityAssessment(
            signals=signals, brisque_quality_score=ranking,
            assessed_width_px=gray.shape[1], assessed_height_px=gray.shape[0],
            region=region, unavailable_reasons=reasons,
            dark_cutoff=self.dark_cutoff, bright_cutoff=self.bright_cutoff,
            opencv_version=cv2.__version__,
            elapsed_ms=(perf_counter() - start) * 1000,
        )
