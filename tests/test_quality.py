"""Verify pretrained BRISQUE assessment and interpretable OpenCV quality signals."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from library_claim.capture.quality import QualityAssessor
from library_claim.domain.models import Book, BookObservation, Box, Frame, Quality, Sweep
from library_claim.storage.repository import Repository
from datetime import datetime, timezone


def sample_image():
    """Create a deterministic textured image with sharp printed text.

    Returns:
        A uint8 BGR image for quality plumbing tests.
    """
    gray = np.random.default_rng(42).integers(80, 180, (240, 320), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.putText(image, 'BOOK TITLE', (15, 120), cv2.FONT_HERSHEY_SIMPLEX,
                .8, (10, 10, 10), 2, cv2.LINE_AA)
    return image


class QualityTests(unittest.TestCase):
    """Exercise real pretrained inference and defensive input handling."""

    def setUp(self):
        """Create a reusable model and deterministic image for each test."""
        self.assessor = QualityAssessor()
        self.image = sample_image()

    def test_real_brisque_and_blur_signal(self):
        """Return finite model scores and lower sharpness after artificial blur."""
        clear = self.assessor.assess(self.image)
        blurred = self.assessor.assess(cv2.GaussianBlur(self.image, (15, 15), 4))
        self.assertIsNotNone(clear.signals.brisque)
        self.assertIsNotNone(blurred.signals.brisque)
        self.assertGreater(clear.signals.sharpness, blurred.signals.sharpness)
        self.assertGreaterEqual(clear.brisque_quality_score, 0)
        self.assertLessEqual(clear.brisque_quality_score, 1)
        self.assertGreater(clear.elapsed_ms, 0)
        self.assertIsNone(clear.signals.glare_fraction)
        self.assertIsNone(clear.signals.temporal_change)

    def test_extreme_exposure_and_constant_images(self):
        """Measure exposure extremes without inventing a score for constant images."""
        black = self.assessor.assess(np.zeros((64, 64), dtype=np.uint8))
        white = self.assessor.assess(np.full((64, 64), 255, dtype=np.uint8))
        self.assertEqual(black.signals.dark_fraction, 1)
        self.assertEqual(white.signals.bright_fraction, 1)
        self.assertEqual(black.signals.sharpness, 0)
        self.assertIsNone(black.signals.brisque)
        self.assertIsNone(white.brisque_quality_score)
        self.assertIn('brisque_undefined_for_constant_image', black.unavailable_reasons)

    def test_temporal_change_distinguishes_repeated_from_changed_pixels(self):
        """Measure image change without claiming that it is physical motion."""
        same = self.assessor.assess(self.image, previous=self.image.copy())
        changed = self.assessor.assess(255 - self.image, previous=self.image)
        self.assertEqual(same.signals.temporal_change, 0)
        self.assertGreater(changed.signals.temporal_change, 0)

    def test_region_and_resizing_are_explicit(self):
        """Keep preprocessing dimensions and region references in the result."""
        large = cv2.resize(self.image, (1280, 960))
        result = self.assessor.assess(large)
        self.assertEqual((result.assessed_width_px, result.assessed_height_px), (640, 480))
        region = Box(x_min=.25, y_min=.25, x_max=.75, y_max=.75)
        crop = self.assessor.assess(self.image, region=region)
        self.assertEqual((crop.assessed_width_px, crop.assessed_height_px), (160, 120))
        self.assertEqual(crop.region, region)
        np.testing.assert_array_equal(self.image, sample_image())

    def test_invalid_inputs_and_incompatible_previous_frame(self):
        """Reject unsupported formats, tiny crops, and mismatched frame layouts."""
        for invalid in (None, np.zeros((32, 32), dtype=float),
                        np.zeros((32, 32, 4), dtype=np.uint8),
                        np.zeros((1, 32), dtype=np.uint8)):
            with self.subTest(shape=getattr(invalid, 'shape', None)), self.assertRaises(ValueError):
                self.assessor.assess(invalid)
        with self.assertRaises(ValueError):
            self.assessor.assess(self.image, previous=self.image[:100])
        with self.assertRaises(ValueError):
            self.assessor.assess(self.image, region=Box(x_min=0, y_min=0, x_max=.01, y_max=.01))
        with self.assertRaises(ValueError):
            QualityAssessor(max_dimension=0)

    def test_unavailable_model_score_preserves_other_measurements(self):
        """Expose scoring failures while retaining measured quality diagnostics."""
        for score in (float('nan'), float('inf')):
            with patch.object(self.assessor, '_brisque') as model:
                model.compute.return_value = (score, 0, 0, 0)
                result = self.assessor.assess(self.image)
            self.assertIsNone(result.signals.brisque)
            self.assertIsNone(result.brisque_quality_score)
            self.assertGreater(result.signals.sharpness, 0)
            self.assertIn('brisque_nonfinite_result', result.unavailable_reasons)
        with patch.object(self.assessor, '_brisque') as model:
            model.compute.side_effect = cv2.error('inference failure')
            result = self.assessor.assess(self.image)
        self.assertIn('brisque_computation_failed', result.unavailable_reasons)

    def test_raw_out_of_range_score_is_retained(self):
        """Bound only the ranking score while preserving raw model outputs."""
        with patch.object(self.assessor, '_brisque') as model:
            model.compute.return_value = (-5.0, 0, 0, 0)
            result = self.assessor.assess(self.image)
        self.assertEqual(result.signals.brisque, -5)
        self.assertEqual(result.brisque_quality_score, 1)

    def test_quality_persists_and_old_payloads_remain_readable(self):
        """Persist new diagnostics and load legacy quality objects with defaults."""
        self.assertIsNone(Quality.model_validate({'sharpness': 2}).bright_fraction)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'claims.sqlite3'
            with Repository(path) as repo:
                sweep = repo.add(Sweep(captured_at=datetime.now(timezone.utc)))
                book = repo.add(Book(sweep_id=sweep.id))
                frame = repo.add(Frame(sweep_id=sweep.id, captured_at=sweep.captured_at,
                                       evidence_key='a' * 64 + '.jpg', width_px=320, height_px=240))
                result = self.assessor.assess(self.image)
                observation = repo.add(BookObservation(
                    sweep_id=sweep.id, book_id=book.id, frame_id=frame.id,
                    bbox=Box(x_min=0, y_min=0, x_max=1, y_max=1), quality=result.signals))
            with Repository(path) as repo:
                self.assertEqual(repo.get(BookObservation, observation.id).quality, result.signals)
