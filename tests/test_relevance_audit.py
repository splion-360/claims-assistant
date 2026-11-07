"""Verify diagnostic separation metrics and reference image preprocessing."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from library_claim.capture.relevance.audit import audit, center_crop, summarize
from library_claim.capture.relevance.classifier import prepare_image
from library_claim.config import CONFIG


class RelevanceAuditTests(unittest.TestCase):
    """Guard against misleading evaluation statistics and silently coerced labels."""

    def test_overlap_and_auc_reflect_known_book_scores(self):
        """Measure tied rankings and positive detection separately from score means."""
        scores = np.array([[.3], [.2], [.2], [.25]])
        result = summarize(scores, np.array([True, True, False, False]), [1, 2, 3, 4])
        self.assertAlmostEqual(result['book_score_auc'], .625)
        self.assertTrue(result['book_score_ranges_overlap'])
        self.assertEqual(result['book_predictions_on_books'], 1)
        self.assertEqual(result['book_predictions_on_negatives'], 1)
        self.assertEqual(result['uncertain_count'], 2)

    def test_separated_scores_are_not_reported_as_overlapping(self):
        """Distinguish complete sample separation from noisy overlap."""
        scores = np.array([[.4], [.35], [.15], [.1]])
        result = summarize(scores, np.array([True, True, False, False]), [1, 1, 1, 1])
        self.assertEqual(result['book_score_auc'], 1)
        self.assertFalse(result['book_score_ranges_overlap'])
        self.assertEqual(result['book_predictions_on_books'], 2)

    def test_reference_crop_has_expected_shape_and_removes_edge_object(self):
        """Demonstrate the edge-coverage tradeoff rather than assuming crop equivalence."""
        size = CONFIG.relevance.image_size
        image = np.zeros((size, size * 2, 3), np.uint8)
        image[:, :size // 4] = (0, 0, 255)
        centered, padded = center_crop(image), prepare_image(image)
        self.assertEqual(centered.shape, (1, 3, size, size))
        self.assertEqual(centered.dtype, np.float32)
        self.assertTrue(np.isfinite(centered).all())
        self.assertLess(centered[0, 0].max(), 0)
        self.assertGreater(padded[0, 0].max(), 1)

    def test_string_labels_and_one_class_samples_fail_before_model_load(self):
        """Do not silently treat string false as a positive annotation."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'labels.json'
            for labels in ([True, 'false'], [False, False]):
                path.write_text(json.dumps({'images': [{'path': 'missing.jpg', 'book_present': value} for value in labels]}))
                with self.assertRaises(ValueError):
                    audit(path, Path(directory), Path(directory) / 'report.json')
