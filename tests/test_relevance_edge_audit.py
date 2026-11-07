"""Verify crop coverage and paired-score selection in the offline edge experiment."""
import unittest

import numpy as np

from library_claim.capture.relevance.edge_audit import crop_boxes, grouped_scores, select_crop
from library_claim.capture.relevance.classifier import PROMPTS


class EdgeAuditTests(unittest.TestCase):
    """Check that edge coverage and evidence selection preserve their contracts."""

    def test_landscape_and_portrait_crops_cover_every_pixel(self):
        """Keep both edges covered without gaps or out-of-bounds crops."""
        for width, height in ((1024, 576), (576, 1024)):
            with self.subTest(width=width):
                coverage = np.zeros((height, width), dtype=bool)
                for left, top, right, bottom in crop_boxes(width, height, 3):
                    self.assertGreaterEqual(left, 0)
                    self.assertGreaterEqual(top, 0)
                    self.assertLessEqual(right, width)
                    self.assertLessEqual(bottom, height)
                    self.assertEqual(right - left, bottom - top)
                    coverage[top:bottom, left:right] = True
                self.assertTrue(coverage.all())

    def test_square_deduplicates_and_extreme_aspect_rejects_gaps(self):
        """Avoid redundant square inference and incomplete panorama coverage."""
        self.assertEqual(crop_boxes(224, 224, 3), [(0, 0, 224, 224)])
        with self.assertRaises(ValueError):
            crop_boxes(2000, 100, 3)
        with self.assertRaises(ValueError):
            crop_boxes(100, 100, 1)

    def test_highest_book_crop_wins(self):
        """Use the strongest crop regardless of where the book appears."""
        scores = np.array([[.21], [.30], [.25]])
        self.assertEqual(select_crop(scores), 1)

    def test_tied_crops_select_first(self):
        """Keep deterministic provenance when two crops have the same score."""
        self.assertEqual(select_crop(np.array([[.3], [.3], [.2]])), 0)

    def test_prompt_groups_reduce_each_crop_independently(self):
        """Use max similarity within configured target prompts for each view."""
        count = sum(map(len, PROMPTS.values()))
        similarities = np.arange(2 * count).reshape(2, count) / (2 * count)
        output = grouped_scores(similarities)
        boundary = len(PROMPTS['books'])
        np.testing.assert_array_equal(output[:, 0], similarities[:, :boundary].max(axis=1))
        self.assertEqual(output.shape, (2, 1))
