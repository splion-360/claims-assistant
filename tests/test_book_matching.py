"""Verify partial text compatibility and crop-level visual association safeguards."""
import unittest

import numpy as np

from library_claim.identification.matching import crop_features, preferred_reading, text_relation, visual_match
from library_claim.domain.models import Box


class BookMatchingTests(unittest.TestCase):
    """Check matching signals independently of inventory storage."""

    def test_normalization_partial_and_fuzzy_readings(self):
        """Accept supported variants while retaining distinct title words."""
        self.assertEqual(text_relation('The Good People!', 'the good people'), 'exact')
        self.assertEqual(text_relation('Good People', 'The Good People'), 'partial')
        self.assertEqual(text_relation('The Good Peop1e', 'The Good People'), 'fuzzy')
        self.assertEqual(text_relation('Good People', 'Bad People'), 'conflicting')
        self.assertEqual(text_relation(None, 'Good People'), 'unknown')
        self.assertEqual(text_relation('The', 'The Good People'), 'unknown')

    def test_display_preserves_fuller_reading_without_synthesizing_text(self):
        """Select literal compatible evidence and reject substantial contradictions."""
        self.assertEqual(preferred_reading(['Good People', 'The Good People']), 'The Good People')
        self.assertIsNone(preferred_reading(['Good People', 'Bad People']))
        self.assertEqual(preferred_reading(['The Good People', 'The Good People', 'The Good Peoplle']), 'The Good People')

    def test_textureless_crop_is_unknown_not_a_visual_match(self):
        """Reject blank crop correspondence even when the pixels are identical."""
        box = Box(x_min=0, y_min=0, x_max=1, y_max=1)
        features = crop_features(np.zeros((100, 100, 3), dtype=np.uint8), box)
        self.assertIsNone(visual_match(features, features))
