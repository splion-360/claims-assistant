"""Verify the local exposure and sharpness gate that runs before Drex."""
import unittest

from library_claim.config import CONFIG
from live_demo.camera_gate import local_gate
from test_camera_logging import quality


class CameraGateTests(unittest.TestCase):
    """Check the gate's boundaries and that BRISQUE no longer rejects frames."""

    def test_exposure_and_sharpness_boundaries(self):
        """Reject only beyond the configured limits."""
        limit = CONFIG.quality.max_extreme_fraction
        self.assertEqual(local_gate(quality(dark_fraction=limit + .01)), 'extreme_exposure')
        self.assertEqual(local_gate(quality(bright_fraction=limit + .01)), 'extreme_exposure')
        self.assertIsNone(local_gate(quality(dark_fraction=limit)))
        self.assertEqual(local_gate(quality(sharpness=CONFIG.quality.min_sharpness - .1)),
                         'insufficient_texture_or_sharpness')
        self.assertIsNone(local_gate(quality(sharpness=CONFIG.quality.min_sharpness)))

    def test_brisque_does_not_gate(self):
        """Leave perceptual-quality judgement to the Drex frame check."""
        self.assertIsNone(local_gate(quality(brisque=95)))
