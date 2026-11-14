"""Verify browser WebXR depth payloads are decoded and bounded."""
import base64
import unittest

import numpy as np

from library_claim.storage.depth import parse_depth_payload


def payload(width=96, height=54, mm=1000, **overrides):
    """Build a browser depth payload with uniform depth.

    Args:
        width: Grid columns.
        height: Grid rows.
        mm: Depth of every cell in millimetres.
        **overrides: Field replacements.

    Returns:
        A payload dict as sent by the browser.
    """
    grid = np.full((height, width), mm, '<u2')
    data = {'width': width, 'height': height, 'mm': base64.b64encode(grid.tobytes()).decode(),
            'fx': 800.0, 'fy': 800.0, 'cx': 512.0, 'cy': 288.0}
    data.update(overrides)
    return data


class DepthPayloadTests(unittest.TestCase):
    """Check unit conversion and rejection of malformed grids."""

    def test_parse_depth_payload_scales_mm(self):
        """Convert millimetres to metres and keep intrinsics in JPEG pixels."""
        depth = parse_depth_payload(payload(mm=1234), 1024, 576)
        self.assertEqual(depth.depth_m.shape, (54, 96))
        self.assertAlmostEqual(float(depth.depth_m[0, 0]), 1.234, places=3)
        self.assertEqual((depth.fx, depth.cx, depth.image_width, depth.method), (800.0, 512.0, 1024, 'webxr_depth'))

    def test_parse_rejects_oversized_grid(self):
        """Bound memory by refusing grids larger than 160×120."""
        with self.assertRaises(ValueError):
            parse_depth_payload(payload(width=200, height=100), 1024, 576)

    def test_parse_rejects_wrong_length_and_bad_intrinsics(self):
        """Refuse byte counts that do not match the grid and non-finite intrinsics."""
        with self.assertRaises(ValueError):
            parse_depth_payload(payload(mm=1, width=96, height=54) | {'height': 50}, 1024, 576)
        with self.assertRaises(ValueError):
            parse_depth_payload(payload(fx=float('nan')), 1024, 576)
        with self.assertRaises(ValueError):
            parse_depth_payload(payload(fx=0), 1024, 576)
