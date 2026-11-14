"""Verify per-frame depth sidecars round-trip without loss."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from library_claim.measurement.spine import DepthMap
from library_claim.storage.depth import load_depth, save_depth


class DepthStorageTests(unittest.TestCase):
    """Check saving, loading and absence of depth sidecars."""

    def test_depth_round_trip(self):
        """Preserve the grid, intrinsics, image size and method."""
        depth = DepthMap(depth_m=np.arange(12, dtype=np.float32).reshape(3, 4), fx=1.5, fy=2.5, cx=3.5, cy=4.5,
                         image_width=640, image_height=480, method='webxr_depth')
        with tempfile.TemporaryDirectory() as directory:
            save_depth(Path(directory), 'frame1', depth)
            loaded = load_depth(Path(directory), 'frame1')
            self.assertIsNone(load_depth(Path(directory), 'missing'))
        np.testing.assert_array_equal(loaded.depth_m, depth.depth_m)
        self.assertEqual((loaded.fx, loaded.fy, loaded.cx, loaded.cy), (1.5, 2.5, 3.5, 4.5))
        self.assertEqual((loaded.image_width, loaded.image_height, loaded.method), (640, 480, 'webxr_depth'))


class DepthProTests(unittest.TestCase):
    """Check the Depth Pro adapter's failure contract without loading weights."""

    def test_load_failure_raises_depth_unavailable(self):
        """Turn any model loading error into DepthUnavailable."""
        from unittest.mock import patch
        from library_claim.measurement import depth_pro

        with patch.object(depth_pro, '_run', side_effect=OSError('no weights')):
            with self.assertRaises(depth_pro.DepthUnavailable):
                depth_pro.estimate_depth(b'\xff\xd8\xff')
