"""Verify metric spine measurement from synthetic depth maps with known geometry."""
import unittest

import numpy as np

from library_claim.domain.models import BookObservation, Box
from library_claim.measurement.spine import DepthMap, book_dimensions, measure_spine, shelf_run_m

SIZE = 1000
BOX = Box(x_min=0.48, y_min=0.40, x_max=0.52, y_max=0.60)


def depth_map(depth: np.ndarray) -> DepthMap:
    """Wrap a depth array with a 1000-pixel focal length centred camera.

    Args:
        depth: Depth in metres at full image resolution.

    Returns:
        A depth map for a 1000×1000 image.
    """
    return DepthMap(depth_m=depth.astype(np.float32), fx=1000, fy=1000, cx=500, cy=500,
                    image_width=SIZE, image_height=SIZE, method='depth_pro')


class SpineMeasurementTests(unittest.TestCase):
    """Exercise geometry, background rejection, tilt, and refusal cases."""

    def test_fronto_parallel_spine(self):
        """A 200×40 px spine at 1 m with f=1000 px measures 20×4 cm."""
        result = measure_spine(BOX, 'spine', [], depth_map(np.ones((SIZE, SIZE))))
        self.assertAlmostEqual(result.height_cm, 20.0, delta=0.2)
        self.assertAlmostEqual(result.thickness_cm, 4.0, delta=0.04)
        self.assertEqual(result.method, 'depth_pro')

    def test_background_at_box_edges_does_not_inflate(self):
        """Ignore a far wall visible in the outer rim of a loose box."""
        depth = np.full((SIZE, SIZE), 3.0)
        # The book occupies the inner 80 % of the box; the rim shows the wall behind it.
        depth[420:580, 484:516] = 1.0
        result = measure_spine(BOX, 'spine', [], depth_map(depth))
        self.assertAlmostEqual(result.height_cm, 20.0, delta=0.4)

    def test_tilted_spine_uses_depth_gradient(self):
        """A spine receding in depth is longer than its fronto-parallel projection."""
        depth = np.tile(np.linspace(0.9, 1.1, SIZE)[:, None], (1, SIZE))
        flat = measure_spine(BOX, 'spine', [], depth_map(np.ones((SIZE, SIZE))))
        tilted = measure_spine(BOX, 'spine', [], depth_map(depth))
        self.assertGreater(tilted.height_cm, flat.height_cm)

    def test_non_spine_or_cut_off_not_measured(self):
        """Refuse covers and clipped or occluded spines."""
        depth = depth_map(np.ones((SIZE, SIZE)))
        self.assertIsNone(measure_spine(BOX, 'cover', [], depth))
        self.assertIsNone(measure_spine(BOX, 'spine', ['cut_off'], depth))
        self.assertIsNone(measure_spine(BOX, 'spine', ['occlusion'], depth))

    def test_missing_depth_returns_none(self):
        """Unknown depth (zeros) yields no measurement."""
        self.assertIsNone(measure_spine(BOX, 'spine', [], depth_map(np.zeros((SIZE, SIZE)))))

    def test_low_resolution_depth_grid_scales_to_image(self):
        """A 96×54 browser grid samples the same geometry as full resolution."""
        grid = DepthMap(depth_m=np.ones((54, 96), np.float32), fx=1000, fy=1000, cx=500, cy=500,
                        image_width=SIZE, image_height=SIZE, method='webxr_depth')
        self.assertAlmostEqual(measure_spine(BOX, 'spine', [], grid).height_cm, 20.0, delta=0.2)

    def test_book_dimensions_median_and_shelf_run(self):
        """Use the median of measured observations and sum thicknesses for shelf run."""
        rows = [BookObservation(sweep_id='s', book_id='b', frame_id=f'f{i}', bbox=BOX, spine_height_cm=h,
                                spine_thickness_cm=3.0, scale_method='depth_pro')
                for i, h in enumerate((20.0, 21.0, 22.0))]
        rows.append(BookObservation(sweep_id='s', book_id='b', frame_id='f9', bbox=BOX))
        self.assertEqual(book_dimensions(rows), (21.0, 3.0, 'depth_pro'))
        self.assertIsNone(book_dimensions(rows[-1:]))
        self.assertAlmostEqual(shelf_run_m([3.0, 2.5]), 0.055)
