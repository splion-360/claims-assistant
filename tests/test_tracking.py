"""Verify frame association and perspective mapping on controlled camera views."""
import unittest

import cv2
import numpy as np

from library_claim.capture.tracking.geometry import intersection_area, transform, verify
from library_claim.capture.tracking.tracker import FrameTracker


def textured_view(seed=7):
    """Generate a repeatable textured reference with distributed features.

    Args:
        seed: Random generator seed defining the synthetic scene.

    Returns:
        A grayscale image suitable for geometric tracking tests.
    """
    rng = np.random.default_rng(seed)
    image = rng.integers(30, 225, (360, 480), dtype=np.uint8)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    for index in range(10):
        cv2.putText(image, f'BOOK {seed} {index}', (10, 30 + index * 32),
                    cv2.FONT_HERSHEY_SIMPLEX, .6, 250, 1)
    return image


class TrackingTests(unittest.TestCase):
    """Exercise flow, keyframe retrieval, and uncertain associations."""

    def test_repeated_and_translated_views_follow_reference(self):
        """Maintain a stable reference while tracking moderate image translation."""
        image = textured_view()
        tracker = FrameTracker()
        first = tracker.update(image, 'first', 0)
        repeated = tracker.update(image, 'repeat', .2)
        matrix = np.float64([[1, 0, 8], [0, 1, 5], [0, 0, 1]])
        shifted = cv2.warpPerspective(image, matrix, (480, 360))
        result = tracker.update(shifted, 'shifted', .4)
        self.assertEqual(first.status, 'reference_created')
        self.assertEqual(repeated.status, 'tracked')
        self.assertEqual(result.reference_frame_id, 'first')
        np.testing.assert_allclose(result.map_polygon([(.5, .5)]),
                                   [[.5 - 8 / 479, .5 - 5 / 359]], atol=.01)
        self.assertGreater(result.reference_overlap, .9)

    def test_recovery_verifies_tilt_and_maps_book_corners(self):
        """Recover a reference after a time gap with perspective distortion."""
        image = textured_view()
        tracker = FrameTracker()
        tracker.update(image, 'reference', 0)
        source = np.float32([[0, 0], [479, 0], [479, 359], [0, 359]])
        destination = np.float32([[30, 20], [450, 0], [470, 330], [5, 350]])
        matrix = cv2.getPerspectiveTransform(source, destination)
        tilted = cv2.warpPerspective(image, matrix, (480, 360))
        result = tracker.update(tilted, 'tilted', 3)
        self.assertEqual(result.status, 'recovered')
        self.assertEqual(result.reference_frame_id, 'reference')
        book = np.float32([[100, 100], [140, 100], [140, 270], [100, 270]])
        observed = transform(book, matrix) / [479, 359]
        np.testing.assert_allclose(result.map_polygon(observed.tolist()), book / [479, 359], atol=.015)

    def test_blank_view_is_uncertain_and_return_recovers(self):
        """Avoid treating a featureless wall as a matched or new book region."""
        image = textured_view()
        tracker = FrameTracker()
        tracker.update(image, 'reference', 0)
        blank = tracker.update(np.zeros_like(image), 'wall', .2)
        self.assertEqual(blank.status, 'uncertain')
        self.assertIsNone(blank.current_to_reference)
        with self.assertRaises(ValueError):
            blank.map_polygon([(.2, .2)])
        returned = tracker.update(image, 'return', .4)
        self.assertEqual(returned.status, 'recovered')
        self.assertEqual(returned.reference_frame_id, 'reference')

    def test_unrelated_views_do_not_merge_and_cache_is_bounded(self):
        """Create provisional references for unmatched scenes within the budget."""
        tracker = FrameTracker(capacity=2)
        for i in range(4):
            result = tracker.update(textured_view(10 + i), f'frame{i}', i * 2)
            self.assertEqual(result.status, 'reference_created')
            self.assertEqual(result.reference_frame_id, f'frame{i}')
        self.assertEqual(len(tracker.keyframes.views), 2)

    def test_ransac_rejects_spatially_concentrated_support(self):
        """Reject matches restricted to a tiny repeated label."""
        rng = np.random.default_rng(1)
        points = rng.uniform(30, 40, (30, 2)).astype(np.float32)
        self.assertIsNone(verify(points, points + 3, (360, 480), (360, 480), robust=True))

    def test_input_time_and_projective_horizon_validation(self):
        """Reject out-of-order frames and invalid projective mappings."""
        tracker = FrameTracker()
        tracker.update(textured_view(), 'one', 1)
        with self.assertRaises(ValueError):
            tracker.update(textured_view(), 'two', .5)
        with self.assertRaises(ValueError):
            transform(np.float32([[0, 0], [1, 1]]), np.zeros((3, 3)))


class IntersectionAreaTests(unittest.TestCase):
    """Measure shared area where coincident edges defeat OpenCV's convex intersection."""

    RECTANGLE = np.float32([[0, 0], [479, 0], [479, 359], [0, 359]])

    def test_coincident_outlines_report_the_whole_area(self):
        """Report full overlap for a mapping that is an identity within float noise."""
        # A perfect re-match of a frame against itself produces exactly this polygon.
        dusted = np.float32([[6.1e-14, 8.6e-14], [479, 1.1e-13], [479, 359], [-3.5e-14, 359]])
        self.assertAlmostEqual(intersection_area(dusted, self.RECTANGLE), 479 * 359, places=3)

    def test_identical_outlines_report_the_whole_area(self):
        """Report full overlap for two bitwise-identical rectangles."""
        self.assertAlmostEqual(intersection_area(self.RECTANGLE, self.RECTANGLE), 479 * 359, places=3)

    def test_winding_direction_does_not_change_the_area(self):
        """Give the same area whichever direction the corners are listed in."""
        reversed_corners = self.RECTANGLE[::-1].copy()
        self.assertAlmostEqual(intersection_area(reversed_corners, self.RECTANGLE), 479 * 359, places=3)

    def test_offset_rectangles_report_their_shared_region(self):
        """Measure the shared region of two rectangles offset by a known amount."""
        shifted = self.RECTANGLE + np.float32([8, 5])
        self.assertAlmostEqual(intersection_area(shifted, self.RECTANGLE), (479 - 8) * (359 - 5), places=3)

    def test_contained_polygon_reports_its_own_area(self):
        """Report the smaller area when one polygon lies wholly inside the other."""
        inner = np.float32([[100, 100], [300, 100], [300, 200], [100, 200]])
        self.assertAlmostEqual(intersection_area(inner, self.RECTANGLE), 200 * 100, places=3)
        triangle = np.float32([[10, 10], [210, 10], [10, 110]])
        self.assertAlmostEqual(intersection_area(triangle, self.RECTANGLE), 200 * 100 / 2, places=3)

    def test_disjoint_and_edge_touching_polygons_share_no_area(self):
        """Report no shared area for separated outlines and for ones meeting on an edge."""
        self.assertEqual(intersection_area(self.RECTANGLE + np.float32([1000, 0]), self.RECTANGLE), 0)
        self.assertEqual(intersection_area(self.RECTANGLE + np.float32([479, 0]), self.RECTANGLE), 0)

    def test_degenerate_vertices_are_rejected(self):
        """Refuse inputs that cannot describe a polygon instead of returning an area."""
        with self.assertRaises(ValueError):
            intersection_area(np.float32([[0, 0], [1, 1]]), self.RECTANGLE)
        with self.assertRaises(ValueError):
            intersection_area(np.float32([[0, 0], [1, 0], [np.nan, 1]]), self.RECTANGLE)
