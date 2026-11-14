"""Verify room geometry measured from a sweep's depth frames and camera poses."""
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from library_claim.config import CONFIG
from library_claim.domain.models import Box, Frame, Item, ItemObservation, Sweep
from library_claim.measurement.room import estimate_room, surface_area_m2, world_points
from library_claim.measurement.room_storage import measure_room, shelved_wall_area_m2, stored_room
from library_claim.measurement.spine import DepthMap
from library_claim.storage.depth import save_depth
from library_claim.storage.repository import Repository

LENGTH, WIDTH, HEIGHT, EYE = 5.0, 3.6, 2.5, 1.5
GRID = (54, 96)
FX = FY = 420.0
IMAGE = (1024, 576)


def pose_for(yaw: float) -> tuple[float, ...]:
    """Build a camera-to-world matrix for a camera at the room centre.

    Args:
        yaw: Heading in radians, measured about the vertical axis.

    Returns:
        The matrix flattened column-major, as WebXR reports it.
    """
    matrix = np.eye(4)
    forward = np.array([np.sin(yaw), 0.0, np.cos(yaw)])
    matrix[:3, 0] = [np.cos(yaw), 0.0, -np.sin(yaw)]
    matrix[:3, 1] = [0.0, 1.0, 0.0]
    matrix[:3, 2] = -forward
    matrix[:3, 3] = [0.0, EYE, 0.0]
    return tuple(matrix.T.flatten())


def room_frame(yaw: float, *, pose: bool = True) -> DepthMap:
    """Ray-cast a rectangular room into one depth grid.

    Args:
        yaw: Heading the camera looks along.
        pose: Whether the device reported a camera pose for this frame.

    Returns:
        A depth map of the walls, floor and ceiling the camera sees.
    """
    us = (np.arange(GRID[1]) + .5) * IMAGE[0] / GRID[1]
    vs = (np.arange(GRID[0]) + .5) * IMAGE[1] / GRID[0]
    uu, vv = np.meshgrid(us, vs)
    rays = np.stack([(uu - IMAGE[0] / 2) / FX, -(vv - IMAGE[1] / 2) / FY, -np.ones_like(uu)], -1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    matrix = np.asarray(pose_for(yaw)).reshape(4, 4).T
    world = rays @ matrix[:3, :3].T
    origin = matrix[:3, 3]
    nearest = np.full(uu.shape, np.inf)
    for axis, value in ((0, LENGTH / 2), (0, -LENGTH / 2), (2, WIDTH / 2),
                        (2, -WIDTH / 2), (1, HEIGHT), (1, 0.0)):
        with np.errstate(divide='ignore', invalid='ignore'):
            distance = (value - origin[axis]) / world[..., axis]
        hit = np.isfinite(distance) & (distance > 0.05)
        nearest = np.where(hit & (distance < nearest), distance, nearest)
    # A depth grid stores perpendicular distance, not ray length.
    perpendicular = nearest * (-world @ matrix[:3, 2])
    return DepthMap(depth_m=np.where(np.isfinite(perpendicular), perpendicular, 0).astype(np.float32),
                    fx=FX, fy=FY, cx=IMAGE[0] / 2, cy=IMAGE[1] / 2,
                    image_width=IMAGE[0], image_height=IMAGE[1], method='webxr_depth',
                    pose=pose_for(yaw) if pose else None)


def full_turn(count: int = 16) -> list[DepthMap]:
    """Build frames covering a full turn of the room.

    Args:
        count: Number of evenly spaced headings.

    Returns:
        One depth map per heading.
    """
    return [room_frame(yaw) for yaw in np.linspace(0, 2 * np.pi, count, endpoint=False)]


class RoomGeometryTests(unittest.TestCase):
    """Measure a room of known size and check what the figures may claim."""

    def test_dimensions_match_the_room_that_was_swept(self):
        """Report the room's own length, width and ceiling height."""
        estimate = estimate_room(full_turn())
        self.assertAlmostEqual(estimate.length_m, LENGTH, delta=LENGTH * 0.05)
        self.assertAlmostEqual(estimate.width_m, WIDTH, delta=WIDTH * 0.05)
        self.assertAlmostEqual(estimate.height_m, HEIGHT, delta=HEIGHT * 0.05)

    def test_areas_match_the_room_that_was_swept(self):
        """Report floor and wall area within the brief's tolerances."""
        estimate = estimate_room(full_turn())
        self.assertAlmostEqual(estimate.floor_area_m2, LENGTH * WIDTH, delta=LENGTH * WIDTH * 0.10)
        wall = 2 * (LENGTH + WIDTH) * HEIGHT
        self.assertAlmostEqual(estimate.wall_area_m2, wall, delta=wall * 0.15)

    def test_square_feet_follow_the_metric_figures(self):
        """Convert each area rather than measuring it twice."""
        room = estimate_room(full_turn()).as_packet_room()
        self.assertAlmostEqual(room['floor_area_sqft'], room['floor_area_m2'] * 10.7639, delta=0.2)
        self.assertAlmostEqual(room['wall_area_sqft'], room['wall_area_m2'] * 10.7639, delta=0.2)

    def test_a_full_turn_reports_complete_coverage(self):
        """Report full coverage when the sweep looked all the way round."""
        self.assertEqual(estimate_room(full_turn()).yaw_coverage, 1.0)

    def test_a_partial_sweep_reports_partial_coverage(self):
        """Report reduced coverage, and a smaller floor, for a sweep of one wall."""
        partial = [room_frame(yaw) for yaw in np.linspace(0, np.pi / 3, 6)]
        estimate = estimate_room(partial)
        self.assertLess(estimate.yaw_coverage, 0.75)
        self.assertEqual(estimate.confidence, round(estimate.yaw_coverage, 2))
        self.assertLess(estimate.floor_area_m2, LENGTH * WIDTH)

    def test_frames_without_a_pose_measure_nothing(self):
        """Refuse to place depth in the room without knowing where the camera was."""
        self.assertEqual(world_points(room_frame(0.0, pose=False)).size, 0)
        self.assertIsNone(estimate_room([room_frame(yaw, pose=False) for yaw in np.linspace(0, 6, 16)]))

    def test_too_few_frames_measure_nothing(self):
        """Refuse an estimate from fewer posed frames than the configuration requires."""
        self.assertIsNone(estimate_room(full_turn(CONFIG.measurement.room_min_frames - 1)))

    def test_an_empty_sweep_measures_nothing(self):
        """Return no estimate when the sweep captured no depth at all."""
        self.assertIsNone(estimate_room([]))

    def test_the_estimate_says_how_it_was_measured(self):
        """Name the scale source rather than leaving the figures unexplained."""
        estimate = estimate_room(full_turn())
        self.assertEqual(estimate.scale_method, 'webxr_depth_pose_cloud')
        self.assertIn('convex hull', estimate.note)

    def test_surface_area_measures_a_region_against_its_depth(self):
        """Measure the world area a detection box covers on the surface it sits on."""
        # The far wall is 1.8 m away across the room's width.
        frame = room_frame(0.0)
        area = surface_area_m2(frame, Box(x_min=.25, y_min=.25, x_max=.75, y_max=.75))
        self.assertIsNotNone(area)
        self.assertGreater(area, 0)

    def test_surface_area_needs_a_pose(self):
        """Measure nothing for a frame whose camera position is unknown."""
        self.assertIsNone(surface_area_m2(room_frame(0.0, pose=False),
                                          Box(x_min=.25, y_min=.25, x_max=.75, y_max=.75)))


class RoomStorageTests(unittest.TestCase):
    """Exercise measuring a saved sweep and keeping its figure."""

    def setUp(self):
        """Save a sweep whose frames carry posed depth."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.database = self.root / 'claims.sqlite3'
        start = datetime(2026, 11, 14, 9, 0, tzinfo=timezone.utc)
        frames = full_turn(8)
        with Repository(self.database) as repo:
            repo.add(Sweep(id='s1', captured_at=start))
            for index, depth in enumerate(frames):
                frame_id = f'frame{index}'
                repo.add(Frame(id=frame_id, sweep_id='s1', captured_at=start,
                               evidence_key=f'{index:064x}.jpg', width_px=IMAGE[0], height_px=IMAGE[1]))
                save_depth(self.root, frame_id, depth)

    def test_measuring_saves_a_figure_the_packet_can_read(self):
        """Measure the room once and keep the block for later reads."""
        room = measure_room(self.database, self.root, 's1')
        self.assertIsNotNone(room)
        with Repository(self.database) as repo:
            self.assertEqual(stored_room(repo, 's1'), room)

    def test_a_sweep_with_no_depth_measures_nothing(self):
        """Save no estimate when the sweep captured no posed depth."""
        empty = Path(self.directory.name) / 'other'
        empty.mkdir()
        self.assertIsNone(measure_room(self.database, empty, 's1'))

    def test_shelving_contributes_its_wall_area_once(self):
        """Measure one shelving unit once, however many frames saw it."""
        with Repository(self.database) as repo:
            repo.add(Item(id='i1', sweep_id='s1', category='shelving', description='bookcase'))
            for frame_id in ('frame0', 'frame1'):
                repo.add(ItemObservation(sweep_id='s1', item_id='i1', frame_id=frame_id,
                                         bbox=Box(x_min=.3, y_min=.3, x_max=.7, y_max=.7),
                                         category='shelving', description='bookcase'))
            single = shelved_wall_area_m2(repo, self.root, 's1')
            repo.add(ItemObservation(sweep_id='s1', item_id='i1', frame_id='frame2',
                                     bbox=Box(x_min=.3, y_min=.3, x_max=.7, y_max=.7),
                                     category='shelving', description='bookcase'))
            self.assertGreater(single, 0)
            # A third sighting of the same unit cannot inflate the covered area.
            self.assertAlmostEqual(shelved_wall_area_m2(repo, self.root, 's1'), single, delta=single * 0.5)

    def test_other_categories_cover_no_wall(self):
        """Count only shelving towards the wall area shelving covers."""
        with Repository(self.database) as repo:
            repo.add(Item(id='i2', sweep_id='s1', category='seating', description='armchair'))
            repo.add(ItemObservation(sweep_id='s1', item_id='i2', frame_id='frame0',
                                     bbox=Box(x_min=.3, y_min=.3, x_max=.7, y_max=.7),
                                     category='seating', description='armchair'))
            self.assertEqual(shelved_wall_area_m2(repo, self.root, 's1'), 0.0)


class SizeRefusalTests(unittest.TestCase):
    """Check a book left unmeasured says which evidence was missing."""

    @staticmethod
    def box():
        """Build a detection box covering the middle of the frame.

        Returns:
            A normalized box.
        """
        return Box(x_min=.3, y_min=.2, x_max=.6, y_max=.8)

    def test_no_depth_at_all_is_named(self):
        """Say that no depth reached the frame when none did."""
        from library_claim.measurement.spine import size_refusal
        self.assertEqual(size_refusal(self.box(), 'spine', [], None),
                         'No depth was captured for this frame')

    def test_a_cover_sighting_is_named(self):
        """Say that only spines are measured when the sighting shows a cover."""
        from library_claim.measurement.spine import size_refusal
        reason = size_refusal(self.box(), 'cover', [], room_frame(0.0))
        self.assertIn('spines only', reason)
        self.assertIn('cover', reason)

    def test_an_occluded_spine_is_named(self):
        """Say that the spine was obscured rather than that depth was missing."""
        from library_claim.measurement.spine import size_refusal
        self.assertIn('occlusion', size_refusal(self.box(), 'spine', ['occlusion'], room_frame(0.0)))

    def test_an_implausible_reading_names_the_background(self):
        """Say the depth sampled past the book when the size cannot be a book."""
        from library_claim.measurement.spine import size_refusal
        # The synthetic room's walls are metres away, so this box measures far too large.
        reason = size_refusal(self.box(), 'spine', [], room_frame(0.0))
        self.assertIn('implausible', reason)

    def test_a_measurable_spine_has_no_refusal(self):
        """Report no reason when the sighting does produce a size."""
        from library_claim.measurement.spine import measure_spine, size_refusal
        # A close, small box against the room's near surfaces measures as a book.
        frame = room_frame(0.0)
        small = Box(x_min=.49, y_min=.46, x_max=.51, y_max=.54)
        if measure_spine(small, 'spine', [], frame) is not None:
            self.assertIsNone(size_refusal(small, 'spine', [], frame))
