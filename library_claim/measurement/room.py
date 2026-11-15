"""Measure a room's floor and wall area from the depth and poses of one sweep."""
from dataclasses import dataclass
import logging

import cv2
import numpy as np

from library_claim.config import CONFIG
from .spine import DepthMap

logger = logging.getLogger(__name__)
SQUARE_FEET_PER_SQUARE_METRE = 10.763910416709722


# An outline filling this share of its bounding rectangle is reported as a rectangle.
RECTANGLE_MIN = 0.9
SHAPE_METHOD = ('convex hull of floor points seen in posed depth; floor area is the hull area and wall '
                'area is the hull perimeter times ceiling height, so recesses are filled in')


@dataclass(frozen=True)
class RoomEstimate:
    """One sweep's room geometry, with the coverage that backs it."""
    length_m: float
    width_m: float
    height_m: float
    floor_area_m2: float
    wall_area_m2: float
    shelved_wall_area_m2: float
    scale_method: str
    confidence: float
    # Fraction of a full turn the sweep looked across, and the points it stood on.
    yaw_coverage: float
    sample_count: int
    footprint_m: list[tuple[float, float]]
    note: str

    def as_packet_room(self) -> dict:
        """Render the packet's room block, in metres and square feet.

        Returns:
            The block the output contract fixes for a room.
        """
        return {
            'length_m': round(self.length_m, 2), 'width_m': round(self.width_m, 2),
            'height_m': round(self.height_m, 2),
            'floor_area_m2': round(self.floor_area_m2, 2),
            'floor_area_sqft': round(self.floor_area_m2 * SQUARE_FEET_PER_SQUARE_METRE, 1),
            'wall_area_m2': round(self.wall_area_m2, 2),
            'wall_area_sqft': round(self.wall_area_m2 * SQUARE_FEET_PER_SQUARE_METRE, 1),
            'shelved_wall_area_m2': round(self.shelved_wall_area_m2, 2),
            'scale_method': self.scale_method, 'confidence': round(self.confidence, 2),
            'yaw_coverage': round(self.yaw_coverage, 2), 'sample_count': self.sample_count,
            'footprint_m': [[round(x, 2), round(z, 2)] for x, z in self.footprint_m],
            'shape_note': self.note,
            # How the outline was formed, and how close it is to the length-by-width rectangle.
            'shape_method': SHAPE_METHOD,
            'rectangularity': round(self.rectangularity, 2),
            'shape': 'rectangle' if self.rectangularity >= RECTANGLE_MIN else 'irregular',
        }

    @property
    def rectangularity(self) -> float:
        """Floor area as a share of the bounding length-by-width rectangle.

        Returns:
            1.0 for a rectangle; lower for L-shapes and angled walls.
        """
        bounding = self.length_m * self.width_m
        return min(1.0, self.floor_area_m2 / bounding) if bounding else 0.0


def world_points(depth: DepthMap, stride: int = 2) -> np.ndarray:
    """Back-project a depth grid into the sweep's world frame.

    Args:
        depth: A depth map carrying intrinsics and a camera pose.
        stride: Sample every `stride` cells, bounding the point count per frame.

    Returns:
        An N-by-3 array of world points, empty when the frame has no usable pose
        or no depth cell within the configured range.
    """
    if depth.pose is None:
        return np.empty((0, 3))
    grid = np.asarray(depth.depth_m, dtype=float)[::stride, ::stride]
    rows, columns = grid.shape
    # Grid cells map onto the JPEG that the intrinsics are expressed in.
    scale_x = depth.image_width / (grid.shape[1] * stride)
    scale_y = depth.image_height / (grid.shape[0] * stride)
    us = (np.arange(columns) * stride * scale_x + scale_x / 2)[None, :].repeat(rows, 0)
    vs = (np.arange(rows) * stride * scale_y + scale_y / 2)[:, None].repeat(columns, 1)
    low, high = CONFIG.measurement.room_depth_range_m
    usable = np.isfinite(grid) & (grid > low) & (grid < high)
    if not usable.any():
        return np.empty((0, 3))
    z = grid[usable]
    # WebXR cameras look down -z, so a point at depth z sits at -z in camera space.
    camera = np.stack([(us[usable] - depth.cx) / depth.fx * z,
                       -(vs[usable] - depth.cy) / depth.fy * z,
                       -z, np.ones(z.size)])
    matrix = np.asarray(depth.pose, dtype=float).reshape(4, 4).T
    return (matrix @ camera)[:3].T


def _footprint(points: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    """Fit the room's floor outline and its enclosing rectangle.

    Args:
        points: World points, N-by-3 with y upward.

    Returns:
        The hull of the floor projection, its area, and the rectangle's sides.
    """
    plan = points[:, [0, 2]].astype(np.float32)
    hull = cv2.convexHull(plan)
    area = float(cv2.contourArea(hull))
    (_, _), (side_a, side_b), _ = cv2.minAreaRect(hull)
    return hull.reshape(-1, 2), area, float(max(side_a, side_b)), float(min(side_a, side_b))


def estimate_room(depths: list[DepthMap], shelved_area_m2: float = 0.0) -> RoomEstimate | None:
    """Estimate the room's dimensions and areas from a sweep's depth frames.

    The sweep's AR space is gravity aligned, so height is read along y and the
    floor outline is the hull of the points projected onto the x-z plane. That
    outline is what the camera actually saw: a sweep that does not turn through
    the room measures only the part it looked at, which `yaw_coverage` reports
    and `confidence` reflects.

    Args:
        depths: Depth maps from the sweep, in any order.
        shelved_area_m2: Wall area covered by shelving, measured separately.

    Returns:
        The estimate, or None when too few posed frames carry usable depth.

    Raises:
        ValueError: A depth map is malformed.
    """
    posed = [depth for depth in depths if depth.pose is not None]
    if len(posed) < CONFIG.measurement.room_min_frames:
        return None
    clouds, yaws = [], []
    for depth in posed:
        points = world_points(depth)
        if points.size:
            clouds.append(points)
            matrix = np.asarray(depth.pose, dtype=float).reshape(4, 4).T
            # The camera's forward axis, flattened onto the floor, gives the heading it looked along.
            forward = -matrix[:3, 2]
            # A frame covers its whole field of view, not just the heading at its centre.
            half_fov = float(np.arctan(depth.image_width / (2 * depth.fx)))
            yaws.append((float(np.arctan2(forward[0], forward[2])), half_fov))
    if not clouds:
        return None
    points = np.concatenate(clouds)
    if len(points) < CONFIG.measurement.room_min_points:
        return None
    hull, floor_area, length, width = _footprint(points)
    # Trimmed extremes keep one stray reading through a doorway from raising the ceiling.
    low, high = np.percentile(points[:, 1], [1, 99])
    height = float(high - low)
    perimeter = float(cv2.arcLength(hull.astype(np.float32).reshape(-1, 1, 2), True))
    # Headings are binned around a full turn; the share of filled bins is the coverage.
    bins = np.zeros(CONFIG.measurement.room_yaw_bins, dtype=bool)
    edges = (np.arange(len(bins)) + 0.5) / len(bins) * 2 * np.pi - np.pi
    for yaw, half_fov in yaws:
        # Wrap the angular difference into [-pi, pi] before testing the wedge.
        offset = np.abs((edges - yaw + np.pi) % (2 * np.pi) - np.pi)
        bins |= offset <= half_fov
    coverage = float(bins.mean())
    note = ('Floor outline is the convex hull of the surfaces the sweep saw, so alcoves and '
            'recesses are measured as filled. A sweep that does not turn through the whole '
            'room measures only the part it looked at.')
    return RoomEstimate(
        length_m=length, width_m=width, height_m=height, floor_area_m2=floor_area,
        wall_area_m2=perimeter * height, shelved_wall_area_m2=shelved_area_m2,
        scale_method='webxr_depth_pose_cloud', confidence=round(coverage, 2),
        yaw_coverage=coverage, sample_count=int(len(points)),
        footprint_m=[(float(x), float(z)) for x, z in hull], note=note)


def surface_area_m2(depth: DepthMap, box) -> float | None:
    """Measure the world area a detection's box covers on the surface it sits on.

    Args:
        depth: Depth map for the frame the detection came from, carrying a pose.
        box: Normalized detection box in that frame.

    Returns:
        The area in square metres, or None when the region has no usable depth.
    """
    if depth.pose is None:
        return None
    grid = np.asarray(depth.depth_m, dtype=float)
    rows, columns = grid.shape
    left, right = int(box.x_min * columns), max(int(box.x_max * columns), int(box.x_min * columns) + 1)
    top, bottom = int(box.y_min * rows), max(int(box.y_max * rows), int(box.y_min * rows) + 1)
    region = grid[top:bottom, left:right]
    low, high = CONFIG.measurement.room_depth_range_m
    usable = np.isfinite(region) & (region > low) & (region < high)
    if usable.sum() < 4:
        return None
    # The median distance over the region scales its angular size into metres.
    distance = float(np.median(region[usable]))
    width_m = (box.x_max - box.x_min) * depth.image_width / depth.fx * distance
    height_m = (box.y_max - box.y_min) * depth.image_height / depth.fy * distance
    return float(width_m * height_m)
