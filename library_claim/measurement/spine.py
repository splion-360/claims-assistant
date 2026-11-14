"""Measure spine height and thickness in centimetres from a metric depth map."""
from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import Literal

import numpy as np

from library_claim.config import CONFIG
from library_claim.domain.models import BookObservation, Box


@dataclass(frozen=True)
class DepthMap:
    """Metric depth for one frame plus pinhole intrinsics in source-image pixels."""
    depth_m: np.ndarray
    fx: float
    fy: float
    cx: float
    cy: float
    image_width: int
    image_height: int
    method: Literal['webxr_depth', 'depth_pro']


@dataclass(frozen=True)
class SpineMeasurement:
    """Measured spine size and the source of its metric scale."""
    height_cm: float
    thickness_cm: float
    method: str


def _depth_at(depth: DepthMap, u: float, v: float) -> float | None:
    """Return the median known depth in a small window around an image point.

    Args:
        depth: Depth map, possibly coarser than the image.
        u: Horizontal image coordinate in pixels.
        v: Vertical image coordinate in pixels.

    Returns:
        Depth in metres, or None when the window holds no known depth.
    """
    rows, cols = depth.depth_m.shape
    # Depth grids may be coarser than the JPEG; map image pixels onto grid cells.
    gx = int(np.clip(u * cols / depth.image_width, 0, cols - 1))
    gy = int(np.clip(v * rows / depth.image_height, 0, rows - 1))
    half = CONFIG.measurement.depth_window_px // 2
    window = depth.depth_m[max(gy - half, 0):gy + half + 1, max(gx - half, 0):gx + half + 1]
    known = window[np.isfinite(window) & (window > 0)]
    return float(np.median(known)) if known.size else None


def _axis_length_m(depth: DepthMap, start: tuple[float, float], end: tuple[float, float]) -> float | None:
    """Back-project two image points and return their 3-D distance.

    Args:
        depth: Depth map with intrinsics.
        start: First point (u, v) in image pixels.
        end: Second point (u, v) in image pixels.

    Returns:
        Distance in metres, or None when either depth is unknown.
    """
    points = []
    for u, v in (start, end):
        z = _depth_at(depth, u, v)
        if z is None:
            return None
        points.append(np.array([(u - depth.cx) * z / depth.fx, (v - depth.cy) * z / depth.fy, z]))
    return float(np.linalg.norm(points[0] - points[1]))


def measure_spine(bbox: Box, view: str | None, issues: list[str], depth: DepthMap) -> SpineMeasurement | None:
    """Measure one detected spine, or refuse when the evidence cannot support it.

    Endpoints are pulled inward by `measurement.depth_inset` of the axis length at each
    end, so loose boxes sample the book rather than the wall behind it; the measured
    distance is then scaled back to the full axis.

    Args:
        bbox: Normalized detection box in the source image.
        view: Visible view reported for the detection.
        issues: Readability issues reported for the detection.
        depth: Metric depth and intrinsics for the same image.

    Returns:
        Height (long axis) and thickness (short axis) in centimetres, or None.
    """
    if view != 'spine' or {'cut_off', 'occlusion'} & set(issues):
        return None
    x0, x1 = bbox.x_min * depth.image_width, bbox.x_max * depth.image_width
    y0, y1 = bbox.y_min * depth.image_height, bbox.y_max * depth.image_height
    cu, cv = (x0 + x1) / 2, (y0 + y1) / 2
    keep = 1 - 2 * CONFIG.measurement.depth_inset
    vertical = ((cu, cv - (y1 - y0) / 2 * keep), (cu, cv + (y1 - y0) / 2 * keep))
    horizontal = ((cu - (x1 - x0) / 2 * keep, cv), (cu + (x1 - x0) / 2 * keep, cv))
    long_axis, short_axis = (vertical, horizontal) if y1 - y0 >= x1 - x0 else (horizontal, vertical)
    height, thickness = _axis_length_m(depth, *long_axis), _axis_length_m(depth, *short_axis)
    if height is None or thickness is None or height <= 0 or thickness <= 0:
        return None
    return SpineMeasurement(height_cm=height / keep * 100, thickness_cm=thickness / keep * 100, method=depth.method)


def book_dimensions(observations: list[BookObservation]) -> tuple[float, float, str] | None:
    """Combine a book's measured observations into one size.

    Args:
        observations: Observations of one physical book.

    Returns:
        Median height and thickness in centimetres with the majority scale method, or None.
    """
    measured = [row for row in observations if row.spine_height_cm and row.spine_thickness_cm]
    if not measured:
        return None
    method = Counter(row.scale_method for row in measured).most_common(1)[0][0]
    return (median(row.spine_height_cm for row in measured), median(row.spine_thickness_cm for row in measured), method)


def shelf_run_m(thicknesses_cm: list[float]) -> float:
    """Total shelf length occupied by books placed side by side.

    Args:
        thicknesses_cm: One thickness per book in centimetres.

    Returns:
        Shelf run in metres.
    """
    return sum(thicknesses_cm) / 100
