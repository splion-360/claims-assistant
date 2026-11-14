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
    # Camera-to-world transform of the AR reference space, column-major, when the
    # device reported one. Room geometry needs it; spine sizes do not.
    pose: tuple[float, ...] | None = None


@dataclass(frozen=True)
class SpineMeasurement:
    """Measured spine size and the source of its metric scale."""
    height_cm: float
    thickness_cm: float
    method: str


def _to_grid(depth: DepthMap, u: float, v: float) -> tuple[int, int]:
    """Map an image pixel to the depth-grid cell containing it.

    Args:
        depth: Depth map, possibly coarser than the image.
        u: Horizontal image coordinate in pixels.
        v: Vertical image coordinate in pixels.

    Returns:
        Grid column and row.
    """
    rows, cols = depth.depth_m.shape
    return (int(np.clip(u * cols / depth.image_width, 0, cols - 1)),
            int(np.clip(v * rows / depth.image_height, 0, rows - 1)))


def _median_depth(depth: DepthMap, x0: float, y0: float, x1: float, y1: float) -> float | None:
    """Median known depth over the grid cells covering an image rectangle.

    Args:
        depth: Depth map.
        x0: Left edge in image pixels.
        y0: Top edge in image pixels.
        x1: Right edge in image pixels.
        y1: Bottom edge in image pixels.

    Returns:
        Depth in metres, or None when no covered cell has a known depth.
    """
    c0, r0 = _to_grid(depth, x0, y0)
    c1, r1 = _to_grid(depth, x1, y1)
    window = depth.depth_m[r0:r1 + 1, c0:c1 + 1]
    known = window[np.isfinite(window) & (window > 0)]
    return float(np.median(known)) if known.size else None


def _endpoint_depth(depth: DepthMap, u: float, v: float, reference: float) -> float:
    """Depth at an axis endpoint, replaced by the book's reference depth when it looks like background.

    Args:
        depth: Depth map.
        u: Horizontal image coordinate in pixels.
        v: Vertical image coordinate in pixels.
        reference: Median depth of the box interior.

    Returns:
        Depth in metres.
    """
    half = CONFIG.measurement.depth_window_px / 2
    z = _median_depth(depth, u - half, v - half, u + half, v + half)
    # A jump away from the book's own depth means the sample hit the wall or shelf behind it.
    if z is None or abs(z - reference) > CONFIG.measurement.depth_tolerance * reference:
        return reference
    return z


def _axis_length_m(depth: DepthMap, start: tuple[float, float], end: tuple[float, float], reference: float) -> float:
    """Back-project two image points and return their 3-D distance.

    Args:
        depth: Depth map with intrinsics.
        start: First point (u, v) in image pixels.
        end: Second point (u, v) in image pixels.
        reference: Median depth of the box interior.

    Returns:
        Distance in metres.
    """
    points = []
    for u, v in (start, end):
        z = _endpoint_depth(depth, u, v, reference)
        points.append(np.array([(u - depth.cx) * z / depth.fx, (v - depth.cy) * z / depth.fy, z]))
    return float(np.linalg.norm(points[0] - points[1]))


def spine_refusal(bbox: Box, view: str | None, issues: list[str], depth: DepthMap | None) -> str | None:
    """Name why a detection cannot be measured, or None when it can be.

    A book left unmeasured should say which evidence was missing, rather than
    reporting that no depth reached the frame when depth did reach it.

    Args:
        bbox: Normalized detection box in the source image.
        view: Visible view reported for the detection.
        issues: Readability issues reported for the detection.
        depth: Metric depth for the same image, or None when none was captured.

    Returns:
        The reason a measurement was not taken, or None when one can be.
    """
    if depth is None:
        return 'No depth was captured for this frame'
    if view != 'spine':
        return f'Measured from spines only; this sighting shows the {view or "book"}'
    blocking = {'cut_off', 'occlusion'} & set(issues)
    if blocking:
        return f'Spine is {" and ".join(sorted(blocking)).replace("cut_off", "cut off")} in this sighting'
    x0, x1 = bbox.x_min * depth.image_width, bbox.x_max * depth.image_width
    y0, y1 = bbox.y_min * depth.image_height, bbox.y_max * depth.image_height
    cu, cv = (x0 + x1) / 2, (y0 + y1) / 2
    if _median_depth(depth, cu - (x1 - x0) / 4, cv - (y1 - y0) / 4,
                     cu + (x1 - x0) / 4, cv + (y1 - y0) / 4) is None:
        return 'Depth sensing returned nothing where the spine was seen'
    return None


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
    if spine_refusal(bbox, view, issues, depth) is not None:
        return None
    x0, x1 = bbox.x_min * depth.image_width, bbox.x_max * depth.image_width
    y0, y1 = bbox.y_min * depth.image_height, bbox.y_max * depth.image_height
    cu, cv = (x0 + x1) / 2, (y0 + y1) / 2
    keep = 1 - 2 * CONFIG.measurement.depth_inset
    vertical = ((cu, cv - (y1 - y0) / 2 * keep), (cu, cv + (y1 - y0) / 2 * keep))
    horizontal = ((cu - (x1 - x0) / 2 * keep, cv), (cu + (x1 - x0) / 2 * keep, cv))
    long_axis, short_axis = (vertical, horizontal) if y1 - y0 >= x1 - x0 else (horizontal, vertical)
    # The inner half of the box is the most reliable sample of the book's own depth.
    reference = _median_depth(depth, cu - (x1 - x0) / 4, cv - (y1 - y0) / 4, cu + (x1 - x0) / 4, cv + (y1 - y0) / 4)
    if reference is None:
        return None
    height = _axis_length_m(depth, *long_axis, reference) / keep * 100
    thickness = _axis_length_m(depth, *short_axis, reference) / keep * 100
    low_h, high_h = CONFIG.measurement.height_range_cm
    low_t, high_t = CONFIG.measurement.thickness_range_cm
    if not (low_h <= height <= high_h and low_t <= thickness <= high_t):
        # Implausible sizes mean the depth sampled past the book, onto whatever is behind it.
        return None
    return SpineMeasurement(height_cm=height, thickness_cm=thickness, method=depth.method)


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


def size_refusal(bbox: Box, view: str | None, issues: list[str], depth: DepthMap | None) -> str | None:
    """Name why a detection produced no size, including an implausible reading.

    Args:
        bbox: Normalized detection box in the source image.
        view: Visible view reported for the detection.
        issues: Readability issues reported for the detection.
        depth: Metric depth for the same image, or None when none was captured.

    Returns:
        The reason no size was recorded, or None when one was.
    """
    refusal = spine_refusal(bbox, view, issues, depth)
    if refusal is not None:
        return refusal
    if measure_spine(bbox, view, issues, depth) is not None:
        return None
    return 'Depth gave an implausible size, so it sampled past the spine onto what is behind it'
