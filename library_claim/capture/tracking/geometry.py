"""Verify planar mappings and transform normalized image polygons."""
import cv2
import numpy as np

from library_claim.config import CONFIG

SETTINGS = CONFIG.tracking


def transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Project points through a homography while rejecting its horizon.

    Args:
        points: An N-by-2 array of image coordinates.
        matrix: Finite invertible 3-by-3 homography.

    Returns:
        Projected N-by-2 coordinates.

    Raises:
        ValueError: The mapping is singular, nonfinite, or crosses a horizon.
    """
    matrix = np.asarray(matrix, dtype=float)
    points = np.asarray(points, dtype=float)
    if matrix.shape != (3, 3) or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('Expected a 3x3 matrix and Nx2 points')
    if not np.isfinite(matrix).all() or not np.isfinite(points).all() or abs(np.linalg.det(matrix)) < 1e-10:
        raise ValueError('Invalid homography')
    projected = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    denominator = projected[:, 2]
    if np.any(abs(denominator) < 1e-8) or (np.any(denominator < 0) and np.any(denominator > 0)):
        raise ValueError('Mapping crosses the projective horizon')
    return projected[:, :2] / denominator[:, None]


def corners(shape: tuple) -> np.ndarray:
    """Return clockwise image corners in pixel coordinates.

    Args:
        shape: Image array shape beginning with height and width.

    Returns:
        Four corner coordinates.
    """
    h, w = shape[:2]
    return np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])


def verify(source: np.ndarray, destination: np.ndarray, source_shape: tuple,
           destination_shape: tuple, *, robust: bool) -> tuple | None:
    """Fit a planar mapping and reject weak or implausible geometric support.

    Args:
        source: Matched points in the current image.
        destination: Matching points in the reference image.
        source_shape: Current image shape.
        destination_shape: Reference image shape.
        robust: Whether to use bounded RANSAC for recovery.

    Returns:
        Matrix, inlier count, and median reprojection error, or None on failure.
    """
    source = np.asarray(source, dtype=np.float32).reshape(-1, 2)
    destination = np.asarray(destination, dtype=np.float32).reshape(-1, 2)
    if len(source) < SETTINGS.min_matches or len(source) != len(destination):
        return None
    matrix, mask = cv2.findHomography(source, destination, cv2.RANSAC if robust else 0,
                                      SETTINGS.reprojection_threshold_px, maxIters=SETTINGS.ransac_iterations,
                                      confidence=SETTINGS.ransac_confidence)
    if matrix is None:
        return None
    try:
        errors = np.linalg.norm(transform(source, matrix) - destination, axis=1)
        good = (errors <= SETTINGS.reprojection_threshold_px) & mask.ravel().astype(bool)
        if good.sum() < SETTINGS.min_matches or good.mean() < SETTINGS.min_inlier_fraction:
            return None
        # Matches confined to one repeated label cannot establish the whole view.
        for points, shape in ((source[good], source_shape), (destination[good], destination_shape)):
            area = cv2.contourArea(cv2.convexHull(points))
            if area / (shape[0] * shape[1]) < SETTINGS.min_support_area:
                return None
        polygon = transform(corners(source_shape), matrix).astype(np.float32)
        ratio = cv2.contourArea(polygon, oriented=True) / (source_shape[0] * source_shape[1])
        if not cv2.isContourConvex(polygon) or not SETTINGS.min_projected_area_ratio < ratio < SETTINGS.max_projected_area_ratio:
            return None
    except (ValueError, cv2.error):
        return None
    return matrix, int(good.sum()), float(np.median(errors[good]))


def normalized_mapping(matrix: np.ndarray, source_shape: tuple, reference_shape: tuple) -> np.ndarray:
    """Convert a pixel mapping to normalized image coordinates.

    Args:
        matrix: Current-to-reference homography in assessed pixels.
        source_shape: Current assessed image shape.
        reference_shape: Reference assessed image shape.

    Returns:
        A mapping usable with full-resolution normalized polygons.
    """
    source_scale = np.diag([source_shape[1] - 1, source_shape[0] - 1, 1])
    reference_scale = np.diag([1 / (reference_shape[1] - 1), 1 / (reference_shape[0] - 1), 1])
    return reference_scale @ matrix @ source_scale
