"""Compare partial book readings and geometrically consistent crop features."""
import re

import cv2
import numpy as np

from library_claim.config import CONFIG

SETTINGS = CONFIG.identification


def normalize(value: str) -> str:
    """Normalize case, punctuation, and spacing without deleting title words.

    Args:
        value: Literal visual reading.

    Returns:
        Normalized words separated by spaces.
    """
    return ' '.join(re.findall(r'\w+', value.casefold()))


def text_relation(first: str | None, second: str | None) -> str:
    """Classify readings as exact, partial, fuzzy, conflicting, or unknown.

    Args:
        first: Earlier reading, possibly absent.
        second: Current reading, possibly absent.

    Returns:
        Compatibility category, never a probability or identity assertion.
    """
    a, b = normalize(first or ''), normalize(second or '')
    if not a or not b:
        return 'unknown'
    if a == b:
        return 'exact'
    # Contiguous ordered words tolerate missing leading/trailing title portions.
    if f' {a} ' in f' {b} ' or f' {b} ' in f' {a} ' or a.startswith(b) or b.startswith(a):
        return 'partial' if min(len(a), len(b)) >= SETTINGS.text_min_characters else 'unknown'
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    similarity = 1 - previous[-1] / max(len(a), len(b))
    return 'fuzzy' if similarity >= SETTINGS.text_edit_similarity else 'conflicting'


def preferred_reading(values: list[str]) -> str | None:
    """Prefer a fuller compatible reading without inventing text from fragments.

    Args:
        values: Non-null readings in capture order.

    Returns:
        A recorded reading, or None if readings substantially conflict.
    """
    if not values:
        return None
    if any(text_relation(a, b) == 'conflicting' for a in values for b in values):
        return None
    # Fuzzy disagreements do not justify preferring a longer spelling error.
    exact_counts = {normalize(v): sum(normalize(other) == normalize(v) for other in values) for v in values}
    candidates = [v for v in values if all(text_relation(v, other) in ('exact', 'partial') for other in values)]
    return max(candidates, key=lambda v: len(normalize(v))) if candidates else max(values, key=lambda v: exact_counts[normalize(v)])


def crop_features(image: np.ndarray, box) -> tuple:
    """Extract bounded SIFT features from a book's visible crop.

    Args:
        image: Decoded source image.
        box: Normalized book box.

    Returns:
        Keypoint coordinates, descriptors, and crop dimensions.
    """
    h, w = image.shape[:2]
    crop = image[int(box.y_min*h):int(np.ceil(box.y_max*h)), int(box.x_min*w):int(np.ceil(box.x_max*w))]
    if not crop.size or min(crop.shape[:2]) < SETTINGS.visual_min_crop_size:
        return np.empty((0, 2), np.float32), None, (0, 0)
    scale = min(1, SETTINGS.visual_max_dimension / max(crop.shape[:2]))
    if scale < 1:
        crop = cv2.resize(crop, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    points, descriptors = cv2.SIFT_create(nfeatures=SETTINGS.visual_max_features).detectAndCompute(gray, None)
    return np.float32([point.pt for point in points]).reshape(-1, 2), descriptors, gray.shape


def visual_match(first: tuple, second: tuple) -> float | None:
    """Verify distinctive crop correspondences with bounded RANSAC.

    Args:
        first: Current crop features from crop_features.
        second: Previously observed crop features.

    Returns:
        Accepted inlier fraction, or None when visual identity is unsupported.
    """
    a, da, shape_a = first
    b, db, shape_b = second
    if da is None or db is None or min(len(da), len(db)) < SETTINGS.visual_min_matches:
        return None
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
    matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < SETTINGS.visual_ratio * pair[1].distance]
    # Repeated lettering must not map multiple source points to one destination.
    unique = {match.trainIdx: match for match in sorted(matches, key=lambda m: m.distance, reverse=True)}
    matches = list(unique.values())
    if len(matches) < SETTINGS.visual_min_matches:
        return None
    source = np.float32([a[m.queryIdx] for m in matches])
    target = np.float32([b[m.trainIdx] for m in matches])
    matrix, mask = cv2.findHomography(source, target, cv2.RANSAC, SETTINGS.visual_reprojection_px,
                                     maxIters=SETTINGS.visual_ransac_iterations,
                                     confidence=SETTINGS.visual_ransac_confidence)
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        return None
    inliers = mask.ravel().astype(bool)
    if inliers.sum() < SETTINGS.visual_min_matches or inliers.mean() < SETTINGS.visual_min_inlier_fraction:
        return None
    for points, shape in ((source[inliers], shape_a), (target[inliers], shape_b)):
        if cv2.contourArea(cv2.convexHull(points)) / (shape[0] * shape[1]) < SETTINGS.visual_min_support:
            return None
    return float(inliers.mean())
