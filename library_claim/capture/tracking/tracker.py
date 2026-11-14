"""Track planar camera views and recover reference coordinates after interruptions."""
from time import perf_counter
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

from .geometry import corners, intersection_area, normalized_mapping, transform, verify
from .keyframes import Keyframes
from library_claim.config import CONFIG

SETTINGS = CONFIG.tracking


class TrackingResult(BaseModel):
    """Report visual association without asserting physical book identity."""

    model_config = ConfigDict(extra='forbid', frozen=True)
    frame_id: str
    status: Literal['tracked', 'recovered', 'reference_created', 'uncertain']
    reference_frame_id: str | None = None
    current_to_reference: list[list[float]] | None = None
    reference_overlap: float | None = None
    matched_points: int = 0
    reprojection_error_px: float | None = None
    elapsed_ms: float

    def map_polygon(self, points: list[tuple[float, float]]) -> list[list[float]]:
        """Map a normalized current-frame polygon into its reference view.

        Args:
            points: Polygon corners in normalized current-image coordinates.

        Returns:
            Reference coordinates, possibly outside zero to one.

        Raises:
            ValueError: No mapping exists or the projective mapping is invalid.
        """
        if self.current_to_reference is None:
            raise ValueError('No verified reference mapping')
        return transform(np.asarray(points), np.asarray(self.current_to_reference)).tolist()


class FrameTracker:
    """Maintain short-term continuity and bounded session-local reference views."""

    def __init__(self, *, capacity: int = SETTINGS.capacity, reanchor_every: int = SETTINGS.reanchor_every):
        """Configure keyframe memory and periodic recovery verification.

        Args:
            capacity: Maximum saved views.
            reanchor_every: Maximum optical-flow steps before direct SIFT verification.

        Raises:
            ValueError: Reanchor interval is not positive.
        """
        if reanchor_every < 1:
            raise ValueError('Reanchor interval must be positive')
        self.keyframes = Keyframes(capacity)
        self.reanchor_every = reanchor_every
        self.previous = None
        self.points = None
        self.reference = None
        self.mapping = None
        self.steps = 0
        self.timestamp = None

    @staticmethod
    def _flow_options() -> dict:
        """Build matching forward and backward Lucas-Kanade parameters.

        Returns:
            OpenCV keyword arguments from the shared tracking configuration.
        """
        return dict(winSize=(SETTINGS.lk_window, SETTINGS.lk_window), maxLevel=SETTINGS.lk_max_level,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                              SETTINGS.lk_iterations, SETTINGS.lk_epsilon),
                    minEigThreshold=SETTINGS.lk_min_eigen_threshold)

    def _flow(self, gray: np.ndarray) -> tuple | None:
        """Follow corners bidirectionally and verify their planar arrangement.

        Args:
            gray: Current grayscale image.

        Returns:
            Current-to-previous mapping and diagnostics, or None on failure.
        """
        if self.previous is None or self.points is None or len(self.points) < SETTINGS.min_matches:
            return None
        forward, status, _ = cv2.calcOpticalFlowPyrLK(self.previous, gray, self.points, None, **self._flow_options())
        if forward is None:
            return None
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self.previous, forward, None, **self._flow_options())
        if backward is None:
            return None
        good = status.ravel().astype(bool) & back_status.ravel().astype(bool)
        good &= np.linalg.norm(backward - self.points, axis=2).ravel() < SETTINGS.forward_backward_error_px
        good &= np.isfinite(forward).all(axis=(1, 2))
        current = forward.reshape(-1, 2)
        good &= (current[:, 0] >= 0) & (current[:, 0] < gray.shape[1])
        good &= (current[:, 1] >= 0) & (current[:, 1] < gray.shape[0])
        return verify(current[good], self.points.reshape(-1, 2)[good],
                      gray.shape, self.previous.shape, robust=False)

    def update(self, image: np.ndarray, frame_id: str, timestamp: float) -> TrackingResult:
        """Associate a frame with a reference or preserve uncertainty.

        Args:
            image: Decoded grayscale or BGR uint8 image.
            frame_id: Identifier of the current capture.
            timestamp: Strictly increasing monotonic capture time in seconds.

        Returns:
            Association diagnostics and normalized perspective mapping.

        Raises:
            ValueError: Image, identifier, or timestamp is invalid.
        """
        start = perf_counter()
        if not frame_id or not np.isfinite(timestamp) or (self.timestamp is not None and timestamp <= self.timestamp):
            raise ValueError('Expected a frame ID and increasing finite timestamp')
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim not in (2, 3):
            raise ValueError('Expected a uint8 grayscale or BGR image')
        if min(image.shape[:2]) < 32 or (image.ndim == 3 and image.shape[2] != 3):
            raise ValueError('Image is too small or has unsupported channels')
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        scale = min(1, SETTINGS.max_dimension / max(gray.shape))
        if scale < 1:
            gray = cv2.resize(gray, (round(gray.shape[1] * scale), round(gray.shape[0] * scale)))
        if min(gray.shape) < 32:
            raise ValueError('Image aspect ratio is unsupported')
        gap = self.timestamp is None or timestamp - self.timestamp > SETTINGS.max_gap_s
        self.timestamp = timestamp
        status, count, error = 'uncertain', 0, None
        matrix = None
        if not gap and self.previous is not None and gray.shape == self.previous.shape and self.steps < self.reanchor_every:
            flow = self._flow(gray)
            if flow is not None and self.mapping is not None:
                relative, count, error = flow
                matrix = self.mapping @ relative
                status = 'tracked'
                self.steps += 1

        # Direct matching anchors drift and handles view returns and large motion.
        if matrix is None:
            points, descriptors = self.keyframes.describe(gray)
            verified = []
            for view, matches in self.keyframes.candidates(descriptors):
                source = points[[m.queryIdx for m in matches]]
                destination = view.points[[m.trainIdx for m in matches]]
                fitted = verify(source, destination, gray.shape, view.gray.shape, robust=True)
                if fitted is not None:
                    verified.append((view, fitted))
            if len(verified) == 1:
                self.reference, (matrix, count, error) = verified[0]
                status = 'recovered'
            elif not verified and descriptors is not None and len(points) >= SETTINGS.min_matches:
                self.reference = self.keyframes.add(frame_id, gray, points, descriptors)
                matrix, status = np.eye(3), 'reference_created'
            # Multiple verified references remain ambiguous rather than picking an identity.
            self.steps = 0

        mapping, overlap, reference_id = None, None, None
        if matrix is not None:
            try:
                normalized = normalized_mapping(matrix, gray.shape, self.reference.gray.shape)
                polygon = transform(corners(gray.shape), matrix).astype(np.float32)
                reference_corners = corners(self.reference.gray.shape)
                if not cv2.isContourConvex(polygon) or cv2.contourArea(polygon, oriented=True) <= 0:
                    raise ValueError('Invalid mapped view')
                area = intersection_area(polygon, reference_corners)
                overlap = float(np.clip(area / cv2.contourArea(polygon), 0, 1))
                if overlap < SETTINGS.min_reference_overlap:
                    raise ValueError('Insufficient reference overlap')
                mapping, reference_id = normalized.tolist(), self.reference.frame_id
            except (ValueError, cv2.error):
                matrix, status, overlap = None, 'uncertain', None
        self.mapping = matrix
        self.previous = gray
        # Replenish local corners instead of letting the tracked set shrink forever.
        self.points = cv2.goodFeaturesToTrack(gray, maxCorners=SETTINGS.max_corners, qualityLevel=SETTINGS.corner_quality,
                                              minDistance=SETTINGS.corner_min_distance_px)
        return TrackingResult(frame_id=frame_id, status=status, reference_frame_id=reference_id,
                              current_to_reference=mapping, reference_overlap=overlap,
                              matched_points=count, reprojection_error_px=error,
                              elapsed_ms=(perf_counter() - start) * 1000)
