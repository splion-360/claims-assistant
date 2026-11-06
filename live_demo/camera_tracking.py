"""Decode live camera views and expose bounded tracking diagnostics."""
import cv2
import numpy as np

from library_claim.capture.tracking.tracker import FrameTracker, TrackingResult
from live_demo.camera_quality import logger


class CameraTracking:
    """Own one tracker for one camera activation."""

    def __init__(self):
        """Initialize a bounded tracker and diagnostic log cadence."""
        self.tracker = FrameTracker()
        self.last_log_at = float('-inf')

    def assess(self, jpeg: bytes, frame_id: str, timestamp: float) -> TrackingResult:
        """Decode and associate one captured view on a background worker.

        Args:
            jpeg: Encoded image received from the camera.
            frame_id: Current capture identifier.
            timestamp: Monotonic arrival time.

        Returns:
            Tracking diagnostics and any verified reference mapping.

        Raises:
            ValueError: The JPEG cannot be decoded or the frame is invalid.
        """
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Could not decode tracking frame')
        return self.tracker.update(image, frame_id, timestamp)

    def log(self, result: TrackingResult, timestamp: float) -> None:
        """Log a concise tracking update at most once per second.

        Args:
            result: Completed tracking result.
            timestamp: Monotonic capture arrival time.
        """
        if timestamp - self.last_log_at < 1:
            return
        self.last_log_at = timestamp
        # A new reference maps to itself; that is not a recovered-view match.
        if result.status == 'reference_created':
            overlap = 'N/A — new reference'
        else:
            overlap = 'unknown' if result.reference_overlap is None else f'{result.reference_overlap:.0%}'
        logger.info('Camera tracking: %s | matches=%d | reference overlap=%s | %.1f ms',
                    result.status, result.matched_points, overlap, result.elapsed_ms)
        logger.debug('camera_tracking %s', result.model_dump_json())
