"""Decode camera frames and log local image-quality measurements."""
import logging

import cv2
import numpy as np

from library_claim.capture.quality import QualityAssessment, QualityAssessor

# Inherit Uvicorn's console handler so INFO scores appear with --log-level info.
logger = logging.getLogger('uvicorn.error.camera_quality')


class CameraQuality:
    """Keep one model and previous frame for a single camera stream."""

    def __init__(self):
        """Initialize stream state without loading the model on the event loop."""
        self._assessor = None
        self._previous = None

    def assess(self, jpeg: bytes) -> QualityAssessment:
        """Decode and score a frame on a background worker.

        Args:
            jpeg: JPEG bytes received from the browser camera.

        Returns:
            Quality scores for the decoded frame.

        Raises:
            ValueError: The image cannot be decoded or assessed.
            RuntimeError: The pretrained model cannot be loaded.
        """
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Could not decode camera frame')
        if self._assessor is None:
            self._assessor = QualityAssessor()
        previous = self._previous
        if previous is not None and previous.shape != image.shape:
            previous = None
        result = self._assessor.assess(image, previous=previous)
        self._previous = image
        return result


def log_scores(session_id: str, frame_id: str, result: QualityAssessment) -> None:
    """Log exact-frame quality diagnostics at debug level.

    Args:
        session_id: Owning browser session identifier.
        frame_id: Identifier of the scored capture.
        result: Local assessment including timing and unavailable signals.
    """
    logger.debug('camera_quality session=%s frame=%s scores=%s',
                 session_id, frame_id, result.model_dump_json())
