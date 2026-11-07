"""Track live camera views and persist selected evidence with quality metadata."""
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from library_claim.capture.quality import QualityAssessor
from library_claim.capture.selection import compare_evidence
from library_claim.capture.tracking.tracker import FrameTracker, TrackingResult
from library_claim.storage.live_capture import LiveCaptureStore
from live_demo.camera_quality import logger
from live_demo.camera_relevance import CameraRelevance


class CameraTracking:
    """Own tracking, sampled quality, and retained evidence for one camera activation."""

    def __init__(self, store: LiveCaptureStore | None = None,
                 relevance: CameraRelevance | None = None):
        """Initialize bounded stream state and optional persistence.

        Args:
            store: Persistent destination for selected frames from this stream.
            relevance: Optional local classifier used for diagnostics only.
        """
        self.tracker = FrameTracker()
        self.store = store
        self.relevance = relevance
        self.latest_relevance = None
        self.relevance_frame_id = None
        self.last_log_at = float('-inf')
        self.last_quality_at = float('-inf')
        self.latest_quality = None
        self.quality_frame_id = None
        self._assessor = None
        self._previous_quality_image = None
        self._retained = {}
        self._saved_references = set()
        self.saved_path: Path | None = None
        self.saved_reason = None

    def assess(self, jpeg: bytes, frame_id: str, timestamp: float) -> TrackingResult:
        """Track a frame and save selected evidence on the same background worker.

        Args:
            jpeg: Encoded camera image.
            frame_id: Identifier of the current capture.
            timestamp: Monotonic arrival time.

        Returns:
            Tracking diagnostics and any verified reference mapping.

        Raises:
            ValueError: The JPEG or tracking input is invalid.
            OSError: Selected evidence cannot be written.
            sqlite3.Error: Selected evidence cannot be committed.
        """
        captured_at = datetime.now(timezone.utc)
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Could not decode tracking frame')
        result = self.tracker.update(image, frame_id, timestamp)
        self.saved_path, self.saved_reason = None, None
        quality = None
        if timestamp - self.last_quality_at >= 1:
            if self._assessor is None:
                self._assessor = QualityAssessor()
            previous = self._previous_quality_image
            if previous is not None and previous.shape != image.shape:
                previous = None
            quality = self._assessor.assess(image, previous=previous)
            self.latest_quality, self.quality_frame_id = quality, frame_id
            self._previous_quality_image = image
            self.last_quality_at = timestamp

        # Persist every new reference before later views can cite its mapping.
        reason = 'tracking_reference' if result.status == 'reference_created' else None
        if reason is None and quality is not None and result.reference_frame_id in self._saved_references:
            previous_result, previous_quality = self._retained.get(result.reference_frame_id, (None, None))
            decision = compare_evidence(result, quality, previous_result, previous_quality)
            if previous_quality is None and result.status in ('tracked', 'recovered'):
                reason = 'quality_baseline'
            elif decision.action == 'analyze_candidate':
                reason = decision.reason
        # Unverified views still need semantic inspection; a plain wall may not track.
        self.latest_relevance, self.relevance_frame_id = None, None
        if self.relevance is not None and quality is not None and (reason is not None or result.status == 'uncertain'):
            self.latest_relevance = self.relevance.assess(image, quality)
            self.relevance_frame_id = frame_id
        if self.store is not None and reason is not None:
            try:
                self.saved_path = self.store.save(jpeg, result, captured_at, image.shape[1], image.shape[0], reason, quality,
                                                  relevance=self.latest_relevance.model_dump(mode='json') if self.latest_relevance else None)
            except Exception:
                # Reacquire after a failed save; future mappings must not cite lost evidence.
                self.tracker = FrameTracker()
                self._retained.clear()
                self._saved_references.clear()
                raise
            self.saved_reason = reason
            if result.status == 'reference_created':
                self._saved_references.add(frame_id)
            self._retained[result.reference_frame_id] = (result, quality)

        # Disk evidence survives eviction, while comparison state remains bounded.
        live_ids = {view.frame_id for view in self.tracker.keyframes.views}
        self._retained = {key: value for key, value in self._retained.items() if key in live_ids}
        self._saved_references.intersection_update(live_ids)
        return result

    def log(self, result: TrackingResult, timestamp: float) -> None:
        """Log selected evidence locations and a rate-limited tracking summary.

        Args:
            result: Completed tracking result.
            timestamp: Monotonic capture arrival time.
        """
        if self.latest_relevance is not None and self.relevance_frame_id == result.frame_id:
            assessment = self.latest_relevance
            scores = ' '.join(f'{key}={value:.3f}' for key, value in assessment.scores.items())
            logger.info('Camera relevance: %s | status=%s | %s | %.1f ms | reason=%s | frame=%s',
                        assessment.label, assessment.status, scores or 'scores=N/A',
                        assessment.elapsed_ms, assessment.reason, result.frame_id)
        if self.saved_path is not None:
            logger.info('Evidence saved: %s | frame=%s | reason=%s',
                        self.saved_path, result.frame_id, self.saved_reason)
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
