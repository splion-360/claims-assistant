"""Track live camera views and persist selected evidence with quality metadata."""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from library_claim.capture.quality import QualityAssessment, QualityAssessor
from library_claim.capture.selection import compare_evidence
from library_claim.capture.tracking.tracker import FrameTracker, TrackingResult
from library_claim.storage.live_capture import LiveCaptureStore
from live_demo.camera_quality import logger
from live_demo.camera_logging import CameraSummary
from live_demo.camera_filter import CameraFilter
from live_demo.camera_gate import local_gate
from library_claim.config import CONFIG


@dataclass(frozen=True)
class PendingCheck:
    """Keep a frame awaiting its Drex check together with what is needed to save it."""
    jpeg: bytes
    frame_id: str
    timestamp: float
    captured_at: datetime
    result: TrackingResult
    quality: QualityAssessment
    width: int
    height: int
    saved: bool


class CameraTracking:
    """Own tracking, sampled quality, and retained evidence for one camera activation."""

    def __init__(self, store: LiveCaptureStore | None = None):
        """Initialize bounded stream state and optional persistence.

        Args:
            store: Persistent destination for selected frames from this stream.
        """
        self.tracker = FrameTracker()
        self.store = store
        self.requested_capture = False
        self.pending_check: PendingCheck | None = None
        self.gate_failure: str | None = None
        self.latest_check = None
        self.latest_check_at = None
        self.summary = CameraSummary()
        self.remote_filter = CameraFilter()
        self.summary.remote_filter = self.remote_filter
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
        if timestamp - self.last_quality_at >= CONFIG.capture.quality_interval_s:
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
        # An explicit recapture may revisit a known view, but still must pass local gates.
        if self.requested_capture and quality is not None:
            reason = reason or 'requested_capture'
        if self.store is not None and reason is not None:
            try:
                self.saved_path = self.store.save(jpeg, result, captured_at, image.shape[1], image.shape[0], reason, quality)
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

        # Frames that would need semantic inspection go to Drex unless the local gate rejects them.
        # A plain wall may not track, so uncertain views are checked too; they are saved only if Drex sees books.
        self.pending_check, self.gate_failure = None, None
        if quality is not None and (reason is not None or result.status == 'uncertain'):
            self.gate_failure = local_gate(quality)
            if self.gate_failure is None:
                self.pending_check = PendingCheck(jpeg, frame_id, timestamp, captured_at, result, quality,
                                                  image.shape[1], image.shape[0], saved=self.saved_path is not None)

        # Disk evidence survives eviction, while comparison state remains bounded.
        live_ids = {view.frame_id for view in self.tracker.keyframes.views}
        self._retained = {key: value for key, value in self._retained.items() if key in live_ids}
        self._saved_references.intersection_update(live_ids)
        return result

    def save_book_candidate(self, pending: PendingCheck) -> None:
        """Save a held frame after Drex confirmed it shows books.

        Args:
            pending: Frame held by `assess` and not yet saved.

        Raises:
            OSError: Evidence cannot be written.
            sqlite3.Error: Records cannot be committed.
        """
        if self.store is not None and not pending.saved:
            self.store.save(pending.jpeg, pending.result, pending.captured_at, pending.width, pending.height,
                            'book_candidate', pending.quality)

    def log(self, result: TrackingResult, timestamp: float) -> None:
        """Log detailed diagnostics at debug level and a periodic camera table.

        Args:
            result: Completed tracking result.
            timestamp: Monotonic capture arrival time.
        """
        if self.saved_path is not None:
            logger.debug('Evidence saved: %s | frame=%s | reason=%s',
                         self.saved_path, result.frame_id, self.saved_reason)
        logger.debug('camera_tracking %s', result.model_dump_json())
        self.summary.log(result, timestamp, quality=self.latest_quality,
                         quality_at=self.last_quality_at, check=self.latest_check, check_at=self.latest_check_at,
                         saved=self.saved_path is not None)
