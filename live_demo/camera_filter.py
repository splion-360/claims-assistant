"""Admit fresh, Drex-checked book frames to bounded remote image inference."""
from dataclasses import dataclass

from library_claim.config import CONFIG
from library_claim.decisions.frame_check import FrameCheck, is_book_candidate


@dataclass(frozen=True)
class CameraCandidate:
    """Keep approved image bytes tied to their capture identity and activation."""

    data: bytes
    frame_id: str
    timestamp: float
    revision: int


class CameraFilter:
    """Keep at most one pending book frame while enforcing remote cadence."""

    def __init__(self):
        """Initialize a filter for one camera activation."""
        self.pending = None
        self.last_attempt_at = float('-inf')
        self.last_success_data = None
        self.reason = 'waiting_for_assessment'
        self.submitted = 0
        self.rejected = 0

    def offer(self, candidate: CameraCandidate, check: FrameCheck | None) -> None:
        """Retain only a frame in which Drex saw a book.

        Args:
            candidate: Exact frame corresponding to the check.
            check: Same-frame Drex result, or None for unchecked views.
        """
        if check is None:
            self.reason = 'not_sampled_or_no_new_evidence'
            return
        if not is_book_candidate(check):
            self.reason = 'below_book_threshold'
            self.rejected += 1
            return
        if candidate.data == self.last_success_data:
            self.reason = 'duplicate_sent_frame'
            return
        self.pending = candidate
        self.reason = 'book_candidate'

    def take(self, now: float, revision: int, *, busy: bool, at_limit: bool) -> CameraCandidate | None:
        """Release one fresh candidate when the remote slot and rate limit allow.

        Args:
            now: Current monotonic time.
            revision: Current camera activation revision.
            busy: Whether a remote capture is already running.
            at_limit: Whether the session has reached its evidence-photo limit.

        Returns:
            Approved immutable candidate, or None when no request should start.
        """
        candidate = self.pending
        if candidate is None:
            return None
        if candidate.revision != revision or now - candidate.timestamp > CONFIG.capture.frame_max_age_s:
            self.pending = None
            self.reason = 'stale_candidate'
            return None
        if candidate.data == self.last_success_data:
            self.pending = None
            self.reason = 'duplicate_sent_frame'
            return None
        if at_limit or busy or now - self.last_attempt_at < CONFIG.capture.remote_interval_s:
            self.reason = 'photo_limit' if at_limit else 'remote_busy' if busy else 'rate_limited'
            return None
        self.pending = None
        self.last_attempt_at = now
        self.reason = 'submitted_book_frame'
        self.submitted += 1
        return candidate

    def finish(self, candidate: CameraCandidate, success: bool) -> None:
        """Remember successful bytes or retain a failed candidate for bounded retry.

        Args:
            candidate: The completed request's exact frame.
            success: Whether remote analysis successfully pinned evidence.
        """
        if success:
            self.last_success_data = candidate.data
        elif self.pending is None:
            # A newer candidate takes precedence over retrying an older failure.
            self.pending = candidate
