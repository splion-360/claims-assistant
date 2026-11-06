"""Persist selected live camera evidence and its verified frame associations."""
from datetime import datetime
from pathlib import Path

from library_claim.capture.quality import QualityAssessment
from library_claim.capture.tracking.tracker import TrackingResult
from library_claim.domain.models import Frame, FrameAssociation, Sweep
from .evidence import EvidenceStore
from .repository import Repository


class LiveCaptureStore:
    """Save selected frames under one sweep using thread-owned connections."""

    def __init__(self, root: str | Path, sweep_id: str):
        """Configure storage without performing filesystem work.

        Args:
            root: Application-owned directory for SQLite and image evidence.
            sweep_id: Browser session identifier used as the persistent sweep ID.
        """
        self.root = Path(root).resolve()
        self.database = self.root / 'claims.sqlite3'
        self.sweep_id = sweep_id

    def save(self, jpeg: bytes, result: TrackingResult, captured_at: datetime,
             width: int, height: int, reason: str,
             quality: QualityAssessment | None = None) -> Path:
        """Save an image and its mapping with traceable metadata.

        Args:
            jpeg: Original encoded camera image.
            result: Tracking result for that exact capture.
            captured_at: Timezone-aware wall-clock capture time.
            width: Decoded source width in pixels.
            height: Decoded source height in pixels.
            reason: Selection reason for retaining the view.
            quality: Assessment of this exact frame, if measured.

        Returns:
            Absolute path of the saved image.

        Raises:
            ValueError: A mapping reference is missing or belongs to another sweep.
            OSError: Evidence cannot be written.
            sqlite3.Error: Records cannot be persisted.
        """
        files = EvidenceStore(self.root / 'frames')
        with Repository(self.database) as repo:
            records = []
            try:
                repo.get(Sweep, self.sweep_id)
            except KeyError:
                records.append(Sweep(id=self.sweep_id, captured_at=captured_at,
                                     device='browser camera'))
            association = None
            if result.status in ('tracked', 'recovered'):
                if result.current_to_reference is None or result.reference_frame_id is None:
                    raise ValueError('Verified mapping is missing its reference')
                try:
                    reference = repo.get(Frame, result.reference_frame_id)
                except KeyError as error:
                    raise ValueError('Save reference evidence before its associations') from error
                if reference.sweep_id != self.sweep_id:
                    raise ValueError('Reference belongs to a different sweep')
                association = FrameAssociation(
                    id=result.frame_id, sweep_id=self.sweep_id, frame_id=result.frame_id,
                    reference_frame_id=result.reference_frame_id,
                    matrix=result.current_to_reference,
                    method='optical_flow' if result.status == 'tracked' else 'sift_ransac',
                    matched_points=result.matched_points,
                    reprojection_error_px=result.reprojection_error_px,
                )
            # Write the file first so committed frame references are always openable.
            key = files.save(jpeg, 'image/jpeg')
            frame = Frame(id=result.frame_id, sweep_id=self.sweep_id, captured_at=captured_at,
                          evidence_key=key, width_px=width, height_px=height,
                          selection_reason=reason, quality=quality.signals if quality else None,
                          quality_metadata=quality.model_dump(mode='json', exclude={'signals'}) if quality else None)
            records.append(frame)
            if association is not None:
                records.append(association)
            repo.add_many(records)
        return files.resolve(key)
