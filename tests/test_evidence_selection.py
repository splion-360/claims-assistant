"""Verify evidence comparison and persisted perspective associations."""
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import tempfile
import unittest

from library_claim.capture.quality import QualityAssessment
from library_claim.capture.selection import compare_evidence
from library_claim.capture.tracking.tracker import TrackingResult
from library_claim.domain.models import Frame, FrameAssociation, Quality, Sweep
from library_claim.storage.repository import Repository


class EvidenceSelectionTests(unittest.TestCase):
    """Check candidate decisions without claiming semantic relevance."""

    def setUp(self):
        """Create reference-aligned observations with comparable quality settings."""
        self.view = TrackingResult(frame_id='one', reference_frame_id='reference',
                                   status='tracked', current_to_reference=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                   elapsed_ms=1)
        self.quality = QualityAssessment(signals=Quality(sharpness=10, brisque=40),
                                         assessed_width_px=640, assessed_height_px=480,
                                         dark_cutoff=16, bright_cutoff=239, opencv_version='4.10', elapsed_ms=1)

    def test_repeats_skip_and_unmatched_views_remain_candidates(self):
        """Skip unchanged evidence and retain unfamiliar views for analysis."""
        self.assertEqual(compare_evidence(self.view, self.quality).action, 'analyze_candidate')
        self.assertEqual(compare_evidence(self.view, self.quality, self.view, self.quality).action, 'skip')
        uncertain = self.view.model_copy(update={'status': 'uncertain', 'current_to_reference': None})
        self.assertEqual(compare_evidence(uncertain, self.quality).action, 'review_association')

    def test_improved_detail_and_coverage_are_candidates(self):
        """Distinguish a close-up and expanded coverage from an unchanged view."""
        close = self.view.model_copy(update={'current_to_reference': [[.6, 0, .2], [0, .6, .2], [0, 0, 1]]})
        moved = self.view.model_copy(update={'current_to_reference': [[1, 0, .3], [0, 1, 0], [0, 0, 1]]})
        self.assertEqual(compare_evidence(close, self.quality, self.view, self.quality).reason,
                         'more_pixel_detail_candidate')
        self.assertEqual(compare_evidence(moved, self.quality, self.view, self.quality).reason,
                         'additional_coverage_candidate')
        self.assertEqual(compare_evidence(close, self.quality, self.view, self.quality, unresolved=False).action, 'skip')

    def test_better_quality_only_matters_for_unresolved_evidence(self):
        """Reanalyze improved unresolved evidence without repeating complete work."""
        better = self.quality.model_copy(update={'signals': Quality(sharpness=20, brisque=30)})
        self.assertEqual(compare_evidence(self.view, better, self.view, self.quality).action, 'analyze_candidate')
        self.assertEqual(compare_evidence(self.view, better, self.view, self.quality, unresolved=False).action, 'skip')

    def test_persisted_mapping_requires_both_frames_in_same_sweep(self):
        """Persist mappings with traceable frame references and sweep isolation."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'db'
            with Repository(path) as repo:
                sweep = repo.add(Sweep(captured_at=datetime.now(timezone.utc)))
                other = repo.add(Sweep(captured_at=sweep.captured_at))
                frames = [repo.add(Frame(sweep_id=owner, captured_at=sweep.captured_at,
                                         evidence_key='a' * 64 + '.jpg', width_px=640, height_px=480))
                          for owner in (sweep.id, sweep.id, other.id)]
                association = FrameAssociation(sweep_id=sweep.id, frame_id=frames[0].id,
                                               reference_frame_id=frames[1].id, matrix=self.view.current_to_reference,
                                               method='sift_ransac', matched_points=20, reprojection_error_px=.5)
                repo.add(association)
                with self.assertRaises(sqlite3.IntegrityError):
                    repo.add(association.model_copy(update={'id': 'bad', 'reference_frame_id': frames[2].id}))
            with Repository(path) as repo:
                self.assertEqual(repo.get(FrameAssociation, association.id), association)
