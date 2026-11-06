"""Verify selective live-frame persistence and restart-safe evidence references."""
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import sqlite3
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from library_claim.capture.tracking.tracker import TrackingResult
from library_claim.domain.models import Frame, FrameAssociation, Sweep
from library_claim.storage.evidence import EvidenceStore
from library_claim.storage.live_capture import LiveCaptureStore
from library_claim.storage.repository import Repository
from live_demo.camera_tracking import CameraTracking
from test_tracking import textured_view


def jpeg(image):
    """Encode a synthetic frame for the same path used by browser captures.

    Args:
        image: Uint8 grayscale image.

    Returns:
        Encoded JPEG bytes.
    """
    success, encoded = cv2.imencode('.jpg', image)
    if not success:
        raise ValueError('Test frame encoding failed')
    return encoded.tobytes()


class LivePersistenceTests(unittest.TestCase):
    """Exercise persistent references, selection, and failure boundaries."""

    def setUp(self):
        """Create isolated storage and a camera activation with real tracking."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.store = LiveCaptureStore(self.root, 'sweep1')
        self.stream = CameraTracking(self.store)
        self.image = textured_view()

    def test_repeated_views_are_not_saved_per_frame(self):
        """Retain one reference while discarding repeated intermediate views."""
        data = jpeg(self.image)
        for i in range(8):
            self.stream.assess(data, f'f{i}', i * .2)
        with Repository(self.store.database) as repo:
            frames = repo.list_for_sweep(Frame, 'sweep1')
            self.assertEqual(len(frames), 1)
            self.assertEqual(frames[0].selection_reason, 'tracking_reference')
            self.assertIsNotNone(frames[0].quality.brisque)
            self.assertEqual(frames[0].quality_metadata['opencv_version'], cv2.__version__)
            self.assertEqual(EvidenceStore(self.root / 'frames').read(frames[0].evidence_key), data)
        self.assertEqual(len(list((self.root / 'frames').glob('*.jpg'))), 1)

    def test_added_coverage_keeps_original_and_verified_mapping_after_restart(self):
        """Save a changed view with a mapping to an openable persistent reference."""
        first = jpeg(self.image)
        self.stream.assess(first, 'reference', 0)
        matrix = np.float64([[1, 0, 95], [0, 1, 0], [0, 0, 1]])
        shifted = jpeg(cv2.warpPerspective(self.image, matrix, (480, 360)))
        result = self.stream.assess(shifted, 'shifted', 2)
        self.assertEqual(result.status, 'recovered')
        self.assertEqual(self.stream.saved_reason, 'additional_coverage_candidate')
        # Reopening the repository simulates a new process inspecting the evidence.
        with Repository(self.store.database) as reopened:
            frames = reopened.list_for_sweep(Frame, 'sweep1')
            self.assertEqual(len(frames), 2)
            mapping = reopened.get(FrameAssociation, 'shifted')
            self.assertEqual(mapping.reference_frame_id, 'reference')
            self.assertEqual(mapping.method, 'sift_ransac')
            files = EvidenceStore(self.root / 'frames')
            for frame in frames:
                self.assertTrue(files.resolve(frame.evidence_key).exists())
                self.assertIsNotNone(cv2.imdecode(np.frombuffer(files.read(frame.evidence_key), np.uint8), 0))

    def test_new_activation_preserves_disk_history_and_deduplicates_image_bytes(self):
        """Reacquire tracking without deleting the earlier activation's evidence."""
        data = jpeg(self.image)
        self.stream.assess(data, 'first', 0)
        second = CameraTracking(LiveCaptureStore(self.root, 'sweep1'))
        second.assess(data, 'second', 5)
        with Repository(self.store.database) as repo:
            self.assertEqual(len(repo.list_for_sweep(Frame, 'sweep1')), 2)
            self.assertEqual(repo.get(Sweep, 'sweep1').device, 'browser camera')
        self.assertEqual(len(list((self.root / 'frames').glob('*.jpg'))), 1)

    def test_missing_reference_is_not_committed(self):
        """Reject associations whose supporting image was never saved."""
        result = TrackingResult(frame_id='current', reference_frame_id='missing', status='tracked',
                                current_to_reference=np.eye(3).tolist(), matched_points=20,
                                reprojection_error_px=.5, elapsed_ms=1)
        with self.assertRaises(ValueError):
            self.store.save(jpeg(self.image), result, datetime.now(timezone.utc), 480, 360, 'candidate')
        with Repository(self.store.database) as repo:
            self.assertEqual(repo.list_for_sweep(Frame, 'sweep1'), [])
        self.assertEqual(list((self.root / 'frames').glob('*.jpg')), [])

    def test_failed_save_reacquires_instead_of_citing_unsaved_reference(self):
        """Recover from a transient persistence failure on the next good frame."""
        data = jpeg(self.image)
        with patch.object(self.store, 'save', side_effect=OSError('disk unavailable')):
            with self.assertRaises(OSError):
                self.stream.assess(data, 'failed', 0)
        result = self.stream.assess(data, 'retry', 1.2)
        self.assertEqual(result.status, 'reference_created')
        with Repository(self.store.database) as repo:
            self.assertEqual([f.id for f in repo.list_for_sweep(Frame, 'sweep1')], ['retry'])

    def test_batch_insert_rolls_back_frame_when_mapping_fails(self):
        """Leave no partial metadata when a relational constraint rejects a batch."""
        self.stream.assess(jpeg(self.image), 'reference', 0)
        with Repository(self.store.database) as repo:
            frame = repo.get(Frame, 'reference').model_copy(update={'id': 'new'})
            mapping = FrameAssociation(sweep_id='sweep1', frame_id='new', reference_frame_id='missing',
                                       matrix=np.eye(3).tolist(), method='optical_flow',
                                       matched_points=20, reprojection_error_px=1)
            with self.assertRaises(sqlite3.IntegrityError):
                repo.add_many([frame, mapping])
            with self.assertRaises(KeyError):
                repo.get(Frame, 'new')
