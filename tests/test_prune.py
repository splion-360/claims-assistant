"""Verify end-of-sweep pruning keeps every frame evidence depends on and removes the rest."""
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2

from library_claim.capture.tracking.tracker import TrackingResult
from library_claim.decisions.frame_check import save_frame_check
from library_claim.domain.models import Frame, FrameAssociation, ProcessingJob
from library_claim.identification.models import BookFrameAnalysis
from library_claim.identification.storage import inventory, save_analysis
from library_claim.storage.live_capture import LiveCaptureStore
from library_claim.storage.prune import prune_sweep
from library_claim.storage.repository import Repository
from test_book_observations import detection, margin_decider
from test_frame_check import check
from test_tracking import textured_view

IDENTITY = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def jpeg(seed: int) -> bytes:
    """Encode a distinct synthetic frame.

    Args:
        seed: Scene seed; different seeds give different bytes.

    Returns:
        JPEG bytes.
    """
    return cv2.imencode('.jpg', textured_view(seed))[1].tobytes()


class PruneTests(unittest.TestCase):
    """Exercise the keep set, file removal, shared files and idempotence."""

    def setUp(self):
        """Save a sweep: a read frame tracked to a reference, plus an unrelated reference chain."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        guard = patch.dict('os.environ', {'DREX_API_KEY': ''})
        guard.start()
        self.addCleanup(guard.stop)
        self.root = Path(directory.name)
        self.store = LiveCaptureStore(self.root, 'sweep')
        self.save('ref', 1)
        self.save('read', 2, reference='ref')
        self.save('other_ref', 3)
        self.save('other', 4, reference='other_ref')
        save_analysis(self.store.database, 'sweep', 'read', BookFrameAnalysis(books=[detection(title='Dune')], complete=True),
                      jpeg(2), 'test-model', decide_same=margin_decider)
        save_frame_check(self.store.database, 'sweep', 'other', check(0.1))

    def save(self, name, seed, reference=None, store=None):
        """Persist one frame, optionally tracked to a reference frame.

        Args:
            name: Frame ID.
            seed: Image seed.
            reference: Reference frame ID, or None for a new reference.
            store: Capture store; defaults to this sweep's.
        """
        result = TrackingResult(frame_id=name, status='tracked' if reference else 'reference_created',
                                reference_frame_id=reference or name, current_to_reference=IDENTITY if reference else None,
                                matched_points=30 if reference else 0, reprojection_error_px=0 if reference else None,
                                elapsed_ms=1)
        (store or self.store).save(jpeg(seed), result, datetime.now(timezone.utc), 480, 360, 'test')

    def frame_ids(self):
        """List the sweep's remaining frame IDs.

        Returns:
            Sorted frame IDs.
        """
        with Repository(self.store.database) as repo:
            return sorted(frame.id for frame in repo.list_for_sweep(Frame, 'sweep'))

    def test_keeps_cited_frames_and_their_references(self):
        """The read frame and its tracking reference stay; the unrelated chain goes."""
        removed = prune_sweep(self.store.database, 'sweep')
        self.assertEqual(removed, 2)
        self.assertEqual(self.frame_ids(), ['read', 'ref'])
        with Repository(self.store.database) as repo:
            self.assertEqual([row.frame_id for row in repo.list_for_sweep(FrameAssociation, 'sweep')], ['read'])
            self.assertNotIn('check_other', [job.id for job in repo.list_for_sweep(ProcessingJob, 'sweep')])
            self.assertEqual(inventory(repo, 'sweep')[0]['title'], 'Dune')
        self.assertEqual(len(list((self.root / 'frames').glob('*.jpg'))), 2)

    def test_pruning_twice_removes_nothing_more(self):
        """A second prune is a no-op."""
        prune_sweep(self.store.database, 'sweep')
        self.assertEqual(prune_sweep(self.store.database, 'sweep'), 0)

    def test_file_shared_with_another_sweep_is_kept(self):
        """An image still used by another sweep's frame is not deleted."""
        self.save('elsewhere', 4, store=LiveCaptureStore(self.root, 'sweep2'))
        prune_sweep(self.store.database, 'sweep')
        with Repository(self.store.database) as repo:
            key = repo.get(Frame, 'elsewhere').evidence_key
        self.assertTrue((self.root / 'frames' / key).exists())

    def test_prune_finished_skips_live_sweeps(self):
        """Startup pruning leaves sweeps that are still live untouched."""
        from library_claim.storage.prune import prune_finished
        self.save('live_only', 5, store=LiveCaptureStore(self.root, 'live'))
        removed = prune_finished(self.store.database, live={'live'}, idle_s=0)
        self.assertEqual(removed, 2)
        with Repository(self.store.database) as repo:
            self.assertEqual([frame.id for frame in repo.list_for_sweep(Frame, 'live')], ['live_only'])


class SessionPruneTests(unittest.IsolatedAsyncioTestCase):
    """An expired session's sweep is pruned."""

    async def test_discarding_a_session_prunes_its_sweep(self):
        """discard_session hands the session's sweep to the pruner."""
        from live_demo import server
        session = server.IntakeSession('expired', owner='owner')
        with patch.object(server, 'prune_sweep') as prune:
            await server.discard_session(session)
        prune.assert_called_once_with(server.CAPTURE_DATA_DIR / 'claims.sqlite3', 'expired')


class StartupPruneSafetyTests(unittest.TestCase):
    """Startup pruning never touches a sweep that could still be in use."""

    def test_recently_active_sweep_is_left_alone(self):
        """A sweep with frames newer than the idle threshold is skipped even if not known as live."""
        from library_claim.storage.prune import prune_finished
        with tempfile.TemporaryDirectory() as directory:
            store = LiveCaptureStore(Path(directory), 'busy')
            result = TrackingResult(frame_id='f', status='reference_created', reference_frame_id='f', elapsed_ms=1)
            store.save(jpeg(1), result, datetime.now(timezone.utc), 480, 360, 'test')
            self.assertEqual(prune_finished(store.database, live=set(), idle_s=1800), 0)
            with Repository(store.database) as repo:
                self.assertEqual(len(repo.list_for_sweep(Frame, 'busy')), 1)

    def test_app_startup_in_tests_cannot_reach_real_data(self):
        """Starting the app in a test prunes only the temporary data folder it is given."""
        from live_demo import server
        from fastapi.testclient import TestClient
        with tempfile.TemporaryDirectory() as directory, patch.object(server, 'CAPTURE_DATA_DIR', Path(directory)), \
                patch.object(server, 'prune_finished') as prune:
            store = LiveCaptureStore(Path(directory), 'old')
            store.save(jpeg(1), TrackingResult(frame_id='f', status='reference_created', reference_frame_id='f',
                                               elapsed_ms=1), datetime.now(timezone.utc), 480, 360, 'test')
            with TestClient(server.app, base_url='http://127.0.0.1:4177'):
                pass
        self.assertEqual(prune.call_args.args[0], Path(directory) / 'claims.sqlite3')
