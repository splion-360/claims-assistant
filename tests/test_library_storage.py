"""Verify capture persistence, association constraints, and evidence integrity."""
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
import tempfile
import unittest

from pydantic import ValidationError

from library_claim.domain.models import (
    Book, BookObservation, Box, Frame, ProcessingJob, ReviewIssue, Sweep,
)
from library_claim.storage.database import migrate
from library_claim.storage.evidence import EvidenceStore
from library_claim.storage.repository import Repository

NOW = datetime.now(timezone.utc)
BOX = Box(x_min=.1, y_min=.2, x_max=.3, y_max=.9)


class StorageTests(unittest.TestCase):
    """Exercise persistence invariants with isolated temporary storage."""
    def setUp(self):
        """Create an isolated database, evidence directory, and capture records."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'claims.sqlite3'
        self.repo = Repository(self.path)
        self.addCleanup(self.repo.close)
        self.evidence = EvidenceStore(Path(self.directory.name) / 'frames')
        self.sweep = self.repo.add(Sweep(captured_at=NOW))
        self.book = self.repo.add(Book(sweep_id=self.sweep.id))
        self.key = self.evidence.save(b'\xff\xd8\xfftest capture', 'image/jpeg')
        self.frame = self.repo.add(Frame(sweep_id=self.sweep.id, captured_at=NOW,
                                        evidence_key=self.key, width_px=640, height_px=480))

    def observation(self, **changes):
        """Build an observation with optional field overrides.

        Args:
            **changes: Fields that override the fixture association and box.

        Returns:
            A validated book observation.
        """
        values = dict(sweep_id=self.sweep.id, book_id=self.book.id,
                      frame_id=self.frame.id, bbox=BOX)
        return BookObservation(**(values | changes))

    def test_multiple_observations_survive_restart_without_increasing_book_count(self):
        """Verify that reopened storage retains two observations of one physical book."""
        first = self.repo.add(self.observation(unresolved_fields=['title']))
        frame2 = self.repo.add(Frame(sweep_id=self.sweep.id, captured_at=NOW,
                                    evidence_key=self.key, width_px=640, height_px=480))
        second = self.repo.add(self.observation(frame_id=frame2.id, title='Visible title'))
        with Repository(self.path) as reopened:
            observations = reopened.observations_for(self.book.id)
            self.assertEqual([o.id for o in observations], [first.id, second.id])
            self.assertIsNone(observations[0].title)
            self.assertEqual(observations[1].title, 'Visible title')
            self.assertEqual(len(reopened.list_for_sweep(Book, self.sweep.id)), 1)
            self.assertIsNone(reopened.get(Book, self.book.id).title)
            self.assertEqual(self.evidence.read(reopened.get(Frame, frame2.id).evidence_key),
                             b'\xff\xd8\xfftest capture')

    def test_foreign_keys_prevent_missing_or_cross_sweep_associations(self):
        """Verify invalid associations roll back without poisoning the connection."""
        other = self.repo.add(Sweep(captured_at=NOW))
        other_book = self.repo.add(Book(sweep_id=other.id))
        for changes in ({'book_id': 'missing'}, {'book_id': other_book.id},
                        {'frame_id': 'missing'}, {'sweep_id': other.id}):
            with self.subTest(changes=changes), self.assertRaises(sqlite3.IntegrityError):
                self.repo.add(self.observation(**changes))
        self.assertEqual(self.repo.observations_for(self.book.id), [])
        self.repo.add(self.observation())  # A failed insert does not poison the connection.

    def test_duplicate_ids_and_evidence_updates_do_not_overwrite_history(self):
        """Verify duplicate inserts and evidence updates preserve observation history."""
        observation = self.repo.add(self.observation())
        with self.assertRaises(sqlite3.IntegrityError):
            self.repo.add(observation)
        with self.assertRaises(ValueError):
            self.repo.update(observation.model_copy(update={'title': 'replacement'}))
        self.assertIsNone(self.repo.get(BookObservation, observation.id).title)

    def test_book_exclusion_review_and_job_persist(self):
        """Verify exclusions, review issues, and job status survive reopening."""
        self.repo.update(self.book.model_copy(update={'excluded': True}))
        issue = self.repo.add(ReviewIssue(sweep_id=self.sweep.id, book_id=self.book.id,
                                         field='title', reason='Unreadable spine'))
        job = self.repo.add(ProcessingJob(sweep_id=self.sweep.id, frame_id=self.frame.id,
                                         book_id=self.book.id, task='identify', created_at=NOW))
        self.repo.update(job.model_copy(update={'status': 'running'}))
        with Repository(self.path) as reopened:
            self.assertTrue(reopened.get(Book, self.book.id).excluded)
            self.assertEqual(reopened.get(ReviewIssue, issue.id).status, 'open')
            self.assertEqual(reopened.get(ProcessingJob, job.id).status, 'running')
            self.assertEqual(reopened.connection.execute('SELECT status FROM jobs').fetchone()[0], 'running')
        with self.assertRaises(ValueError):
            self.repo.update(job.model_copy(update={'frame_id': 'another'}))

    def test_migration_is_idempotent_and_rejects_newer_schema(self):
        """Verify repeated migration preserves data and rejects unsupported versions."""
        migrate(self.repo.connection)
        self.assertEqual(self.repo.get(Book, self.book.id), self.book)
        self.repo.connection.execute('PRAGMA user_version = 999')
        with self.assertRaises(ValueError):
            Repository(self.path)

    def test_contract_validation(self):
        """Verify invalid geometry, confidence, and timestamps are rejected."""
        with self.assertRaises(ValidationError):
            Box(x_min=.7, y_min=0, x_max=.2, y_max=1)
        with self.assertRaises(ValidationError):
            self.observation(association_confidence=1.5)
        with self.assertRaises(ValidationError):
            Sweep(captured_at=datetime.now())
        with self.assertRaises(ValidationError):
            self.repo.update(self.book.model_copy(update={'id_confidence': float('nan')}))

    def test_evidence_deduplication_paths_and_integrity(self):
        """Verify file deduplication, path validation, and corruption detection."""
        self.assertEqual(self.evidence.save(b'\xff\xd8\xfftest capture', 'image/jpeg'), self.key)
        self.assertEqual(len(list(self.evidence.root.iterdir())), 1)
        with self.assertRaises(ValueError):
            self.evidence.resolve('../outside.jpg')
        with self.assertRaises(ValueError):
            self.evidence.save(b'not an image', 'image/jpeg')
        self.evidence.resolve(self.key).write_bytes(b'corrupt')
        with self.assertRaises(ValueError):
            self.evidence.read(self.key)
        with self.assertRaises(ValueError):
            self.evidence.save(b'\xff\xd8\xfftest capture', 'image/jpeg')


if __name__ == '__main__':
    unittest.main()
