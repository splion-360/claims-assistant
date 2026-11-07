"""Verify visual book persistence, conservative association, and unknown readings."""
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from pydantic import ValidationError

from library_claim.capture.tracking.tracker import TrackingResult
from library_claim.domain.models import Book, BookObservation, Box, ProcessingJob, ReviewIssue
from library_claim.identification.models import BookDetection, BookFrameAnalysis
from library_claim.identification.storage import save_analysis
from library_claim.storage.live_capture import LiveCaptureStore
from library_claim.storage.repository import Repository
from test_tracking import textured_view


def detection(left=.2, right=.4, title=None):
    """Construct a book with an explicitly nullable reading.

    Args:
        left: Normalized left edge.
        right: Normalized right edge.
        title: Visible title or None.

    Returns:
        A validated spine detection.
    """
    return BookDetection(bbox=Box(x_min=left, x_max=right, y_min=.2, y_max=.8), view='spine', title=title)


class BookObservationTests(unittest.TestCase):
    """Exercise real SQLite transactions and geometric copy association."""

    def setUp(self):
        """Create an isolated sweep with a real saved reference JPEG."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = LiveCaptureStore(Path(self.directory.name), 'sweep')
        _, encoded = cv2.imencode('.jpg', textured_view())
        self.jpeg = encoded.tobytes()
        self.save_frame('reference')

    def save_frame(self, name, matrix=None, jpeg=None):
        """Persist a frame with optional verified mapping to the original reference.

        Args:
            name: Frame ID.
            matrix: Normalized current-to-reference matrix, or None.
            jpeg: Optional distinct image bytes.
        """
        result = TrackingResult(frame_id=name, status='tracked' if matrix is not None else 'reference_created',
                                reference_frame_id='reference' if matrix is not None else name,
                                current_to_reference=matrix, matched_points=30 if matrix is not None else 0,
                                reprojection_error_px=0 if matrix is not None else None, elapsed_ms=1)
        image = cv2.imdecode(np.frombuffer(jpeg or self.jpeg, np.uint8), cv2.IMREAD_COLOR)
        self.store.save(jpeg or self.jpeg, result, datetime.now(timezone.utc), image.shape[1], image.shape[0], 'test')

    def persist(self, frame='reference', books=None, jpeg=None, complete=True):
        """Store extraction results against exact source bytes.

        Args:
            frame: Source frame ID.
            books: Detection list; defaults to one unreadable book.
            jpeg: Source bytes override.
            complete: Whether the provider enumerated the whole frame.

        Returns:
            Persisted detection summary and inventory.
        """
        return save_analysis(self.store.database, 'sweep', frame,
                             BookFrameAnalysis(books=books if books is not None else [detection()], complete=complete),
                             jpeg or self.jpeg, 'test-model')

    def test_unknown_book_and_region_quality_are_saved_with_source(self):
        """Preserve null identity fields and crop quality instead of guessing titles."""
        result = self.persist()
        self.assertEqual(result['detected_count'], 1)
        self.assertIsNone(result['inventory'][0]['title'])
        with Repository(self.store.database) as repo:
            row = repo.list_for_sweep(BookObservation, 'sweep')[0]
            self.assertEqual(row.frame_id, 'reference')
            self.assertEqual(row.source_model, 'test-model')
            self.assertIn('title', row.unresolved_fields)
            self.assertIsNotNone(row.quality.sharpness)
            self.assertEqual(row.quality_metadata['region'], detection().bbox.model_dump())
            self.assertEqual(repo.list_for_sweep(Book, 'sweep')[0].identification_status, 'unidentified')

    def test_retry_is_idempotent_including_zero_detections(self):
        """Keep one extraction result per frame even for a legitimate empty result."""
        self.persist(books=[])
        result = self.persist(books=[detection(title='Invented retry')])
        self.assertEqual(result['detected_count'], 0)
        self.assertFalse(result['inventory'])
        with Repository(self.store.database) as repo:
            self.assertEqual(len(repo.list_for_sweep(ProcessingJob, 'sweep')), 1)

    def test_two_copies_of_same_title_stay_separate_and_repeat_views_reuse_ids(self):
        """Use position to distinguish copies and geometry to associate later views."""
        books = [detection(.1, .3, 'Same title'), detection(.6, .8, 'Same title')]
        first = self.persist(books=books)
        self.save_frame('second', np.eye(3).tolist())
        second = self.persist('second', books=books)
        self.assertEqual([b['id'] for b in first['inventory']], [b['id'] for b in second['inventory']])
        self.assertEqual([b['observation_count'] for b in second['inventory']], [2, 2])

    def test_projective_mapping_reuses_copy_in_shifted_view(self):
        """Match changed image coordinates through a verified current-to-reference map."""
        first = self.persist(books=[detection(.2, .4)])
        _, encoded = cv2.imencode('.jpg', np.roll(textured_view(), 5, axis=1))
        jpeg = encoded.tobytes()
        self.save_frame('shifted', [[1, 0, -.1], [0, 1, 0], [0, 0, 1]], jpeg)
        result = self.persist('shifted', books=[detection(.3, .5)], jpeg=jpeg)
        self.assertEqual(len(result['inventory']), 1)
        self.assertEqual(result['inventory'][0]['id'], first['inventory'][0]['id'])

    def test_missing_mapping_or_conflicting_reading_creates_reviewable_copy(self):
        """Avoid merging a title match without geometry or a conflicting title with it."""
        self.persist(books=[detection(title='First')])
        _, encoded = cv2.imencode('.jpg', 255 - textured_view())
        jpeg = encoded.tobytes()
        self.save_frame('unmapped', jpeg=jpeg)
        result = self.persist('unmapped', books=[detection(title='First')], jpeg=jpeg)
        self.assertEqual(len(result['inventory']), 2)
        self.assertTrue(result['inventory'][-1]['review_reasons'])
        self.save_frame('conflict', np.eye(3).tolist())
        result = self.persist('conflict', books=[detection(title='Different')])
        self.assertEqual(len(result['inventory']), 3)

    def test_competing_detections_do_not_merge_into_one_existing_book(self):
        """Keep ambiguous one-to-many matches provisional instead of reducing count."""
        self.persist(books=[detection(.2, .5)])
        self.save_frame('ambiguous', np.eye(3).tolist())
        result = self.persist('ambiguous', books=[detection(.2, .49), detection(.21, .5)])
        self.assertEqual(len(result['inventory']), 3)
        self.assertTrue(result['inventory'][-1]['review_reasons'])

    def test_additional_covisible_same_title_copy_increases_quantity_once(self):
        """Reuse one earlier copy and retain the other simultaneously visible copy."""
        self.persist(books=[detection(.2, .5, 'Same title')])
        self.save_frame('two_now', np.eye(3).tolist())
        result = self.persist('two_now', books=[detection(.2, .49, 'Same title'), detection(.21, .5, 'Same title')])
        self.assertEqual(len(result['inventory']), 2)
        self.assertEqual(sorted(b['observation_count'] for b in result['inventory']), [1, 2])
        self.assertTrue(all(b['status'] == 'provisional' for b in result['inventory']))

    def test_partial_title_updates_existing_identity_and_display(self):
        """Retain the copy ID while a later view supplies a fuller literal title."""
        first = self.persist(books=[detection(title='Good People')])
        self.save_frame('clearer', np.eye(3).tolist())
        result = self.persist('clearer', books=[detection(title='The Good People')])
        self.assertEqual(len(result['inventory']), 1)
        self.assertEqual(result['inventory'][0]['id'], first['inventory'][0]['id'])
        self.assertEqual(result['inventory'][0]['title'], 'The Good People')
        self.assertEqual(result['inventory'][0]['observation_count'], 2)

    def test_crop_features_recover_a_book_without_shared_frame_geometry(self):
        """Recover a repeated textured crop after tracking establishes another root."""
        self.persist(books=[detection(title='The Good People')])
        image = textured_view()
        image[:20] = 0
        _, encoded = cv2.imencode('.jpg', image)
        jpeg = encoded.tobytes()
        self.save_frame('new_root', jpeg=jpeg)
        result = self.persist('new_root', books=[detection(title='Good People')], jpeg=jpeg)
        self.assertEqual(len(result['inventory']), 1)
        with Repository(self.store.database) as repo:
            rows = repo.list_for_sweep(BookObservation, 'sweep')
            self.assertEqual(rows[-1].association_method, 'appearance_and_compatible_readings')

    def test_similar_titles_and_different_crops_remain_unresolved(self):
        """Do not treat compatible titles as proof of identity or count uncertainty."""
        self.persist(books=[detection(title='Good People')])
        _, encoded = cv2.imencode('.jpg', np.full_like(textured_view(), 120))
        jpeg = encoded.tobytes()
        self.save_frame('different', jpeg=jpeg)
        result = self.persist('different', books=[detection(title='The Good People')], jpeg=jpeg)
        self.assertEqual(len(result['inventory']), 2)
        self.assertEqual(result['inventory'][-1]['status'], 'unresolved')
        self.assertEqual(result['inventory'][0]['observation_count'], 1)

    def test_geometry_keeps_similar_titles_at_distinct_positions_separate(self):
        """Preserve two visible books even when one title contains the other."""
        books = [detection(.1, .3, 'Good People'), detection(.6, .8, 'The Good People')]
        self.persist(books=books)
        self.save_frame('both_again', np.eye(3).tolist())
        result = self.persist('both_again', books=books)
        self.assertEqual(len(result['inventory']), 2)
        self.assertEqual([b['observation_count'] for b in result['inventory']], [2, 2])

    def test_author_conflict_blocks_same_title_geometric_match(self):
        """Avoid merging different authors merely because their titles agree."""
        self.persist(books=[detection(title='Good People').model_copy(update={'author': 'Alice Smith'})])
        self.save_frame('other_author', np.eye(3).tolist())
        result = self.persist('other_author', books=[detection(title='Good People').model_copy(update={'author': 'Robert Jones'})])
        self.assertEqual(len(result['inventory']), 2)

    def test_invalid_boxes_duplicate_entries_and_mismatched_bytes_are_rejected(self):
        """Reject malformed model coordinates and a source-image identity mismatch."""
        with self.assertRaises(ValidationError):
            detection(.8, .2)
        with self.assertRaises(ValidationError):
            BookFrameAnalysis(books=[detection(), detection()])
        with self.assertRaises(ValueError):
            self.persist(jpeg=b'not-the-saved-image')
        with Repository(self.store.database) as repo:
            self.assertFalse(repo.list_for_sweep(BookObservation, 'sweep'))

    def test_tiny_region_retains_unidentified_book_with_review(self):
        """Preserve evidence when the book crop is too small for reliable quality scoring."""
        result = self.persist(books=[detection(.2, .201)], complete=False)
        self.assertEqual(result['detected_count'], 1)
        with Repository(self.store.database) as repo:
            row = repo.list_for_sweep(BookObservation, 'sweep')[0]
            self.assertIsNone(row.quality.sharpness)
            self.assertEqual(len(repo.list_for_sweep(ReviewIssue, 'sweep')), 2)


class LiveBookExtractionTests(unittest.IsolatedAsyncioTestCase):
    """Verify structured camera responses become persisted live inventory."""

    async def test_approved_capture_persists_books_and_updates_ui_state(self):
        """Exercise the real capture-to-storage path with only the provider mocked."""
        import time
        from unittest.mock import AsyncMock, patch
        from live_demo import server
        from live_demo.camera_filter import CameraCandidate

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = LiveCaptureStore(root, 'session')
            _, encoded = cv2.imencode('.jpg', textured_view())
            jpeg = encoded.tobytes()
            store.save(jpeg, TrackingResult(frame_id='frame', reference_frame_id='frame',
                                           status='reference_created', elapsed_ms=1),
                       datetime.now(timezone.utc), 480, 360, 'book_candidate')
            session = server.IntakeSession('session', camera_enabled=True, camera_mode_revision=1)
            candidate = CameraCandidate(jpeg, 'frame', time.monotonic(), 1)
            response = server.FrameObservation(observation='A book', supports_claimant_description=False,
                                               books=[detection(title='Visible Title')], complete=True)
            with patch.object(server, 'CAPTURE_DATA_DIR', root), \
                 patch.object(server, 'structured', new=AsyncMock(return_value=response)) as provider:
                result = await server._pin_evidence_photo(session, {}, candidate=candidate)
            self.assertTrue(result['pinned'])
            self.assertIn('normalized from 0 to 1', provider.call_args.args[0])
            self.assertEqual(provider.call_args.args[2], server.FrameObservation)
            state = server._current_ui_state(session)
            self.assertEqual(state['book_inventory'][0]['title'], 'Visible Title')
            self.assertEqual(state['evidence_photos'][0]['books'][0]['bbox'], detection().bbox.model_dump())
            self.assertEqual(state['evidence_photos'][0]['book_count'], 1)
            with Repository(store.database) as repo:
                self.assertEqual(len(repo.list_for_sweep(BookObservation, 'session')), 1)
