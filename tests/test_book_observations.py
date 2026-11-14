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


def margin_decider(crop, detection, candidates):
    """Stand in for Drex with the former score-margin rule, so association tests stay offline.

    Args:
        crop: Detection crop (unused).
        detection: New detection (unused).
        candidates: Surviving candidates ordered by score.

    Returns:
        A confident match for a clear winner, otherwise an ambiguous probability.
    """
    runner_up = candidates[1].score if len(candidates) > 1 else 0
    clear = candidates[0].score - runner_up >= 0.15
    return [0.95 if clear else 0.5] + [0.05] * (len(candidates) - 1)


class BookObservationTests(unittest.TestCase):
    """Exercise real SQLite transactions and geometric copy association."""

    def setUp(self):
        """Create an isolated sweep with a real saved reference JPEG."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        # No association test may reach the real Drex service.
        from unittest.mock import patch
        guard = patch.dict('os.environ', {'DREX_API_KEY': ''})
        guard.start()
        self.addCleanup(guard.stop)
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
                             jpeg or self.jpeg, 'test-model', decide_same=margin_decider)

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
            # Crop matching made it a candidate; Drex made the identity decision.
        self.assertEqual(rows[-1].association_method, 'drex')

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

    def test_manual_merge_preserves_observations_and_rejects_covisibility(self):
        """Merge separate sightings audibly while keeping simultaneous copies distinct."""
        from library_claim.identification.review import review_books
        self.persist(books=[detection(title='First reading')])
        self.save_frame('second')
        result = self.persist('second', books=[detection(title='Different reading')])
        ids = [row['id'] for row in result['inventory']]
        result = review_books(self.store.database, 'sweep', ids[1], ids[0], 'same')
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['observation_count'], 2)
        with Repository(self.store.database) as repo:
            self.assertEqual(len(repo.list_for_sweep(BookObservation, 'sweep')), 2)
            self.assertEqual(repo.connection.execute('SELECT origin FROM book_reviews').fetchone()[0], 'user')
        self.save_frame('both')
        result = self.persist('both', books=[detection(.1,.3,'Alpha'), detection(.6,.8,'Beta')])
        current = [book for book in result['inventory'] if 'both' in book['frame_ids']]
        with self.assertRaises(ValueError):
            review_books(self.store.database, 'sweep', current[0]['id'], current[1]['id'], 'same')

    def test_separate_decision_prevents_later_merge(self):
        """Respect a user's physical-identity decision on future review attempts."""
        from library_claim.identification.review import review_books
        self.persist(books=[detection(title='First')])
        self.save_frame('other')
        result = self.persist('other', books=[detection(title='Different')])
        a, b = [book['id'] for book in result['inventory']]
        review_books(self.store.database, 'sweep', a, b, 'separate')
        with self.assertRaises(ValueError):
            review_books(self.store.database, 'sweep', a, b, 'same')
        with self.assertRaises(ValueError):
            review_books(self.store.database, 'another-sweep', a, b, 'same')

    def test_clear_capture_reconciles_earlier_fragment(self):
        """Use later geometric evidence to combine an earlier unsupported sighting."""
        from library_claim.identification.review import identities
        from library_claim.domain.models import FrameAssociation
        self.persist(books=[detection(title='Good People')])
        image = np.full_like(textured_view(), 120)
        _, encoded = cv2.imencode('.jpg', image)
        jpeg = encoded.tobytes()
        self.save_frame('unmapped_fragment', jpeg=jpeg)
        self.persist('unmapped_fragment', books=[detection(title='The Good People')], jpeg=jpeg)
        # A later verified bridge makes the historical fragment comparable.
        with Repository(self.store.database) as repo:
            repo.add(FrameAssociation(sweep_id='sweep', frame_id='unmapped_fragment', reference_frame_id='reference',
                                      matrix=np.eye(3).tolist(), method='sift_ransac', matched_points=30, reprojection_error_px=0))
        self.save_frame('clear', np.eye(3).tolist())
        result = self.persist('clear', books=[detection(title='The Good People')])
        self.assertEqual(len(result['inventory']), 1)
        self.assertEqual(result['inventory'][0]['observation_count'], 3)
        with Repository(self.store.database) as repo:
            roots, _ = identities(repo, 'sweep')
            self.assertEqual(len(set(roots.values())), 1)
            self.assertEqual(repo.connection.execute('SELECT origin FROM book_reviews').fetchone()[0], 'automatic')

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


class BookMeasurementTests(unittest.TestCase):
    """Exercise metric measurements stored with observations and inventory totals."""

    # Reuse the fixture helpers without re-running the inherited association tests.
    setUp = BookObservationTests.setUp
    save_frame = BookObservationTests.save_frame
    persist = BookObservationTests.persist

    def depth(self):
        """Build a constant 1 m depth map for the 480×360 test frame.

        Returns:
            A depth map with a 500-pixel focal length.
        """
        from library_claim.measurement.spine import DepthMap
        return DepthMap(depth_m=np.ones((360, 480), np.float32), fx=500, fy=500, cx=240, cy=180,
                        image_width=480, image_height=360, method='depth_pro')

    def test_save_analysis_stores_measurements(self):
        """A 216×96 px spine at 1 m with f=500 px measures 43.2×19.2 cm."""
        result = save_analysis(self.store.database, 'sweep', 'reference',
                               BookFrameAnalysis(books=[detection()], complete=True), self.jpeg, 'test-model',
                               depth=self.depth())
        with Repository(self.store.database) as repo:
            row = repo.list_for_sweep(BookObservation, 'sweep')[0]
        self.assertAlmostEqual(row.spine_height_cm, 43.2, delta=0.5)
        self.assertAlmostEqual(row.spine_thickness_cm, 19.2, delta=0.3)
        self.assertEqual(row.scale_method, 'depth_pro')
        book = result['inventory'][0]
        self.assertAlmostEqual(book['spine_height_cm'], 43.2, delta=0.5)
        self.assertEqual(book['scale_method'], 'depth_pro')
        self.assertNotIn('No metric depth for this frame', book['review_reasons'])

    def test_missing_depth_adds_review_reason(self):
        """Save books without dimensions and say why when no depth exists."""
        book = self.persist()['inventory'][0]
        self.assertIsNone(book['spine_height_cm'])
        self.assertIn('No metric depth for this frame', book['review_reasons'])

    def test_background_measurement_fills_dimensions(self):
        """A later measure job adds dimensions and clears the missing-depth reason."""
        from library_claim.identification.storage import inventory, save_measurements
        self.persist()
        save_measurements(self.store.database, 'sweep', 'reference', self.depth())
        save_measurements(self.store.database, 'sweep', 'reference', self.depth())
        with Repository(self.store.database) as repo:
            book = inventory(repo, 'sweep')[0]
            jobs = [job for job in repo.list_for_sweep(ProcessingJob, 'sweep') if job.task == 'measure']
        self.assertEqual(len(jobs), 1)
        self.assertAlmostEqual(book['spine_height_cm'], 43.2, delta=0.5)
        self.assertNotIn('No metric depth for this frame', book['review_reasons'])

    def test_inventory_totals_exclude_excluded_books(self):
        """Count and sum shelf run only over books the user has not excluded."""
        from library_claim.identification.storage import inventory_totals
        books = [{'excluded': False, 'title': 'A', 'spine_thickness_cm': 3.0},
                 {'excluded': False, 'title': None, 'spine_thickness_cm': 2.0},
                 {'excluded': False, 'title': None, 'spine_thickness_cm': None},
                 {'excluded': True, 'title': 'B', 'spine_thickness_cm': 5.0}]
        self.assertEqual(inventory_totals(books), {'book_count': 3, 'books_identified': 1,
                                                   'books_unidentified': 2, 'shelf_run_m': 0.05})
