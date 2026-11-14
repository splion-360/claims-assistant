"""Verify spoken-correction intents, target resolution and their effects on the inventory."""
import json
import os
import unittest
from unittest.mock import patch

import httpx

from library_claim.decisions.corrections import classify_intent, resolve_target, target_frame
from library_claim.domain.models import Book, BookObservation, ReviewIssue
from library_claim.identification.corrections import apply_correction
from library_claim.identification.storage import inventory
from library_claim.storage.repository import Repository
import test_book_observations as fixtures
from test_book_observations import detection


def client(handler) -> httpx.Client:
    """Route Drex requests to a test handler.

    Args:
        handler: Callable returning responses.

    Returns:
        An httpx client with a mock transport.
    """
    return httpx.Client(transport=httpx.MockTransport(handler))


def answer(question: str, label: str, probability: float) -> httpx.Response:
    """Build a Drex choice response.

    Args:
        question: Question ID.
        label: Chosen label.
        probability: Its probability.

    Returns:
        HTTP response with a System One body.
    """
    return httpx.Response(200, json={'model': 'drex-max-v1.1', 'answers': {question: {
        'type': 'choice', 'choice': label, 'probabilities': {label: probability}}}})


class CorrectionDecisionTests(unittest.TestCase):
    """Exercise D5 intent and target decisions."""

    def setUp(self):
        """Provide a Drex key."""
        guard = patch.dict(os.environ, {'DREX_API_KEY': 'k'})
        guard.start()
        self.addCleanup(guard.stop)

    def test_classify_intent_maps_choice_and_failure_is_chat(self):
        """Return Drex's intent, or plain conversation when Drex fails."""
        def handler(request):
            self.assertIn('exclude_books', json.loads(request.content)['questions']['intent']['criteria'])
            return answer('intent', 'mark_signed', 0.9)

        self.assertEqual(classify_intent("that one's signed", client=client(handler)), 'mark_signed')
        self.assertEqual(classify_intent('skip those', client=client(lambda request: httpx.Response(402))),
                         'question_or_chat')

    def test_single_visible_book_needs_no_drex(self):
        """Resolve 'that' directly when only one book is in view."""
        def handler(request):
            raise AssertionError('Drex should not be called')

        books = [{'id': 'b1', 'title': 'Dune', 'author': None, 'observations': [{'bbox': {'x_min': 0.2}}]}]
        self.assertEqual(resolve_target('that one', books, allow_all=False, client=client(handler)), ['b1'])

    def test_target_choice_and_low_confidence(self):
        """Pick a book or all visible books, and refuse below the 0.6 threshold."""
        books = [{'id': 'b1', 'title': 'Dune', 'author': None, 'observations': [{'bbox': {'x_min': 0.6}}]},
                 {'id': 'b2', 'title': None, 'author': None, 'observations': [{'bbox': {'x_min': 0.1}}]}]

        def handler(request):
            criteria = json.loads(request.content)['questions']['target']['criteria']
            self.assertIn('position 1 from the left', criteria['b2'])
            self.assertIn('all_visible', criteria)
            return answer('target', 'all_visible', 0.8)

        self.assertEqual(resolve_target('skip those', books, allow_all=True, client=client(handler)), ['b1', 'b2'])
        low = client(lambda request: answer('target', 'b1', 0.5))
        self.assertIsNone(resolve_target('that one', books, allow_all=False, client=low))

    def test_no_recent_frame_asks_instead_of_excluding(self):
        """Use only a frame with detections from the last 30 seconds."""
        self.assertEqual(target_frame([('old', 10.0), ('new', 50.0)], 60.0), 'new')
        self.assertIsNone(target_frame([('old', 10.0)], 60.0))
        self.assertIsNone(target_frame([], 60.0))


class ApplyCorrectionTests(unittest.TestCase):
    """Exercise correction effects against a real SQLite store."""

    setUp = fixtures.BookObservationTests.setUp
    save_frame = fixtures.BookObservationTests.save_frame
    persist = fixtures.BookObservationTests.persist

    def book_id(self):
        """Save one legible spine.

        Returns:
            The saved book's ID.
        """
        return self.persist(books=[detection(title='Dune')])['inventory'][0]['id']

    def test_exclude_sets_flag_and_records_statement(self):
        """Mark the book excluded and keep the user's words as evidence."""
        book_id = self.book_id()
        apply_correction(self.store.database, 'sweep', 'exclude_books', [book_id], 'reference', 'not mine')
        with Repository(self.store.database) as repo:
            self.assertTrue(repo.get(Book, book_id).excluded)
            statements = [row for row in repo.list_for_sweep(BookObservation, 'sweep') if row.provenance == 'user_statement']
            entry = inventory(repo, 'sweep')[0]
        self.assertEqual(statements[0].visible_text, 'not mine')
        self.assertTrue(entry['excluded'])
        self.assertEqual(entry['observation_count'], 1)

    def test_first_edition_sets_needs_appraisal(self):
        """Route a claimed first edition to human appraisal."""
        book_id = self.book_id()
        apply_correction(self.store.database, 'sweep', 'mark_first_edition', [book_id], 'reference', 'first edition')
        with Repository(self.store.database) as repo:
            self.assertTrue(repo.get(Book, book_id).needs_appraisal)

    def test_correct_title_keeps_visual_reading(self):
        """Store a stated title for review without overwriting what the spine shows."""
        book_id = self.book_id()
        apply_correction(self.store.database, 'sweep', 'correct_title', [book_id], 'reference',
                         "no, that's Emma", stated_title='Emma')
        with Repository(self.store.database) as repo:
            entry = inventory(repo, 'sweep')[0]
            reasons = [issue.reason for issue in repo.list_for_sweep(ReviewIssue, 'sweep')]
        self.assertEqual(entry['title'], 'Dune')
        self.assertIn('User-stated title differs from spine reading', reasons)
