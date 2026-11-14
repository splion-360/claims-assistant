"""Verify Drex same-book decisions, thresholds and their use in association."""
import json
import os
import unittest
from unittest.mock import patch

import httpx

from library_claim.decisions.drex import DrexError
from library_claim.decisions.same_book import SameBookCandidate, choose_same_book, same_book_probabilities
from library_claim.domain.models import Book, BookObservation, ProcessingJob
from library_claim.identification.models import BookDetection
from library_claim.storage.repository import Repository
import test_book_observations as fixtures
from test_book_observations import detection


class SameBookDecisionTests(unittest.TestCase):
    """Exercise the D2 request and threshold rule."""

    def test_choose_same_book_thresholds(self):
        """Reuse at ≥0.8, create a new book when all ≤0.2, otherwise stay unresolved."""
        self.assertEqual(choose_same_book([0.9, 0.3]), 0)
        self.assertIsNone(choose_same_book([0.1, 0.15]))
        self.assertEqual(choose_same_book([0.5]), 'unresolved')
        self.assertEqual(choose_same_book([0.85, 0.82]), 'unresolved')

    def test_request_sends_crops_and_one_question_per_candidate(self):
        """Send the detection crop first and ask one yes/no per candidate."""
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(len(body['images']), 3)
            self.assertEqual(set(body['questions']), {'same_1', 'same_2'})
            self.assertEqual(body['state']['candidates'][0]['title'], 'Dune')
            return httpx.Response(200, json={'model': 'drex-max-v1.1', 'answers': {
                'same_1': {'type': 'noul', 'noul': 0.91}, 'same_2': {'type': 'noul', 'noul': 0.04}}})

        candidates = [SameBookCandidate('b1', b'c1', 'Dune', None, 'spine', 0.7, 'geometry_and_compatible_readings'),
                      SameBookCandidate('b2', b'c2', 'Dune Messiah', None, 'spine', 0.6, 'geometry_and_compatible_readings')]
        with patch.dict(os.environ, {'DREX_API_KEY': 'k'}):
            result = same_book_probabilities(b'd', detection(title='Dune'), candidates,
                                             client=httpx.Client(transport=httpx.MockTransport(handler)))
        self.assertEqual(result, [0.91, 0.04])


class DrexAssociationTests(unittest.TestCase):
    """Exercise D2 inside save_analysis with an injected decider."""

    setUp = fixtures.BookObservationTests.setUp
    save_frame = fixtures.BookObservationTests.save_frame

    def persist(self, frame, books, decide_same):
        """Save one frame's detections with a chosen same-book decider.

        Args:
            frame: Source frame ID.
            books: Detections.
            decide_same: Injected D2 decision function.

        Returns:
            Saved analysis summary.
        """
        from library_claim.identification.models import BookFrameAnalysis
        from library_claim.identification.storage import save_analysis
        return save_analysis(self.store.database, 'sweep', frame, BookFrameAnalysis(books=books, complete=True),
                             self.jpeg, 'test-model', decide_same=decide_same)

    def methods(self):
        """List association methods of saved observations.

        Returns:
            Methods in insertion order.
        """
        with Repository(self.store.database) as repo:
            return [row.association_method for row in repo.list_for_sweep(BookObservation, 'sweep')]

    def test_association_uses_drex_decision(self):
        """Reuse the existing book when Drex says it is the same one."""
        identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.save_frame('second', identity)
        self.persist('reference', [detection(title='Dune')], lambda crop, det, cands: [0.95] * len(cands))
        self.persist('second', [detection(title='Dune')], lambda crop, det, cands: [0.95] * len(cands))
        self.assertEqual(self.methods()[-1], 'drex')
        with Repository(self.store.database) as repo:
            self.assertEqual(len(repo.list_for_sweep(Book, 'sweep')), 1)

    def test_drex_failure_leaves_unresolved(self):
        """Keep the association open when Drex cannot decide."""
        def failing(crop, det, cands):
            """Fail like an unreachable Drex.

            Raises:
                DrexError: Always.
            """
            raise DrexError('down')

        self.save_frame('second', [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.persist('reference', [detection(title='Dune')], failing)
        self.persist('second', [detection(title='Dune')], failing)
        self.assertEqual(self.methods()[-1], 'unresolved')

    def test_conflicting_titles_never_sent_to_drex(self):
        """Hard vetoes stay in code; Drex only sees surviving candidates."""
        calls = []
        self.save_frame('second', [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.persist('reference', [detection(title='Dune')], lambda *args: calls.append(args) or [0.9])
        self.persist('second', [detection(title='Emma')], lambda *args: calls.append(args) or [0.9])
        self.assertEqual(calls, [])
