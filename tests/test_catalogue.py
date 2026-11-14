"""Verify Open Library candidates, the Drex match decision and catalogue observations."""
import json
import os
import unittest
from unittest.mock import patch

import httpx

from library_claim.domain.models import BookObservation, ReviewIssue
from library_claim.identification.catalogue import (WorkCandidate, choose_work, matching_edition, resolve_book,
                                                    search_works)
from library_claim.storage.repository import Repository
import test_book_observations as fixtures
from test_book_observations import detection

SEARCH = {'docs': [{'key': '/works/OL1W', 'title': 'Dune', 'author_name': ['Frank Herbert'], 'first_publish_year': 1965},
                   {'key': '/works/OL2W', 'title': 'Dune Messiah', 'author_name': ['Frank Herbert']}]}
EDITIONS = {'entries': [{'key': '/books/OL10M', 'title': 'Dune', 'publishers': ['Ace Books'], 'isbn_13': ['9780441172719']},
                        {'key': '/books/OL11M', 'title': 'Dune', 'publishers': ['Hodder'], 'isbn_10': ['0340839937']}]}
CANDIDATES = [WorkCandidate('/works/OL1W', 'Dune', ['Frank Herbert'], 1965),
              WorkCandidate('/works/OL2W', 'Dune Messiah', ['Frank Herbert'], None)]


def drex_choice(label: str, probability: float) -> dict:
    """Build a Drex response choosing one catalogue label.

    Args:
        label: Chosen label.
        probability: Its probability.

    Returns:
        A System One response body.
    """
    return {'model': 'drex-max-v1.1', 'answers': {'match': {'type': 'choice', 'choice': label,
                                                            'probabilities': {label: probability}}}}


def client(handler) -> httpx.Client:
    """Route requests to a test handler.

    Args:
        handler: Callable returning responses.

    Returns:
        An httpx client with a mock transport.
    """
    return httpx.Client(transport=httpx.MockTransport(handler))


class CatalogueTests(unittest.TestCase):
    """Exercise search parsing, the Drex threshold and the edition rule."""

    def setUp(self):
        """Provide a Drex key."""
        guard = patch.dict(os.environ, {'DREX_API_KEY': 'k'})
        guard.start()
        self.addCleanup(guard.stop)

    def test_search_parses_candidates(self):
        """Query by title and author and keep the top works."""
        def handler(request):
            self.assertEqual(request.url.host, 'openlibrary.org')
            self.assertEqual(request.url.params['title'], 'Dune')
            self.assertEqual(request.url.params['author'], 'Herbert')
            self.assertEqual(request.headers['User-Agent'], 'library-claim/1')
            return httpx.Response(200, json=SEARCH)

        self.assertEqual(search_works('Dune', 'Herbert', client=client(handler)), CANDIDATES)

    def test_choose_work_rejects_low_probability(self):
        """Refuse a match below the 0.7 catalogue threshold."""
        drex = client(lambda request: httpx.Response(200, json=drex_choice('c0', 0.6)))
        self.assertIsNone(choose_work({'title': 'Dune'}, CANDIDATES, client=drex))

    def test_choose_work_accepts_and_none_label_returns_none(self):
        """Accept a confident pick and treat 'none' as no match."""
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(set(body['questions']['match']['criteria']), {'c0', 'c1', 'none'})
            return httpx.Response(200, json=drex_choice('c0', 0.9))

        self.assertEqual(choose_work({'title': 'Dune'}, CANDIDATES, client=client(handler)), CANDIDATES[0])
        none = client(lambda request: httpx.Response(200, json=drex_choice('none', 0.95)))
        self.assertIsNone(choose_work({'title': 'Dune'}, CANDIDATES, client=none))

    def test_edition_only_with_unique_publisher_match(self):
        """Fill edition and ISBN only when exactly one edition's publisher matches the spine."""
        editions = client(lambda request: httpx.Response(200, json=EDITIONS))
        self.assertEqual(matching_edition('/works/OL1W', 'ACE books', client=editions),
                         {'edition': 'Dune (Ace Books)', 'isbn': '9780441172719'})
        self.assertIsNone(matching_edition('/works/OL1W', None, client=editions))
        doubled = {'entries': EDITIONS['entries'] + [EDITIONS['entries'][0] | {'key': '/books/OL12M'}]}
        self.assertIsNone(matching_edition('/works/OL1W', 'Ace Books',
                                           client=client(lambda request: httpx.Response(200, json=doubled))))


class ResolveBookTests(unittest.TestCase):
    """Exercise catalogue observations against a real SQLite store."""

    setUp = fixtures.BookObservationTests.setUp
    save_frame = fixtures.BookObservationTests.save_frame
    persist = fixtures.BookObservationTests.persist

    def book(self):
        """Save one legible spine and return its inventory entry.

        Returns:
            The inventory entry.
        """
        return self.persist(books=[detection(title='Dune')])['inventory'][0]

    def test_resolve_writes_catalogue_observation(self):
        """Append a catalogue observation with URL, retrieval time and no invented edition."""
        def http(request):
            if request.url.path == '/search.json':
                return httpx.Response(200, json=SEARCH)
            return httpx.Response(200, json=EDITIONS)

        book = self.book()
        with patch.dict(os.environ, {'DREX_API_KEY': 'k'}):
            resolve_book(self.store.database, 'sweep', book, http=client(http),
                         drex=client(lambda request: httpx.Response(200, json=drex_choice('c0', 0.9))))
        with Repository(self.store.database) as repo:
            rows = [row for row in repo.list_for_sweep(BookObservation, 'sweep') if row.provenance == 'catalogue']
            from library_claim.identification.storage import inventory
            entry = inventory(repo, 'sweep')[0]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].catalogue_url, 'https://openlibrary.org/works/OL1W')
        self.assertIsNotNone(rows[0].retrieved_at)
        self.assertIsNone(rows[0].isbn)
        self.assertEqual(entry['catalogue']['title'], 'Dune')
        self.assertEqual(entry['title'], 'Dune')
        self.assertEqual(entry['observation_count'], 1)

    def test_no_confident_match_adds_review_issue(self):
        """Record why a legible book has no catalogue match."""
        book = self.book()
        with patch.dict(os.environ, {'DREX_API_KEY': 'k'}):
            resolve_book(self.store.database, 'sweep', book,
                         http=client(lambda request: httpx.Response(200, json=SEARCH)),
                         drex=client(lambda request: httpx.Response(200, json=drex_choice('none', 0.9))))
        with Repository(self.store.database) as repo:
            reasons = [issue.reason for issue in repo.list_for_sweep(ReviewIssue, 'sweep')]
        self.assertIn('No confident catalogue match', reasons)

    def test_resolve_failure_writes_nothing(self):
        """Leave the store untouched when Open Library fails."""
        book = self.book()
        resolve_book(self.store.database, 'sweep', book, http=client(lambda request: httpx.Response(503)),
                     drex=client(lambda request: httpx.Response(500)))
        with Repository(self.store.database) as repo:
            rows = repo.list_for_sweep(BookObservation, 'sweep')
            reasons = [issue.reason for issue in repo.list_for_sweep(ReviewIssue, 'sweep')]
        self.assertEqual(len(rows), 1)
        self.assertNotIn('No confident catalogue match', reasons)
