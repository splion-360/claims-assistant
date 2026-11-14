"""Verify Google Books candidates, the Drex match decision and catalogue observations."""
import os
import unittest
from unittest.mock import patch

import httpx

from library_claim.domain.models import BookObservation, ReviewIssue
from library_claim.identification.catalogue import (WorkCandidate, catalogue_sources, choose_work,
                                                    resolve_book, search_google_books)
from library_claim.storage.repository import Repository
import test_book_observations as fixtures
from test_book_observations import detection

VOLUMES = {'items': [
    {'id': 'vol1', 'volumeInfo': {'title': 'Dune', 'authors': ['Frank Herbert'], 'publishedDate': '1965-08-01',
                                  'publisher': 'Ace Books', 'canonicalVolumeLink': 'https://books.google.com/vol1',
                                  'industryIdentifiers': [{'type': 'ISBN_13', 'identifier': '9780441172719'}]}},
    {'id': 'vol2', 'volumeInfo': {'title': 'Dune Messiah', 'authors': ['Frank Herbert'],
                                  'canonicalVolumeLink': 'https://books.google.com/vol2'}},
]}
CANDIDATES = [WorkCandidate('vol1', 'Dune', ['Frank Herbert'], 1965, url='https://books.google.com/vol1',
                            publisher='Ace Books', isbn='9780441172719'),
              WorkCandidate('vol2', 'Dune Messiah', ['Frank Herbert'], None,
                            url='https://books.google.com/vol2')]


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

    def test_google_books_is_the_only_catalogue(self):
        """Try Google Books alone, so one provider answers catalogue lookups."""
        self.assertEqual([name for name, _ in catalogue_sources()], ['Google Books'])

    def test_search_parses_candidates(self):
        """Read title, authors, year, link, publisher and ISBN from volume records."""
        def handler(request):
            """Answer the volumes request and check where it went.

            Args:
                request: The outgoing request.

            Returns:
                A volumes response.
            """
            self.assertEqual(request.url.host, 'www.googleapis.com')
            self.assertIn('intitle:"Dune"', request.url.params.get('q'))
            self.assertIn('inauthor:"Herbert"', request.url.params.get('q'))
            return httpx.Response(200, json=VOLUMES)

        self.assertEqual(search_google_books('Dune', 'Herbert', client=client(handler)), CANDIDATES)

    def test_search_sends_the_api_key_when_configured(self):
        """Send the configured key so the request is not refused on the shared quota."""
        seen = []

        def handler(request):
            """Record the key sent.

            Args:
                request: The outgoing request.

            Returns:
                An empty volumes response.
            """
            seen.append(request.url.params.get('key'))
            return httpx.Response(200, json={'items': []})

        with patch.dict(os.environ, {'GOOGLE_BOOKS_API_KEY': 'books-key'}):
            search_google_books('Dune', None, client=client(handler))
        self.assertEqual(seen, ['books-key'])

    def test_search_skips_records_without_an_identity(self):
        """Drop a volume with no ID or title rather than carrying a blank candidate."""
        partial = {'items': [{'id': 'vol1', 'volumeInfo': {}}, {'volumeInfo': {'title': 'Dune'}}]}
        self.assertEqual(search_google_books('Dune', None, client=client(
            lambda request: httpx.Response(200, json=partial))), [])

    def test_choose_work_rejects_low_probability(self):
        """Keep a match only when Drex reaches the configured threshold."""
        self.assertIsNone(choose_work({'title': 'Dune'}, CANDIDATES,
                                      client=client(lambda request: httpx.Response(200, json=drex_choice('c0', 0.4)))))

    def test_choose_work_accepts_and_none_label_returns_none(self):
        """Accept a confident choice and respect an explicit no-match answer."""
        chosen = choose_work({'title': 'Dune'}, CANDIDATES,
                             client=client(lambda request: httpx.Response(200, json=drex_choice('c0', 0.95))))
        self.assertEqual(chosen.key, 'vol1')
        self.assertIsNone(choose_work({'title': 'Dune'}, CANDIDATES,
                                      client=client(lambda request: httpx.Response(200, json=drex_choice('none', 0.9)))))
        self.assertIsNone(choose_work({'title': 'Dune'}, [], client=client(lambda request: httpx.Response(500))))


class ResolveBookTests(unittest.TestCase):
    """Exercise catalogue observations against a real SQLite store."""

    setUp = fixtures.BookObservationTests.setUp
    save_frame = fixtures.BookObservationTests.save_frame
    persist = fixtures.BookObservationTests.persist

    def book(self, **fields):
        """Save one legible spine and return its inventory entry.

        Args:
            **fields: Overrides applied to the inventory entry.

        Returns:
            The inventory entry.
        """
        return {**self.persist(books=[detection(title='Dune')])['inventory'][0], **fields}

    def resolve(self, book, *, label='c0', probability=0.9, volumes=VOLUMES):
        """Run a catalogue resolution against mocked sources.

        Args:
            book: Inventory entry to resolve.
            label: Drex choice label.
            probability: Probability of that label.
            volumes: Volumes response returned by the catalogue.
        """
        with patch.dict(os.environ, {'DREX_API_KEY': 'k'}):
            resolve_book(self.store.database, 'sweep', book,
                         http=client(lambda request: httpx.Response(200, json=volumes)),
                         drex=client(lambda request: httpx.Response(200, json=drex_choice(label, probability))))

    def catalogue_rows(self):
        """Read the catalogue observations saved for the sweep.

        Returns:
            Catalogue-provenance observations.
        """
        with Repository(self.store.database) as repo:
            return [row for row in repo.list_for_sweep(BookObservation, 'sweep') if row.provenance == 'catalogue']

    def test_resolve_writes_catalogue_observation(self):
        """Append a catalogue observation with its URL, retrieval time and no invented edition."""
        self.resolve(self.book())
        rows = self.catalogue_rows()
        with Repository(self.store.database) as repo:
            from library_claim.identification.storage import inventory
            entry = inventory(repo, 'sweep')[0]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].catalogue_url, 'https://books.google.com/vol1')
        self.assertIsNotNone(rows[0].retrieved_at)
        # The spine showed no publisher, so no edition or ISBN may be attached.
        self.assertIsNone(rows[0].isbn)
        self.assertIsNone(rows[0].edition)
        self.assertEqual(entry['catalogue']['title'], 'Dune')
        self.assertEqual(entry['title'], 'Dune')

    def test_isbn_needs_a_matching_spine_publisher(self):
        """Attach an ISBN only when the publisher read on the spine matches the record."""
        self.resolve(self.book(publisher='Ace Books'))
        rows = self.catalogue_rows()
        self.assertEqual(rows[0].isbn, '9780441172719')
        self.assertEqual(rows[0].edition, 'Dune (Ace Books)')

    def test_conflicting_spine_publisher_withholds_the_isbn(self):
        """Withhold the ISBN when the spine names a different publisher."""
        self.resolve(self.book(publisher='Hodder'))
        self.assertIsNone(self.catalogue_rows()[0].isbn)

    def test_no_confident_match_adds_review_issue(self):
        """Record why a legible book has no catalogue match."""
        self.resolve(self.book(), label='none')
        with Repository(self.store.database) as repo:
            reasons = [issue.reason for issue in repo.list_for_sweep(ReviewIssue, 'sweep')]
        self.assertIn('No confident catalogue match', reasons)

    def test_resolve_failure_writes_nothing(self):
        """Leave the store untouched when the catalogue fails."""
        book = self.book()
        resolve_book(self.store.database, 'sweep', book, http=client(lambda request: httpx.Response(503)),
                     drex=client(lambda request: httpx.Response(500)))
        with Repository(self.store.database) as repo:
            rows = repo.list_for_sweep(BookObservation, 'sweep')
            reasons = [issue.reason for issue in repo.list_for_sweep(ReviewIssue, 'sweep')]
        self.assertEqual(len(rows), 1)
        self.assertNotIn('No confident catalogue match', reasons)

    def test_catalogue_stage_reports_which_provider_answered(self):
        """Name the answering provider in the flow view's catalogue block."""
        from library_claim import telemetry
        events = []
        with telemetry.listening(events.append):
            self.resolve(self.book())
        providers = [event['model'] for event in events
                     if event['stage'] == 'catalogue' and event['phase'] == 'done']
        self.assertEqual(providers, ['Google Books'])
