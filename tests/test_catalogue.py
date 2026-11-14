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


LOC_RECORD = '''<?xml version="1.0"?>
<zs:searchRetrieveResponse xmlns:zs="http://www.loc.gov/zing/srw/"><zs:numberOfRecords>1</zs:numberOfRecords><zs:records><zs:record><zs:recordData>
<mods xmlns="http://www.loc.gov/mods/v3">
  <titleInfo><title>Did I say you could go</title></titleInfo>
  <name type="personal" usage="primary"><namePart>Gideon, Melanie,</namePart><role><roleTerm type="text">author</roleTerm></role></name>
  <originInfo><dateIssued encoding="marc">2023</dateIssued><edition>First edition.</edition></originInfo>
  <originInfo eventType="publication"><agent><namePart>Berkley,</namePart><role><roleTerm>publisher</roleTerm></role></agent></originInfo>
  <identifier type="isbn">9780593546994 (hardcover)</identifier>
  <identifier type="lccn">2022036391</identifier>
</mods></zs:recordData></zs:record></zs:records></zs:searchRetrieveResponse>'''


class LibraryOfCongressFallbackTests(unittest.TestCase):
    """The Library of Congress catalogue is the keyless fallback after Open Library."""

    setUp = fixtures.BookObservationTests.setUp
    save_frame = fixtures.BookObservationTests.save_frame
    persist = fixtures.BookObservationTests.persist

    def test_search_parses_catalogue_records(self):
        """Query the catalogue by title and author and keep link, publisher, edition and ISBN."""
        from library_claim.identification.catalogue import search_library_of_congress

        def handler(request):
            self.assertEqual(request.url.host, 'lx2.loc.gov')
            self.assertEqual(request.url.params['query'], 'dc.title="Did I Say You Could Go" and dc.creator="Gideon"')
            return httpx.Response(200, text=LOC_RECORD)

        [candidate] = search_library_of_congress('Did I Say You Could Go', 'Gideon', client=client(handler))
        self.assertEqual((candidate.source, candidate.title, candidate.authors, candidate.first_publish_year),
                         ('loc', 'Did I say you could go', ['Gideon, Melanie'], 2023))
        self.assertEqual((candidate.url, candidate.publisher, candidate.isbn, candidate.edition),
                         ('https://lccn.loc.gov/2022036391', 'Berkley', '9780593546994', 'First edition.'))

    def resolve(self, publisher=None, open_library=None):
        """Resolve one legible book when Open Library has no match.

        Args:
            publisher: Publisher read from the spine.
            open_library: Optional response factory for Open Library requests.

        Returns:
            Catalogue observations written and the hosts that were queried.
        """
        hosts = []
        book = self.persist(books=[detection(title='Did I Say You Could Go')])['inventory'][0]
        book['publisher'] = publisher

        def http(request):
            hosts.append(request.url.host)
            if request.url.host == 'lx2.loc.gov':
                return httpx.Response(200, text=LOC_RECORD)
            return open_library(request) if open_library else httpx.Response(200, json=SEARCH)

        def drex(request):
            criteria = json.loads(request.content)['questions']['match']['criteria']
            return httpx.Response(200, json=drex_choice('c0' if 'Gideon' in criteria.get('c0', '') else 'none', 0.9))

        with patch.dict(os.environ, {'DREX_API_KEY': 'k'}):
            resolve_book(self.store.database, 'sweep', book, http=client(http), drex=client(drex))
        with Repository(self.store.database) as repo:
            rows = [row for row in repo.list_for_sweep(BookObservation, 'sweep') if row.provenance == 'catalogue']
        return rows, hosts

    def test_library_of_congress_answers_when_open_library_has_no_match(self):
        """Library of Congress matches when Drex rejects every Open Library work."""
        rows, hosts = self.resolve()
        self.assertEqual(rows[0].catalogue_url, 'https://lccn.loc.gov/2022036391')
        self.assertEqual(hosts[-1], 'lx2.loc.gov')
        self.assertIsNone(rows[0].isbn)

    def test_open_library_outage_falls_back_to_library_of_congress(self):
        """An Open Library failure still lets the Library of Congress answer."""
        rows, _ = self.resolve(open_library=lambda request: httpx.Response(503))
        self.assertEqual(rows[0].catalogue_url, 'https://lccn.loc.gov/2022036391')

    def test_catalogue_stage_reports_which_provider_answered(self):
        """The flow view's catalogue block names the provider of each search."""
        from library_claim import telemetry
        events = []
        with telemetry.listening(events.append):
            self.resolve()
        providers = [e['model'] for e in events if e['stage'] == 'catalogue' and e['phase'] == 'done']
        self.assertEqual(providers, ['Open Library', 'Library of Congress'])

    def test_catalogue_edition_and_isbn_need_matching_spine_publisher(self):
        """Edition and ISBN come from the record only when the spine's publisher matches."""
        rows, _ = self.resolve(publisher='Berkley')
        self.assertEqual((rows[0].edition, rows[0].isbn), ('First edition.', '9780593546994'))

    def test_catalogue_order_is_open_library_then_library_of_congress(self):
        """Only the two keyless catalogues are used, in this order."""
        from library_claim.identification.catalogue import catalogue_sources
        self.assertEqual([name for name, _ in catalogue_sources()], ['Open Library', 'Library of Congress'])
