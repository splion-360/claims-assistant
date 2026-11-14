"""Verify books are priced only from retrievable sources, in the policyholder's market."""
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import httpx

from library_claim.config import CONFIG
from library_claim.domain.models import Book, Frame, Sweep
from library_claim.pricing.storage import stored_valuations, value_book, value_sweep
from library_claim.storage.repository import Repository
from library_claim.pricing.markets import known_countries, market
from library_claim.pricing.models import Price
from library_claim.pricing.providers import PriceUnavailable, convert, exchange_rate, google_books_price
from library_claim.pricing.service import appraisal_signals, price_book


def volume(amount, currency, title='The Good People'):
    """Build a Google Books volume carrying a sale price.

    Args:
        amount: Retail amount.
        currency: Currency code of that amount.
        title: Volume title.

    Returns:
        One item of a Google Books volumes response.
    """
    return {'volumeInfo': {'title': title, 'infoLink': 'https://books.example/info'},
            'saleInfo': {'saleability': 'FOR_SALE', 'buyLink': 'https://books.example/buy',
                         'retailPrice': {'amount': amount, 'currencyCode': currency}}}


def responder(by_country, *, rate=1.3227):
    """Build a transport answering price and exchange-rate requests.

    Args:
        by_country: Volumes response items indexed by country code; a missing
            country answers with no items.
        rate: Exchange rate returned for any pair.

    Returns:
        A callable for httpx.MockTransport.
    """
    def handle(request):
        """Answer one recorded request.

        Args:
            request: The outgoing request.

        Returns:
            The matching JSON response.
        """
        if 'frankfurter' in request.url.host or '/latest' in request.url.path:
            target = request.url.params.get('symbols')
            return httpx.Response(200, json={'base': request.url.params.get('base'),
                                             'date': '2026-11-13', 'rates': {target: rate}})
        country = request.url.params.get('country')
        return httpx.Response(200, json={'items': by_country.get(country, [])})
    return handle


def client_for(handler):
    """Build an HTTP client backed by a mock transport.

    Args:
        handler: Request handler.

    Returns:
        A client making no network calls.
    """
    return httpx.Client(transport=httpx.MockTransport(handler))


def book(**fields):
    """Build an inventory entry for pricing.

    Args:
        **fields: Overrides such as title, author or catalogue.

    Returns:
        An inventory entry.
    """
    return {'id': 'b1', 'title': 'The Good People', 'author': 'Hannah Kent',
            'catalogue': None, 'observations': [], **fields}


class MarketTests(unittest.TestCase):
    """Check locale is configuration rather than code."""

    def test_configured_countries_resolve_to_their_currency(self):
        """Read each market's currency from configuration."""
        self.assertEqual(market('US').currency, 'USD')
        self.assertEqual(market('us').currency, 'USD')
        self.assertIn('US', known_countries())

    def test_unconfigured_country_is_refused(self):
        """Refuse a country with no configured market instead of assuming one."""
        with self.assertRaises(KeyError):
            market('ZZ')


class ProviderTests(unittest.TestCase):
    """Check what a price source is allowed to contribute."""

    def test_price_carries_its_source_and_retrieval_time(self):
        """Return the listed amount with the source and URL behind it."""
        with client_for(responder({'US': [volume(18.99, 'USD')]})) as http:
            price = google_books_price('The Good People', 'Hannah Kent', None, market('US'), client=http)
        self.assertEqual(price.amount, 18.99)
        self.assertEqual(price.currency, 'USD')
        self.assertEqual(price.source, 'Google Books')
        self.assertEqual(price.url, 'https://books.example/buy')
        self.assertIsNotNone(price.retrieved_at)
        self.assertFalse(price.converted)

    def test_country_is_sent_so_the_figure_is_local(self):
        """Ask the source for the policyholder's country, not a default one."""
        seen = []

        def handle(request):
            """Record the country asked for.

            Args:
                request: The outgoing request.

            Returns:
                An empty volumes response.
            """
            seen.append(request.url.params.get('country'))
            return httpx.Response(200, json={'items': []})

        with client_for(handle) as http:
            with self.assertRaises(PriceUnavailable):
                google_books_price('The Good People', None, None, market('GB'), client=http)
        self.assertEqual(seen, ['GB'])

    def test_isbn_is_preferred_over_the_text_search(self):
        """Query by ISBN when a catalogue match supplied one."""
        seen = []

        def handle(request):
            """Record the query sent.

            Args:
                request: The outgoing request.

            Returns:
                A volumes response with one priced item.
            """
            seen.append(request.url.params.get('q'))
            return httpx.Response(200, json={'items': [volume(18.99, 'USD')]})

        with client_for(handle) as http:
            google_books_price('The Good People', 'Hannah Kent', '9781760111236', market('US'), client=http)
        self.assertEqual(seen, ['isbn:9781760111236'])

    def test_quota_refusal_is_reported_as_unavailable(self):
        """Treat a refused request as no price rather than as a zero."""
        with client_for(lambda request: httpx.Response(429, json={'error': {}})) as http:
            with self.assertRaises(PriceUnavailable):
                google_books_price('The Good People', None, None, market('US'), client=http)

    def test_volume_without_a_price_yields_none(self):
        """Skip a volume the store does not sell instead of inventing a figure."""
        unpriced = {'volumeInfo': {'title': 'The Good People'}, 'saleInfo': {'saleability': 'NOT_FOR_SALE'}}
        with client_for(responder({'US': [unpriced]})) as http:
            with self.assertRaises(PriceUnavailable):
                google_books_price('The Good People', None, None, market('US'), client=http)


class ConversionTests(unittest.TestCase):
    """Check cross-market figures are labelled and traceable."""

    def test_rate_comes_with_its_publication_date_and_source(self):
        """Return the rate together with what published it and when."""
        with client_for(responder({})) as http:
            rate, published, source = exchange_rate('GBP', 'USD', client=http)
        self.assertEqual(rate, 1.3227)
        self.assertEqual(published, '2026-11-13')
        self.assertIn('base=GBP', source)

    def test_converted_price_is_marked_and_keeps_its_origin(self):
        """Mark a converted figure and record the market it came from."""
        original = Price(amount=10.0, currency='GBP', source='Google Books',
                         url='https://books.example/buy', retrieved_at=datetime.now(timezone.utc))
        with client_for(responder({})) as http:
            converted = convert(original, 'USD', client=http)
        self.assertTrue(converted.converted)
        self.assertEqual(converted.converted_from, 'GBP')
        self.assertEqual(converted.currency, 'USD')
        self.assertAlmostEqual(converted.amount, 13.227, places=3)
        self.assertIn('European Central Bank', converted.fx_source)


class ServiceTests(unittest.TestCase):
    """Check which books may be priced automatically."""

    def test_identified_book_is_priced_in_the_default_market(self):
        """Price a readable title in the configured default country."""
        with client_for(responder({'US': [volume(18.99, 'USD')]})) as http:
            valuation = price_book(book(), client=http)
        self.assertEqual(valuation.country, 'US')
        self.assertEqual(valuation.currency, 'USD')
        self.assertEqual(valuation.replacement.amount, 18.99)
        self.assertFalse(valuation.needs_appraisal)

    def test_same_book_prices_differently_in_a_second_market(self):
        """Return each market's own figure, proving locale is a setting."""
        prices = {'US': [volume(18.99, 'USD')], 'GB': [volume(9.99, 'GBP')]}
        with client_for(responder(prices)) as http:
            american = price_book(book(), 'US', client=http)
            british = price_book(book(), 'GB', client=http)
        self.assertEqual((american.replacement.amount, american.replacement.currency), (18.99, 'USD'))
        self.assertEqual((british.replacement.amount, british.replacement.currency), (9.99, 'GBP'))

    def test_missing_local_price_is_converted_and_labelled(self):
        """Convert a figure from a fallback market and mark it as converted."""
        with client_for(responder({'GB': [volume(10.0, 'GBP')]})) as http:
            valuation = price_book(book(), 'US', client=http)
        self.assertTrue(valuation.replacement.converted)
        self.assertEqual(valuation.replacement.currency, 'USD')
        self.assertEqual(valuation.replacement.converted_from, 'GBP')

    def test_no_price_anywhere_leaves_the_figure_empty_with_a_reason(self):
        """Say why a book has no price instead of writing one."""
        with client_for(responder({})) as http:
            valuation = price_book(book(), 'US', client=http)
        self.assertIsNone(valuation.replacement)
        self.assertTrue(valuation.reasons)

    def test_unidentified_book_is_never_priced(self):
        """Refuse to price a book whose spine was not read."""
        with client_for(responder({'US': [volume(18.99, 'USD')]})) as http:
            valuation = price_book(book(title=None), 'US', client=http)
        self.assertIsNone(valuation.replacement)
        self.assertIn('Unidentified', valuation.reasons[0])

    def test_signed_copy_is_withheld_from_automatic_pricing(self):
        """Flag a signed or rare copy for appraisal and make no lookup."""
        requested = []

        def handle(request):
            """Record any lookup attempt.

            Args:
                request: The outgoing request.

            Returns:
                A priced volumes response.
            """
            requested.append(request.url.path)
            return httpx.Response(200, json={'items': [volume(18.99, 'USD')]})

        with client_for(handle) as http:
            valuation = price_book(book(edition='Signed first edition'), 'US', client=http)
        self.assertTrue(valuation.needs_appraisal)
        self.assertIsNone(valuation.replacement)
        self.assertEqual(requested, [])

    def test_price_above_the_threshold_needs_an_appraiser(self):
        """Withhold a figure above the configured threshold from automatic use."""
        high = CONFIG.pricing.appraisal_threshold + 100
        with client_for(responder({'US': [volume(high, 'USD')]})) as http:
            valuation = price_book(book(), 'US', client=http)
        self.assertTrue(valuation.needs_appraisal)
        self.assertTrue(any('threshold' in reason for reason in valuation.reasons))

    def test_appraisal_keywords_are_read_from_observations(self):
        """Detect appraisal wording in what the spine readings actually said."""
        entry = book(observations=[{'title': 'The Good People, limited edition'}])
        self.assertTrue(appraisal_signals(entry))
        self.assertFalse(appraisal_signals(book()))

    def test_unconfigured_country_prices_nothing(self):
        """Price nothing for a country with no configured market."""
        with client_for(responder({'US': [volume(18.99, 'USD')]})) as http:
            valuation = price_book(book(), 'ZZ', client=http)
        self.assertIsNone(valuation.replacement)
        self.assertEqual(valuation.currency, '')


class StoredValuationTests(unittest.TestCase):
    """Check a book is priced once and the figure is reused until its reading changes."""

    def setUp(self):
        """Save a sweep holding one frame a valuation can be attributed to."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Path(self.directory.name) / 'claims.sqlite3'
        start = datetime(2026, 11, 14, 9, 0, tzinfo=timezone.utc)
        with Repository(self.database) as repo:
            repo.add_many([
                Sweep(id='sweep1', captured_at=start, device='browser camera'),
                Frame(id='f1', sweep_id='sweep1', captured_at=start, evidence_key=f'{"a" * 64}.jpg',
                      width_px=640, height_px=480),
                Book(id='b1', sweep_id='sweep1'),
            ])

    def counting_client(self, amount=18.99):
        """Build a client that counts the price lookups it answers.

        Args:
            amount: Retail amount returned.

        Returns:
            The client and a mutable list of the queries it saw.
        """
        seen = []

        def handle(request):
            """Answer a price lookup and record its query.

            Args:
                request: The outgoing request.

            Returns:
                A volumes response.
            """
            seen.append(request.url.params.get('q'))
            return httpx.Response(200, json={'items': [volume(amount, 'USD')]})

        return client_for(handle), seen

    def entry(self, **fields):
        """Build an inventory entry naming the saved frame.

        Args:
            **fields: Overrides such as title or catalogue.

        Returns:
            An inventory entry.
        """
        return {'id': 'b1', 'title': 'The Good People', 'author': 'Hannah Kent',
                'catalogue': None, 'observations': [], 'frame_ids': ['f1'], **fields}

    def test_second_view_reuses_the_saved_figure(self):
        """Price a book once and answer later reads from the saved job."""
        http, seen = self.counting_client()
        with http:
            first = value_book(self.database, 'sweep1', self.entry(), client=http)
            second = value_book(self.database, 'sweep1', self.entry(), client=http)
        self.assertEqual(len(seen), 1)
        self.assertEqual(first.replacement.amount, 18.99)
        # The retrieval time is the sweep's lookup, not the moment of the second read.
        self.assertEqual(second.replacement.retrieved_at, first.replacement.retrieved_at)

    def test_a_changed_reading_is_priced_again(self):
        """Price again when the title, author, ISBN or market changed."""
        http, seen = self.counting_client()
        with http:
            value_book(self.database, 'sweep1', self.entry(), client=http)
            value_book(self.database, 'sweep1', self.entry(title='Burial Rites'), client=http)
        self.assertEqual(len(seen), 2)

    def test_a_different_market_is_priced_again(self):
        """Price again for another country rather than reusing the first market's figure."""
        http, seen = self.counting_client()
        with http:
            value_book(self.database, 'sweep1', self.entry(), 'US', client=http)
            value_book(self.database, 'sweep1', self.entry(), 'GB', client=http)
        self.assertEqual(len(seen), 2)

    def test_saved_valuation_is_readable_from_the_sweep(self):
        """Expose the saved figure for the inventory to report."""
        http, _ = self.counting_client()
        with http:
            value_book(self.database, 'sweep1', self.entry(), client=http)
        with Repository(self.database) as repo:
            saved = stored_valuations(repo, 'sweep1')
        self.assertEqual(saved['b1']['replacement']['amount'], 18.99)

    def test_book_without_a_frame_is_not_valued(self):
        """Skip a book with no frame to attribute the figure to."""
        http, seen = self.counting_client()
        with http:
            self.assertIsNone(value_book(self.database, 'sweep1', self.entry(frame_ids=[]), client=http))
        self.assertEqual(seen, [])

    def test_a_book_no_source_can_price_does_not_stop_the_others(self):
        """Value every book, leaving an unreachable one empty with its reason."""
        def handle(request):
            """Refuse every lookup for one title and answer the other.

            Args:
                request: The outgoing request.

            Returns:
                A transport error or a volumes response.
            """
            if 'Good People' in request.url.params.get('q'):
                raise httpx.ConnectError('down')
            return httpx.Response(200, json={'items': [volume(18.99, 'USD')]})

        with Repository(self.database) as repo:
            repo.add(Book(id='b2', sweep_id='sweep1'))
        with client_for(handle) as http:
            valued = value_sweep(self.database, 'sweep1',
                                 [self.entry(), self.entry(id='b2', title='Burial Rites')], client=http)
        self.assertIsNone(valued['b1'].replacement)
        self.assertTrue(any('unreachable' in reason for reason in valued['b1'].reasons))
        self.assertEqual(valued['b2'].replacement.amount, 18.99)

    def test_a_fallback_market_answers_when_the_local_one_is_unreachable(self):
        """Use a fallback market when the local lookup cannot be made at all."""
        calls = []

        def handle(request):
            """Fail the first market, answer the next, and publish an exchange rate.

            Args:
                request: The outgoing request.

            Returns:
                A transport error, a volumes response or a rate response.
            """
            if '/latest' in request.url.path:
                return httpx.Response(200, json={'base': 'GBP', 'date': '2026-11-13', 'rates': {'USD': 1.3227}})
            calls.append(request.url.params.get('country'))
            if len(calls) == 1:
                raise httpx.ConnectError('down')
            return httpx.Response(200, json={'items': [volume(9.99, 'GBP')]})

        with client_for(handle) as http:
            valuation = value_book(self.database, 'sweep1', self.entry(), 'US', client=http)
        self.assertEqual(calls[:2], ['US', 'GB'])
        self.assertTrue(valuation.replacement.converted)
        self.assertEqual(valuation.replacement.currency, 'USD')
