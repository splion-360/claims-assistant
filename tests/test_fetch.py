"""Verify outside sources are called with bounded retries, a circuit breaker and a daily cache."""
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import httpx

from library_claim.pricing import fetch
from library_claim.pricing.cache import PriceCache
from library_claim.pricing.models import Price
from library_claim.pricing.providers import PriceUnavailable


def client(handler):
    """Build a client over a mock transport.

    Args:
        handler: Request handler.

    Returns:
        An httpx client.
    """
    return httpx.Client(transport=httpx.MockTransport(handler))


def quiet_sources(case: unittest.TestCase) -> None:
    """Give a test closed breakers and no real backoff sleeps.

    Args:
        case: The running test; cleanups are registered on it.
    """
    fetch.reset_breakers()
    case.addCleanup(fetch.reset_breakers)
    sleeper = patch.object(fetch.time, 'sleep')
    sleeper.start()
    case.addCleanup(sleeper.stop)


class FetchTests(unittest.TestCase):
    """Transient failures are retried once; persistent ones open the breaker."""

    def setUp(self):
        """Reset breakers and skip real backoff sleeps."""
        fetch.reset_breakers()
        sleeper = patch.object(fetch.time, 'sleep')
        self.sleep = sleeper.start()
        self.addCleanup(sleeper.stop)

    def test_timeout_then_success_is_retried(self):
        """A single timeout is retried and the second answer is used."""
        calls = []

        def handler(request):
            calls.append(request)
            if len(calls) == 1:
                raise httpx.ReadTimeout('slow', request=request)
            return httpx.Response(200, json={'ok': True})
        self.assertEqual(fetch.request_json(client(handler), 'GET', 'https://a.example/x', source='A'), {'ok': True})
        self.assertEqual(len(calls), 2)

    def test_repeated_rate_limit_is_transient(self):
        """Two 429s give up with a transient error after exactly one retry."""
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(429, headers={'Retry-After': '30'})
        with self.assertRaises(PriceUnavailable) as caught:
            fetch.request_json(client(handler), 'GET', 'https://a.example/x', source='A')
        self.assertTrue(caught.exception.transient)
        self.assertEqual(len(calls), 2)
        # A long Retry-After is capped so one source cannot stall the packet.
        self.assertLessEqual(self.sleep.call_args.args[0], 2.0)

    def test_junk_body_is_not_retried(self):
        """An unreadable body is a definite failure, not a transient one."""
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(200, content=b'<html>not json')
        with self.assertRaises(PriceUnavailable) as caught:
            fetch.request_json(client(handler), 'GET', 'https://a.example/x', source='A')
        self.assertFalse(caught.exception.transient)
        self.assertEqual(len(calls), 1)

    def test_client_error_is_not_retried(self):
        """A 400 is a definite failure."""
        handler = lambda request: httpx.Response(400)  # noqa: E731
        with self.assertRaises(PriceUnavailable) as caught:
            fetch.request_json(client(handler), 'GET', 'https://a.example/x', source='A')
        self.assertFalse(caught.exception.transient)

    def test_breaker_skips_a_failing_source(self):
        """After repeated transient failures the source is skipped without a request."""
        calls = []

        def handler(request):
            calls.append(request)
            raise httpx.ConnectError('down', request=request)
        for _ in range(fetch.CONFIG.pricing.breaker_failures):
            with self.assertRaises(PriceUnavailable):
                fetch.request_json(client(handler), 'GET', 'https://a.example/x', source='A')
        before = len(calls)
        with self.assertRaises(PriceUnavailable) as caught:
            fetch.request_json(client(handler), 'GET', 'https://a.example/x', source='A')
        self.assertEqual(len(calls), before)
        self.assertTrue(caught.exception.transient)
        self.assertIn('skipped', str(caught.exception))
        # Another source is unaffected.
        ok = client(lambda request: httpx.Response(200, json={}))
        self.assertEqual(fetch.request_json(ok, 'GET', 'https://b.example/x', source='B'), {})


class CacheTests(unittest.TestCase):
    """A price looked up today is reused today, with its original retrieval time."""

    def setUp(self):
        """Open a cache in a temporary folder."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.cache = PriceCache(Path(directory.name) / 'price_cache.sqlite3')
        self.price = Price(amount=9.5, currency='USD', source='S', url='https://s.example',
                           retrieved_at=datetime(2026, 11, 15, 9, tzinfo=timezone.utc))

    def test_round_trip_same_day(self):
        """A stored price comes back unchanged on the same day."""
        key = ('google_books', 'Dune', 'Herbert', '', 'US')
        self.cache.put(key, self.price)
        self.assertEqual(self.cache.get(key), self.price)

    def test_no_price_answer_is_remembered(self):
        """A definite 'no listing' answer is cached as None, distinct from a miss."""
        key = ('google_books', 'Obscure', '', '', 'US')
        self.assertIs(self.cache.get(key), PriceCache.MISS)
        self.cache.put(key, None, reason='No US price listed')
        self.assertIsNone(self.cache.get(key))
        self.assertEqual(self.cache.reason(key), 'No US price listed')

    def test_entries_expire_the_next_day(self):
        """Yesterday's price is not reused."""
        key = ('google_books', 'Dune', '', '', 'US')
        self.cache.put(key, self.price)
        tomorrow = datetime.now(timezone.utc) + timedelta(days=1)
        with patch.object(PriceCache, '_today', return_value=tomorrow.date().isoformat()):
            self.assertIs(self.cache.get(key), PriceCache.MISS)


if __name__ == '__main__':
    unittest.main()
