"""Verify the end-of-sweep step resumes unfinished work and survives failing sources."""
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import httpx

from library_claim.domain.models import Book, BookObservation, Box, Frame, ReviewIssue, Sweep
from library_claim.finish import finish_sweep
from library_claim.packet import build_packet
from library_claim.storage.repository import Repository
from test_fetch import quiet_sources

VOLUME = {'volumeInfo': {'title': 'The Good People', 'infoLink': 'https://books.example/info'},
          'saleInfo': {'buyLink': 'https://books.example/buy', 'retailPrice': {'amount': 18.99, 'currencyCode': 'USD'}}}


def transport(handler):
    """Build a client over a mock transport.

    Args:
        handler: Request handler.

    Returns:
        An httpx client making no network calls.
    """
    return httpx.Client(transport=httpx.MockTransport(handler))


def down(request):
    """Fail every request as unreachable."""
    raise httpx.ConnectError('down', request=request)


FAULTS = {
    'unreachable': down,
    'timeout': lambda request: (_ for _ in ()).throw(httpx.ReadTimeout('slow', request=request)),
    'rate_limited': lambda request: httpx.Response(429),
    'server_error': lambda request: httpx.Response(503),
    'junk': lambda request: httpx.Response(200, content=b'<html>maintenance</html>'),
    'wrong_shape': lambda request: httpx.Response(200, json={'items': [{'saleInfo': {'retailPrice': 'free'}}, 'x']}),
    'json_list': lambda request: httpx.Response(200, json=['unexpected']),
}


class FinishTests(unittest.TestCase):
    """finish_sweep completes what the live loop left, and never invents a value."""

    def setUp(self):
        """Save a sweep with one readable book and no lookups done yet."""
        quiet_sources(self)
        guard = patch.dict('os.environ', {'DREX_API_KEY': ''})
        guard.start()
        self.addCleanup(guard.stop)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.database = self.root / 'claims.sqlite3'
        start = datetime(2026, 11, 14, 9, 0, tzinfo=timezone.utc)
        with Repository(self.database) as repo:
            repo.add_many([
                Sweep(id='s', captured_at=start, device='browser camera', country='US', currency='USD'),
                Frame(id='f1', sweep_id='s', captured_at=start, evidence_key=f'{"a" * 64}.jpg',
                      width_px=640, height_px=480),
                Book(id='b1', sweep_id='s'),
                BookObservation(sweep_id='s', book_id='b1', frame_id='f1', bbox=Box(x_min=.1, y_min=.1, x_max=.5, y_max=.9),
                                association_method='new_provisional', title='The Good People', author='Hannah Kent'),
            ])
        self.resolved, self.measured = [], []

    def finish(self, http):
        """Run finish_sweep with recording stand-ins for catalogue and room steps.

        Args:
            http: Client for price sources.

        Returns:
            The finish summary.
        """
        return finish_sweep(self.database, self.root, 's', http=http,
                            resolve=lambda database, sweep, book, **_: self.resolved.append(book['id']),
                            measure=lambda database, data_dir, sweep, **_: self.measured.append(sweep))

    def test_every_fault_still_produces_a_packet_with_nothing_invented(self):
        """Each kind of source failure leaves the price empty, flagged, and out of the totals."""
        for name, handler in FAULTS.items():
            with self.subTest(fault=name):
                quiet_sources(self)
                with transport(handler) as http:
                    summary = self.finish(http)
                    packet = build_packet(self.database, 's', client=http)
                line = packet['books'][0]
                self.assertEqual(line['title'], 'The Good People')
                self.assertIsNone(line['replacement_cost']['amount'])
                self.assertEqual(packet['totals']['books_replacement_cost'], 0)
                self.assertEqual(packet['totals']['excluded_from_totals'], 1)
                self.assertTrue(any(entry['ref_id'] == 'b1' for entry in packet['review_queue']))
                self.assertIn('priced', summary)

    def test_failed_pricing_resumes_when_the_source_returns(self):
        """A run during an outage is completed by the next run."""
        with transport(down) as http:
            self.finish(http)
        quiet_sources(self)
        with transport(lambda request: httpx.Response(200, json={'items': [VOLUME]})) as http:
            summary = self.finish(http)
            packet = build_packet(self.database, 's', client=http)
        self.assertEqual(summary['priced'], 1)
        self.assertEqual(packet['books'][0]['replacement_cost']['amount'], 18.99)

    def test_unresolved_books_are_looked_up_once(self):
        """A titled book with no catalogue outcome is resolved; one already answered is not."""
        with transport(down) as http:
            self.finish(http)
        self.assertEqual(self.resolved, ['b1'])
        with Repository(self.database) as repo:
            repo.add(ReviewIssue(sweep_id='s', book_id='b1', field='catalogue', reason='No confident catalogue match'))
        self.resolved.clear()
        with transport(down) as http:
            self.finish(http)
        self.assertEqual(self.resolved, [])

    def test_room_is_measured_when_missing(self):
        """No saved room estimate means the room step runs."""
        with transport(down) as http:
            self.finish(http)
        self.assertEqual(self.measured, ['s'])

    def test_a_crashing_step_does_not_stop_the_others(self):
        """A catalogue step that raises still lets pricing and the room run."""
        def broken(*args, **kwargs):
            raise RuntimeError('boom')
        with transport(lambda request: httpx.Response(200, json={'items': [VOLUME]})) as http:
            summary = finish_sweep(self.database, self.root, 's', http=http, resolve=broken,
                                   measure=lambda *args, **kwargs: self.measured.append('s'))
        self.assertEqual(summary['priced'], 1)
        self.assertEqual(self.measured, ['s'])
        self.assertTrue(summary['errors'])


if __name__ == '__main__':
    unittest.main()
