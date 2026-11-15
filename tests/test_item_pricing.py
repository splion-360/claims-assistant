"""Verify non-book contents are priced only from store listings Drex accepts, and saved once."""
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import httpx

from library_claim.domain.models import Frame, Sweep
from library_claim.items.pricing import price_item, value_item
from library_claim.storage.repository import Repository
from test_fetch import quiet_sources
from test_nimble import amazon_result

LISTINGS = (amazon_result('B000000001', 'Brass Floor Lamp 160cm', '$80.00')
            + amazon_result('B000000002', 'Replacement Lamp Shade', '$15.00')
            + amazon_result('B000000003', 'Arc Floor Lamp, Brass Finish', '$140.00'))


def store(html=LISTINGS):
    """Build a client serving one store page through the web listing source.

    Args:
        html: Page HTML.

    Returns:
        A client and the list of fetched page URLs.
    """
    urls = []

    def handle(request):
        urls.append(json.loads(request.content)['url'])
        return httpx.Response(200, json={'status': 'success', 'data': {'html': html}})
    return httpx.Client(transport=httpx.MockTransport(handle)), urls


def drex(probabilities):
    """Build a Drex client answering one probability per listing.

    Args:
        probabilities: noul answers for l1, l2, ...

    Returns:
        A client and the list of request bodies.
    """
    bodies = []

    def handle(request):
        body = json.loads(request.content)
        bodies.append(body)
        answers = {key: {'type': 'noul', 'noul': probabilities[int(key[1:]) - 1], 'confidence': 0.9}
                   for key in body['questions']}
        return httpx.Response(200, json={'model': 'drex-max-v1.1', 'answers': answers,
                                         'usage': {'input_tokens': 10}, 'request_id': 'r'})
    return httpx.Client(transport=httpx.MockTransport(handle)), bodies


def lamp(**fields):
    """Build an object inventory entry.

    Args:
        **fields: Overrides.

    Returns:
        Inventory entry.
    """
    return {'id': 'i1', 'category': 'lighting', 'description': 'brass floor lamp', 'material': 'brass',
            'brand_model': None, 'stated_print': False, 'frame_ids': ['f1'], **fields}


class ItemPricingTests(unittest.TestCase):
    """Ranges, exact products, rejections and appraisal."""

    def setUp(self):
        """Use stand-in keys and closed breakers."""
        quiet_sources(self)
        guard = patch.dict('os.environ', {'NIMBLE_API_KEY': 'k', 'DREX_API_KEY': 'k'})
        guard.start()
        self.addCleanup(guard.stop)

    def test_unbranded_object_gets_a_range_over_accepted_listings(self):
        """The shade Drex rejects is not part of the range."""
        http, urls = store()
        decider, bodies = drex([0.9, 0.05, 0.8])
        valuation = price_item(lamp(), 'US', client=http, drex=decider)
        self.assertEqual((valuation.replacement.amount, valuation.replacement_high.amount), (80.0, 140.0))
        self.assertEqual(valuation.replacement.url, 'https://www.amazon.com/dp/B000000001')
        self.assertIn('range over 2 comparable listings', valuation.replacement.source)
        self.assertIn('brass+floor+lamp', urls[0])
        self.assertIn('comparable replacement', bodies[0]['questions']['l1']['instructions'])

    def test_branded_object_is_priced_at_the_first_matching_product(self):
        """A legible model is searched by name and priced at one listing."""
        http, urls = store()
        decider, bodies = drex([0.1, 0.0, 0.95])
        valuation = price_item(lamp(brand_model='Lumina Arc 2'), 'US', client=http, drex=decider)
        self.assertEqual(valuation.replacement.amount, 140.0)
        self.assertEqual(valuation.replacement_high.amount, 140.0)
        self.assertIn('Lumina+Arc+2', urls[0])
        self.assertIn('same product', bodies[0]['questions']['l1']['instructions'])

    def test_no_accepted_listing_leaves_no_figure(self):
        """When Drex accepts nothing, nothing is priced and the reason says so."""
        http, _ = store()
        decider, _ = drex([0.1, 0.1, 0.2])
        valuation = price_item(lamp(), 'US', client=http, drex=decider)
        self.assertIsNone(valuation.replacement)
        self.assertTrue(any('matched the object' in reason for reason in valuation.reasons))

    def test_drex_outage_is_retried_later(self):
        """Without a decision no listing is trusted, and the lookup is marked for retry."""
        http, _ = store()
        down = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(401)))
        valuation = price_item(lamp(), 'US', client=http, drex=down)
        self.assertIsNone(valuation.replacement)
        self.assertTrue(valuation.retry_later)

    def test_art_is_never_looked_up(self):
        """Framed art goes to appraisal without fetching any page."""
        http, urls = store()
        valuation = price_item(lamp(category='art', description='framed portrait'), 'US', client=http)
        self.assertTrue(valuation.needs_appraisal)
        self.assertEqual(urls, [])

    def test_expensive_object_is_withheld_for_appraisal(self):
        """A listing above the contents threshold sends the object to an appraiser."""
        http, _ = store(amazon_result('B000000009', 'Brass Floor Lamp, designer', '$9,500.00'))
        decider, _ = drex([0.9])
        valuation = price_item(lamp(), 'US', client=http, drex=decider)
        self.assertTrue(valuation.needs_appraisal)

    def test_gb_uses_the_uk_store_and_pounds(self):
        """The UK market searches amazon.co.uk and reads prices in pounds."""
        html = amazon_result('B000000001', 'Brass Floor Lamp', '£65.00')
        http, urls = store(html)
        decider, _ = drex([0.9])
        valuation = price_item(lamp(), 'GB', client=http, drex=decider)
        self.assertTrue(urls[0].startswith('https://www.amazon.co.uk/'))
        self.assertEqual((valuation.currency, valuation.replacement.currency), ('GBP', 'GBP'))


class StoredItemValuationTests(unittest.TestCase):
    """An object is priced once and its figure reused."""

    def setUp(self):
        """Save a sweep with a frame to attribute the valuation to."""
        quiet_sources(self)
        guard = patch.dict('os.environ', {'NIMBLE_API_KEY': 'k', 'DREX_API_KEY': 'k'})
        guard.start()
        self.addCleanup(guard.stop)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.database = Path(directory.name) / 'claims.sqlite3'
        start = datetime(2026, 11, 14, 9, tzinfo=timezone.utc)
        with Repository(self.database) as repo:
            repo.add_many([Sweep(id='s', captured_at=start, device='browser camera'),
                           Frame(id='f1', sweep_id='s', captured_at=start, evidence_key=f'{"a" * 64}.jpg',
                                 width_px=640, height_px=480)])

    def test_second_view_reuses_the_saved_figure(self):
        """No second page fetch for an unchanged reading."""
        http, urls = store()
        decider, _ = drex([0.9, 0.05, 0.8])
        first = value_item(self.database, 's', lamp(), 'US', client=http, drex=decider)
        second = value_item(self.database, 's', lamp(), 'US', client=http, drex=decider)
        self.assertEqual(len(urls), 1)
        self.assertEqual(second.replacement, first.replacement)

    def test_failed_lookup_is_retried(self):
        """A lookup that timed out is attempted again on the next view."""
        def down(request):
            raise httpx.ConnectError('down', request=request)
        with httpx.Client(transport=httpx.MockTransport(down)) as http:
            value_item(self.database, 's', lamp(), 'US', client=http)
        quiet_sources(self)
        with patch.dict('os.environ', {'NIMBLE_API_KEY': 'k', 'DREX_API_KEY': 'k'}):
            http, urls = store()
            decider, _ = drex([0.9, 0.05, 0.8])
            valuation = value_item(self.database, 's', lamp(), 'US', client=http, drex=decider)
        self.assertEqual(valuation.replacement.amount, 80.0)


if __name__ == '__main__':
    unittest.main()
