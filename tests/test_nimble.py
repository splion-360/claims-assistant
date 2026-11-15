"""Verify store listings are read from page HTML exactly as shown, and priced only when they match."""
import json
import unittest
from unittest.mock import patch

import httpx

from library_claim.pricing.markets import market
from library_claim.pricing.nimble import (Listing, fetch_page, main_title, parse_abebooks, parse_amazon,
                                          parse_money, same_title)
from library_claim.pricing.providers import PriceUnavailable, abebooks_used_price, amazon_book_price
from library_claim.pricing.service import price_book
from test_fetch import quiet_sources


def amazon_result(asin, title, price, sponsored=False):
    """Build one Amazon search result block in the page's own markup.

    Args:
        asin: Product ID.
        title: Product title.
        price: Printed price, e.g. "£19.99".
        sponsored: Whether the result is an ad.

    Returns:
        HTML for the result.
    """
    label = '<span>Sponsored</span>' if sponsored else ''
    return (f'<div data-asin="{asin}" data-index="1" data-component-type="s-search-result" class="s-result-item">'
            f'{label}<h2 class="a-size-base"><span>{title}</span></h2>'
            f'<span class="a-price"><span class="a-offscreen">{price}</span></span></div>')


def abebooks_offer(index, title, price, condition='Used - Good', href='/Dune-Herbert/1/bd'):
    """Build one AbeBooks offer block in the page's own markup.

    Args:
        index: Position on the page.
        title: Listed title.
        price: Printed price, e.g. "US$ 5.12".
        condition: Printed condition.
        href: Offer link.

    Returns:
        HTML for the offer.
    """
    return (f'<li data-test-id="listing-item-{index}"><a data-test-id="listing-title-link" href="{href}">'
            f'<h2 data-test-id="listing-title">{title}</h2></a>'
            f'<span data-test-id="listing-condition">{condition}</span>'
            f'<span data-test-id="listing-price">{price}</span></li>')


def nimble(html, status='success'):
    """Build a client answering every page fetch with the given HTML.

    Args:
        html: Page HTML to return.
        status: Nimble task status.

    Returns:
        A client and the list of request bodies it saw.
    """
    seen = []

    def handle(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={'status': status, 'data': {'html': html}})
    return httpx.Client(transport=httpx.MockTransport(handle)), seen


class ParsingTests(unittest.TestCase):
    """Prices, currencies and titles are read literally."""

    def test_money(self):
        """Printed prices become amounts with ISO currencies; anything else is unreadable."""
        self.assertEqual(parse_money('£1,299.00'), (1299.0, 'GBP'))
        self.assertEqual(parse_money('US$ 5.12'), (5.12, 'USD'))
        self.assertEqual(parse_money('$19.99'), (19.99, 'USD'))
        self.assertIsNone(parse_money('Free'))
        self.assertIsNone(parse_money('£12.5x'))

    def test_titles(self):
        """A sequel is not the same work; a format note is."""
        self.assertEqual(main_title('Dune (Dune Chronicles, Book 1)'), 'dune')
        self.assertTrue(same_title('Dune: Deluxe Edition', 'Dune'))
        self.assertFalse(same_title('Dune Messiah', 'Dune'))
        self.assertFalse(same_title('Anything', ''))

    def test_amazon_results(self):
        """Results keep page order, drop repeats and other currencies, and link to the product page."""
        html = (amazon_result('B000000001', 'Brass Floor Lamp', '£59.99', sponsored=True)
                + amazon_result('B000000001', 'Brass Floor Lamp', '£59.99')
                + amazon_result('B000000002', 'Floor Lamp &amp; Shade', '$20.00')
                + amazon_result('B000000003', 'Arc Lamp', '£120.00'))
        listings = parse_amazon(html, 'www.amazon.co.uk', 'GBP')
        self.assertEqual([listing.amount for listing in listings], [59.99, 120.0])
        self.assertEqual(listings[0].url, 'https://www.amazon.co.uk/dp/B000000001')
        self.assertTrue(listings[0].sponsored)

    def test_abebooks_offers(self):
        """Offers carry their condition and an absolute link."""
        html = abebooks_offer(0, 'Dune', 'US$ 6.23') + abebooks_offer(1, 'Dune', 'Free')
        listings = parse_abebooks(html, 'www.abebooks.com', 'USD')
        self.assertEqual(listings, [Listing('Dune', 6.23, 'USD', 'https://www.abebooks.com/Dune-Herbert/1/bd',
                                            'Used - Good')])


class FetchPageTests(unittest.TestCase):
    """The web listing source is only called with a key, and a failed page is transient."""

    def setUp(self):
        """Close breakers."""
        quiet_sources(self)

    def test_missing_key_sends_nothing(self):
        """Without NIMBLE_API_KEY no request is made and the reason says so."""
        http, seen = nimble('<html></html>')
        with patch.dict('os.environ', {'NIMBLE_API_KEY': ''}), self.assertRaises(PriceUnavailable) as caught:
            fetch_page('https://www.amazon.com/s?k=x', 'US', client=http)
        self.assertEqual(seen, [])
        self.assertFalse(caught.exception.transient)

    def test_failed_page_is_transient(self):
        """A page the service could not load is worth retrying later."""
        http, _ = nimble('', status='failed')
        with patch.dict('os.environ', {'NIMBLE_API_KEY': 'k'}), self.assertRaises(PriceUnavailable) as caught:
            fetch_page('https://www.amazon.com/s?k=x', 'US', client=http)
        self.assertTrue(caught.exception.transient)

    def test_request_names_the_country(self):
        """The page is fetched as a visitor in the market's country."""
        http, seen = nimble('<html></html>')
        with patch.dict('os.environ', {'NIMBLE_API_KEY': 'k'}):
            fetch_page('https://www.amazon.co.uk/s?k=x', 'GB', client=http)
        self.assertEqual(seen[0]['country'], 'GB')


class BookProviderTests(unittest.TestCase):
    """Amazon gives a new-copy price and AbeBooks a used value, only for the same work."""

    def setUp(self):
        """Close breakers and set a stand-in key."""
        quiet_sources(self)
        guard = patch.dict('os.environ', {'NIMBLE_API_KEY': 'k'})
        guard.start()
        self.addCleanup(guard.stop)

    def test_amazon_takes_the_first_listing_of_the_same_work(self):
        """A sequel listed first is skipped."""
        http, seen = nimble(amazon_result('B000000001', 'Dune Messiah', '$8.00')
                            + amazon_result('B000000002', 'Dune (Dune Chronicles, Book 1)', '$10.99'))
        price = amazon_book_price('Dune', 'Frank Herbert', None, market('US'), client=http)
        self.assertEqual((price.amount, price.currency), (10.99, 'USD'))
        self.assertEqual(price.url, 'https://www.amazon.com/dp/B000000002')
        self.assertIn('amazon.com/s?k=Dune', seen[0]['url'])

    def test_amazon_without_a_match_has_no_price(self):
        """No listing of the same work means no figure."""
        http, _ = nimble(amazon_result('B000000001', 'Dune Messiah', '$8.00'))
        with self.assertRaises(PriceUnavailable) as caught:
            amazon_book_price('Dune', None, None, market('US'), client=http)
        self.assertFalse(caught.exception.transient)

    def test_used_value_is_the_median_of_matching_used_offers(self):
        """New offers and other works are ignored; the median of the rest is quoted with its basis."""
        html = ''.join([abebooks_offer(0, 'Dune', 'US$ 4.00'), abebooks_offer(1, 'Dune Messiah', 'US$ 1.00'),
                        abebooks_offer(2, 'Dune', 'US$ 6.00', condition='New'), abebooks_offer(3, 'Dune', 'US$ 7.00'),
                        abebooks_offer(4, 'Dune (Movie Tie-In)', 'US$ 9.00', condition='Used - Fair')])
        http, seen = nimble(html)
        price = abebooks_used_price('Dune', 'Frank Herbert', None, market('US'), client=http)
        self.assertEqual(price.amount, 7.0)
        self.assertEqual(price.kind, 'used')
        self.assertIn('median of 3', price.condition_assumed)
        self.assertTrue(price.url.startswith('https://www.abebooks.com/servlet/SearchResults?tn=Dune'))

    def test_too_few_used_offers_gives_no_value(self):
        """One offer is not a market value."""
        http, _ = nimble(abebooks_offer(0, 'Dune', 'US$ 4.00'))
        with self.assertRaises(PriceUnavailable):
            abebooks_used_price('Dune', None, None, market('US'), client=http)

    def test_price_book_uses_fallbacks_and_reports_both_figures(self):
        """Google Books has no price, Amazon gives replacement, AbeBooks gives used value."""
        def handle(request):
            if 'googleapis' in str(request.url):
                return httpx.Response(200, json={'items': []})
            url = json.loads(request.content)['url']
            if 'amazon' in url:
                html = amazon_result('B000000002', 'Dune', '$10.99')
            else:
                html = ''.join(abebooks_offer(n, 'Dune', f'US$ {n + 4}.00') for n in range(3))
            return httpx.Response(200, json={'status': 'success', 'data': {'html': html}})
        book = {'id': 'b1', 'title': 'Dune', 'author': 'Frank Herbert', 'catalogue': None, 'observations': []}
        with httpx.Client(transport=httpx.MockTransport(handle)) as http:
            valuation = price_book(book, 'US', client=http)
        self.assertEqual(valuation.replacement.amount, 10.99)
        self.assertIn('Amazon', valuation.replacement.source)
        self.assertEqual(valuation.used.amount, 5.0)
        self.assertFalse(valuation.needs_appraisal)


if __name__ == '__main__':
    unittest.main()
