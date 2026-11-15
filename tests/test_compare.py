"""Verify the same books are priced independently in a second market."""
import unittest
from unittest.mock import patch

import httpx

from library_claim.pricing.compare import compare_markets
from test_fetch import quiet_sources


def google(request):
    """Answer Google Books with a price in the requested country's currency."""
    country = request.url.params.get('country')
    amount, currency = (10.0, 'USD') if country == 'US' else (8.0, 'GBP')
    return httpx.Response(200, json={'items': [{'volumeInfo': {'title': 'x'},
                                                'saleInfo': {'buyLink': f'https://books.example/{country}',
                                                             'retailPrice': {'amount': amount, 'currencyCode': currency}}}]})


class CompareTests(unittest.TestCase):
    """Each market answers in its own currency, for the same books in the same order."""

    def setUp(self):
        """Close breakers; the web listing source stays unconfigured."""
        quiet_sources(self)

    def test_same_books_in_both_markets(self):
        """Ten identified books are compared; unidentified and excluded ones are skipped."""
        books = ([{'id': 'u', 'title': ''}, {'id': 'x', 'title': 'Gone', 'excluded': True}]
                 + [{'id': f'b{n}', 'title': f'Book {n}', 'catalogue': None, 'observations': []} for n in range(12)])
        with httpx.Client(transport=httpx.MockTransport(google)) as http:
            result = compare_markets(books, ['US', 'GB'], client=http)
        self.assertEqual(result['countries'], ['US', 'GB'])
        self.assertEqual([row['id'] for row in result['books']], [f'b{n}' for n in range(10)])
        first = result['books'][0]['prices']
        self.assertEqual((first['US']['replacement']['amount'], first['US']['currency']), (10.0, 'USD'))
        self.assertEqual((first['GB']['replacement']['amount'], first['GB']['currency']), (8.0, 'GBP'))
        self.assertFalse(first['GB']['replacement']['converted'])

    def test_appraisal_book_shows_no_figure(self):
        """A signed copy is withheld in every market."""
        books = [{'id': 'b', 'title': 'Dune, signed by the author', 'catalogue': None, 'observations': []}]
        with httpx.Client(transport=httpx.MockTransport(google)) as http:
            row = compare_markets(books, ['US', 'GB'], client=http)['books'][0]
        self.assertTrue(row['prices']['GB']['needs_appraisal'])
        self.assertIsNone(row['prices']['GB']['replacement'])


if __name__ == '__main__':
    unittest.main()
