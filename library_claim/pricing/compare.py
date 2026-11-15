"""Price the same books in a second market to show locale is a setting, not hardcoded."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from library_claim.config import CONFIG
from .cache import PriceCache
from .service import price_book


def _figure(price) -> dict | None:
    """Render one price for the comparison table.

    Args:
        price: A Price or None.

    Returns:
        Amount, currency, source, URL, retrieval time and conversion flag, or None.
    """
    if price is None:
        return None
    return {'amount': round(price.amount, 2), 'currency': price.currency, 'source': price.source,
            'url': price.url, 'retrieved_at': price.retrieved_at.isoformat(), 'converted': price.converted}


def compare_markets(books: list[dict], countries: list[str], *, cache_path: Path | None = None,
                    client: httpx.Client | None = None, limit: int = 10) -> dict:
    """Price up to `limit` identified books in each listed market.

    The same books, in the same order, are looked up independently in every market,
    so any difference comes from that market's own sources and currency.

    Args:
        books: Inventory entries; only identified, kept books are used.
        countries: Market codes, the policyholder's first.
        cache_path: Today's price cache, shared with the sweep's own pricing.
        client: Optional HTTP client, used by tests.
        limit: Number of books compared.

    Returns:
        The markets compared and, per book, each market's replacement and used figures.
    """
    chosen = [book for book in books if book.get('title') and not book.get('excluded')][:limit]
    cache = PriceCache(cache_path) if cache_path else None

    def row(book: dict) -> dict:
        prices = {}
        for country in countries:
            valuation = price_book(book, country, client=client, cache=cache)
            # A book withheld for appraisal shows no figure in any market.
            withheld = valuation.needs_appraisal
            prices[country.upper()] = {
                'currency': valuation.currency, 'needs_appraisal': withheld,
                'replacement': None if withheld else _figure(valuation.replacement),
                'used': None if withheld else _figure(valuation.used)}
        return {'id': book['id'], 'title': book['title'], 'author': book.get('author') or '', 'prices': prices}

    with ThreadPoolExecutor(max_workers=CONFIG.pricing.workers) as pool:
        rows = list(pool.map(row, chosen))
    return {'countries': [country.upper() for country in countries], 'books': rows}
