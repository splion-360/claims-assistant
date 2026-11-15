"""Read retail and used-market listings from public store pages through the Nimble web API.

Nimble fetches a page as a visitor in the given country would see it; this module
parses the listings out of that page. Only what the page states is kept: a listing
whose price, currency or link cannot be read is dropped, never filled in.
"""
from dataclasses import dataclass
from html import unescape
import os
import re
from urllib.parse import quote_plus

import httpx

from library_claim import telemetry
from library_claim.config import CONFIG
from library_claim.identification.matching import normalize
from library_claim.sources import HEADERS
from .errors import PriceUnavailable
from .fetch import request_json

NIMBLE = 'https://sdk.nimbleway.com/v2'
# Currency labels as the stores print them. "$" alone is only trusted on a store whose
# market currency is USD; elsewhere it is ambiguous and the listing is dropped.
CURRENCY_LABELS = {'US$': 'USD', 'CA$': 'CAD', 'C$': 'CAD', 'AU$': 'AUD', 'A$': 'AUD', 'NZ$': 'NZD',
                   '£': 'GBP', '€': 'EUR', '₹': 'INR', '¥': 'JPY', '$': 'USD'}
# Longest label first, so "US$" is not read as "$".
_LABELS = sorted(CURRENCY_LABELS, key=len, reverse=True)


@dataclass(frozen=True)
class Listing:
    """One product or book offer exactly as a store page shows it."""
    title: str
    amount: float
    currency: str
    url: str
    condition: str = ''
    sponsored: bool = False


def parse_money(text: str) -> tuple[float, str] | None:
    """Read an amount and its currency from a printed price such as "£1,299.00" or "US$ 5.12".

    Args:
        text: Price text from the page.

    Returns:
        (amount, ISO currency) or None when the text is not a single readable price.
    """
    text = unescape(text or '').replace('\xa0', ' ').strip()
    for label in _LABELS:
        if text.startswith(label):
            number = text[len(label):].strip()
            # Prices are written 1,299.00 on every store this module reads.
            if re.fullmatch(r'\d{1,3}(,\d{3})*(\.\d{1,2})?|\d+(\.\d{1,2})?', number):
                return float(number.replace(',', '')), CURRENCY_LABELS[label]
            return None
    return None


def main_title(title: str) -> str:
    """Strip a subtitle, series note or format note from a listing title.

    Args:
        title: Title as listed, e.g. "Dune (Dune Chronicles, Book 1)" or "Dune: Deluxe Edition".

    Returns:
        The normalized main title, e.g. "dune".
    """
    return normalize(re.split(r'\s*[:(\[]|\s+-\s+|\s+–\s+', title or '', maxsplit=1)[0])


def same_title(listed: str, title: str) -> bool:
    """Say whether a listing is for the same work, not a sequel or companion.

    "Dune Messiah" and "Dune" share a first word but are different books, so the
    main titles must be equal, not merely overlap.

    Args:
        listed: Listing title.
        title: Title read from the spine (or its catalogue match).

    Returns:
        True when the main titles are the same words.
    """
    return bool(main_title(title)) and main_title(listed) == main_title(title)


def _api_key() -> str:
    """Return the Nimble key or explain that none is configured.

    Returns:
        The API key.

    Raises:
        PriceUnavailable: NIMBLE_API_KEY is not set.
    """
    key = os.environ.get('NIMBLE_API_KEY', '').strip()
    if not key:
        raise PriceUnavailable('Web listing source not configured (NIMBLE_API_KEY is empty)')
    return key


def fetch_page(url: str, country: str, *, client: httpx.Client | None = None) -> str:
    """Fetch a store page's HTML as a visitor in a country sees it.

    Args:
        url: Page to fetch.
        country: ISO country the visit appears to come from.
        client: Optional HTTP client, used by tests.

    Returns:
        The page HTML.

    Raises:
        PriceUnavailable: The page could not be fetched.
    """
    key = _api_key()
    owned = client is None
    http = client or httpx.Client(timeout=httpx.Timeout(CONFIG.pricing.nimble_timeout_s,
                                                        connect=CONFIG.pricing.connect_timeout_s),
                                  headers=HEADERS)
    try:
        payload = request_json(http, 'POST', f'{NIMBLE}/extract', source='Web listing source',
                               headers={'Authorization': f'Bearer {key}'},
                               json={'url': url, 'formats': ['html'], 'country': country})
    finally:
        if owned:
            http.close()
    # Only successful page loads are billed by the source.
    telemetry.usage('nimble', calls=1)
    html = ((payload.get('data') or {}).get('html') if isinstance(payload, dict) else None) or ''
    if not isinstance(payload, dict) or payload.get('status') != 'success' or not html:
        # The service answered but the store page did not load (blocked, captcha, empty).
        raise PriceUnavailable('Web listing source could not load the store page', transient=True)
    return html


def _text(fragment: str) -> str:
    """Strip tags and entities from an HTML fragment.

    Args:
        fragment: HTML.

    Returns:
        Plain text with collapsed spaces.
    """
    return ' '.join(unescape(re.sub(r'<[^>]+>', ' ', fragment)).split())


def parse_amazon(html: str, domain: str, currency: str) -> list[Listing]:
    """Read the product results from an Amazon search page.

    Args:
        html: Search page HTML.
        domain: Store host, e.g. www.amazon.co.uk, used to build product links.
        currency: The market's currency; listings priced otherwise are dropped.

    Returns:
        Listings in page order, one per product.
    """
    marker = 'data-component-type="s-search-result"'
    starts = [match.start() for match in re.finditer(re.escape(marker), html)]
    listings, seen = [], set()
    for index, start in enumerate(starts):
        tag = html[html.rfind('<div', 0, start):html.find('>', start)]
        block = html[start:starts[index + 1] if index + 1 < len(starts) else len(html)]
        asin = re.search(r'data-asin="([A-Z0-9]{10})"', tag)
        heading = re.search(r'<h2[^>]*>(.*?)</h2>', block, re.S)
        price = re.search(r'class="a-offscreen">([^<]+)<', block)
        if not (asin and heading and price) or asin.group(1) in seen:
            continue
        money = parse_money(price.group(1))
        if money is None or money[1] != currency:
            continue
        seen.add(asin.group(1))
        listings.append(Listing(title=_text(heading.group(1)), amount=money[0], currency=money[1],
                                url=f'https://{domain}/dp/{asin.group(1)}', sponsored='Sponsored' in block))
    return listings


def parse_abebooks(html: str, domain: str, currency: str) -> list[Listing]:
    """Read the offers from an AbeBooks search results page.

    Args:
        html: Results page HTML.
        domain: Store host, e.g. www.abebooks.co.uk, used to build offer links.
        currency: The market's currency; offers priced otherwise are dropped.

    Returns:
        Offers in page order, each with its stated condition.
    """
    starts = [match.start() for match in re.finditer(r'data-test-id="listing-item-\d+"', html)]
    listings = []
    for index, start in enumerate(starts):
        block = html[start:starts[index + 1] if index + 1 < len(starts) else len(html)]
        title = re.search(r'data-test-id="listing-title"[^>]*>(.*?)</', block, re.S)
        link = re.search(r'data-test-id="listing-title-link"[^>]*href="([^"]+)"', block) or \
            re.search(r'href="([^"]+)"[^>]*data-test-id="listing-title-link"', block)
        price = re.search(r'data-test-id="listing-price"[^>]*>([^<]+)<', block)
        condition = re.search(r'data-test-id="listing-condition"[^>]*>([^<]+)<', block)
        if not (title and link and price):
            continue
        money = parse_money(price.group(1))
        if money is None or money[1] != currency:
            continue
        href = unescape(link.group(1))
        listings.append(Listing(title=_text(title.group(1)), amount=money[0], currency=money[1],
                                url=href if href.startswith('http') else f'https://{domain}{href}',
                                condition=_text(condition.group(1)) if condition else ''))
    return listings


def amazon_search_url(domain: str, keyword: str) -> str:
    """Build an Amazon search URL.

    Args:
        domain: Store host.
        keyword: Search words.

    Returns:
        The search page URL.
    """
    return f'https://{domain}/s?k={quote_plus(keyword)}'


def abebooks_search_url(domain: str, title: str, author: str | None, isbn: str | None) -> str:
    """Build an AbeBooks search URL; conditions are filtered from the parsed offers.

    Args:
        domain: Store host.
        title: Title to search.
        author: Author, when known.
        isbn: ISBN, preferred when known.

    Returns:
        The results page URL, in the store's default relevance order.
    """
    if isbn:
        query = f'isbn={quote_plus(isbn)}'
    else:
        query = f'tn={quote_plus(title)}' + (f'&an={quote_plus(author)}' if author else '')
    return f'https://{domain}/servlet/SearchResults?{query}'
