"""Fetch book prices and exchange rates from sources an adjuster can retrieve."""
from datetime import datetime, timezone
import logging
import statistics

import httpx

from library_claim.config import CONFIG
from library_claim.sources import GOOGLE_BOOKS, HEADERS, google_books_params
from library_claim import telemetry
from .errors import PriceUnavailable
from .fetch import request_json
from .nimble import (abebooks_search_url, amazon_search_url, fetch_page, parse_abebooks, parse_amazon,
                     same_title)
from .models import Price

logger = logging.getLogger(__name__)
GOOGLE_SOURCE = 'Google Books (ebook edition)'


def _client(client: httpx.Client | None) -> tuple[httpx.Client, bool]:
    """Return a client to use and whether the caller must close it.

    Args:
        client: Caller-supplied client, or None to build one.

    Returns:
        The client and a flag saying whether it is owned here.
    """
    if client is not None:
        return client, False
    timeout = httpx.Timeout(CONFIG.pricing.timeout_s, connect=CONFIG.pricing.connect_timeout_s)
    return httpx.Client(timeout=timeout, headers=HEADERS, follow_redirects=True), True


def google_books_price(title: str, author: str | None, isbn: str | None, market, *,
                       client: httpx.Client | None = None) -> Price:
    """Look up a new-copy price in one country's Google Books store.

    The country parameter is what makes the figure local; the same query in
    another market returns that market's own price and currency.

    Args:
        title: Title read from the spine.
        author: Author read from the spine, when known.
        isbn: ISBN from an accepted catalogue match, preferred over the text search.
        market: Market naming the country and its expected currency.
        client: Optional HTTP client, used by tests.

    Returns:
        A replacement cost carrying its source and retrieval time.

    Raises:
        PriceUnavailable: The source answered with no price for this market.
    """
    if isbn:
        query = f'isbn:{isbn}'
    elif title:
        query = f'intitle:"{title}"' + (f' inauthor:"{author}"' if author else '')
    else:
        raise PriceUnavailable('No title or ISBN to price')
    params = google_books_params(q=query, country=market.country, maxResults=5)
    http, owned = _client(client)
    try:
        payload = request_json(http, 'GET', GOOGLE_BOOKS, params=params, source='Google Books')
        telemetry.usage('google_books', calls=1)
    except PriceUnavailable as error:
        # Without a key the anonymous quota is shared and usually spent; say so.
        if 'HTTP 429' in str(error) and 'key' not in params:
            raise PriceUnavailable(f'{error}: quota spent, no API key configured',
                                   transient=error.transient) from error
        raise
    finally:
        if owned:
            http.close()
    if not isinstance(payload, dict):
        raise PriceUnavailable('Google Books returned unreadable data')
    for item in payload.get('items') or []:
        sale = item.get('saleInfo') if isinstance(item, dict) else None
        if not isinstance(sale, dict):
            continue
        listed = sale.get('retailPrice') or sale.get('listPrice') or {}
        # A record in an unexpected shape is skipped, never coerced into a figure.
        if not isinstance(listed, dict) or not isinstance(listed.get('currencyCode'), str):
            continue
        try:
            amount = float(listed.get('amount'))
        except (TypeError, ValueError):
            continue
        currency = listed['currencyCode']
        link = sale.get('buyLink') or (item.get('volumeInfo') or {}).get('infoLink') or ''
        # Google Books sells ebooks; the label says so, since a print copy may cost more.
        return Price(amount=amount, currency=currency, source=GOOGLE_SOURCE, url=link,
                     retrieved_at=datetime.now(timezone.utc), kind='replacement')
    raise PriceUnavailable(f'No {market.country} price listed by Google Books')


def amazon_book_price(title: str, author: str | None, isbn: str | None, market, *,
                      client: httpx.Client | None = None) -> Price:
    """Look up a new-copy price on the market's Amazon store.

    The first listing of the same work wins, in the store's own relevance order;
    sequels and companions are skipped by comparing main titles.

    Args:
        title: Title read from the spine.
        author: Author, when known.
        isbn: ISBN from an accepted catalogue match, preferred as the search.
        market: Market naming the store and currency.
        client: Optional HTTP client, used by tests.

    Returns:
        A replacement cost linking to the product page.

    Raises:
        PriceUnavailable: The store has no listing of this work.
    """
    if not market.amazon_domain:
        raise PriceUnavailable(f'No Amazon store configured for {market.country}')
    if not title:
        raise PriceUnavailable('No title to price')
    keyword = isbn or ' '.join(part for part in (title, author) if part)
    page = fetch_page(amazon_search_url(market.amazon_domain, keyword), market.country, client=client)
    listings = parse_amazon(page, market.amazon_domain, market.currency)[:CONFIG.pricing.max_listings]
    match = next((listing for listing in listings if same_title(listing.title, title)), None)
    if match is None:
        raise PriceUnavailable(f'No listing of this title on {market.amazon_domain}')
    return Price(amount=match.amount, currency=match.currency, source=f'Amazon ({market.amazon_domain})',
                 url=match.url, retrieved_at=datetime.now(timezone.utc), kind='replacement')


def abebooks_used_price(title: str, author: str | None, isbn: str | None, market, *,
                        client: httpx.Client | None = None) -> Price:
    """Quote a used market value as the median of matching used offers on AbeBooks.

    Args:
        title: Title read from the spine.
        author: Author, when known.
        isbn: ISBN, preferred as the search when known.
        market: Market naming the store and currency.
        client: Optional HTTP client, used by tests.

    Returns:
        A used value linking to the search page it was computed from.

    Raises:
        PriceUnavailable: Too few used offers of this work were listed.
    """
    if not market.abebooks_domain:
        raise PriceUnavailable(f'No used-book store configured for {market.country}')
    if not title:
        raise PriceUnavailable('No title to price')
    url = abebooks_search_url(market.abebooks_domain, title, author, isbn)
    offers = [offer for offer in parse_abebooks(fetch_page(url, market.country, client=client),
                                                market.abebooks_domain, market.currency)
              if offer.condition.lower().startswith('used') and same_title(offer.title, title)]
    if len(offers) < CONFIG.pricing.used_min_listings:
        raise PriceUnavailable(f'Only {len(offers)} used offer(s) of this title on {market.abebooks_domain}; '
                               f'a used value needs {CONFIG.pricing.used_min_listings}')
    # The median resists one overpriced collector copy or one damaged bargain.
    amount = statistics.median(offer.amount for offer in offers)
    conditions = ', '.join(sorted({offer.condition for offer in offers}))
    return Price(amount=amount, currency=market.currency, source=f'AbeBooks ({market.abebooks_domain})', url=url,
                 retrieved_at=datetime.now(timezone.utc), kind='used',
                 condition_assumed=f'{CONFIG.pricing.condition_assumed}: median of {len(offers)} used offers '
                                   f'({conditions})')


PROVIDERS = {'google_books': google_books_price, 'amazon': amazon_book_price}
USED_PROVIDERS = {'abebooks': abebooks_used_price}


def exchange_rate(base: str, target: str, *, client: httpx.Client | None = None) -> tuple[float, str, str]:
    """Fetch a dated reference exchange rate.

    Args:
        base: Currency to convert from.
        target: Currency to convert into.
        client: Optional HTTP client, used by tests.

    Returns:
        The rate, the rate's publication date and the source URL.

    Raises:
        PriceUnavailable: No rate was published for that pair.
    """
    http, owned = _client(client)
    url = f'{CONFIG.pricing.fx_base_url}/latest'
    try:
        payload = request_json(http, 'GET', url, params={'base': base, 'symbols': target},
                               source='Exchange rate service')
    finally:
        if owned:
            http.close()
    rates = payload.get('rates') if isinstance(payload, dict) else None
    rate = rates.get(target) if isinstance(rates, dict) else None
    if not isinstance(rate, (int, float)):
        raise PriceUnavailable(f'No published rate from {base} to {target}')
    return float(rate), str(payload.get('date', '')), f'{url}?base={base}&symbols={target}'


def convert(price: Price, target_currency: str, *, client: httpx.Client | None = None) -> Price:
    """Restate a price in another currency and mark it as converted.

    Args:
        price: Sourced price in its original currency.
        target_currency: Currency the packet reports in.
        client: Optional HTTP client, used by tests.

    Returns:
        The converted price, carrying the rate and the rate's source.

    Raises:
        PriceUnavailable: No rate was published for that pair.
    """
    if price.currency == target_currency:
        return price
    rate, published, source = exchange_rate(price.currency, target_currency, client=client)
    return price.model_copy(update={
        'amount': price.amount * rate, 'currency': target_currency, 'converted': True,
        'converted_from': price.currency, 'fx_rate': rate,
        'fx_source': f'European Central Bank reference rate {published} via {source}'})
