"""Fetch book prices and exchange rates from sources an adjuster can retrieve."""
from datetime import datetime, timezone
import logging

import httpx

from library_claim.config import CONFIG
from library_claim.sources import GOOGLE_BOOKS, HEADERS, google_books_params
from .models import Price

logger = logging.getLogger(__name__)


class PriceUnavailable(RuntimeError):
    """Raised when a source returns no usable price. The message is safe to show."""


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
        response = http.get(GOOGLE_BOOKS, params=params)
        if response.status_code == 429:
            raise PriceUnavailable('Price source refused the request (quota); no API key configured')
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as error:
        raise PriceUnavailable(f'Price source unreachable ({type(error).__name__})') from error
    finally:
        if owned:
            http.close()
    for item in payload.get('items') or []:
        sale = item.get('saleInfo') or {}
        listed = sale.get('retailPrice') or sale.get('listPrice') or {}
        amount, currency = listed.get('amount'), listed.get('currencyCode')
        if amount is None or currency is None:
            continue
        link = sale.get('buyLink') or (item.get('volumeInfo') or {}).get('infoLink') or ''
        return Price(amount=float(amount), currency=currency, source='Google Books', url=link,
                     retrieved_at=datetime.now(timezone.utc), kind='replacement')
    raise PriceUnavailable(f'No {market.country} price listed by Google Books')


PROVIDERS = {'google_books': google_books_price}


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
        response = http.get(url, params={'base': base, 'symbols': target})
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPError as error:
        raise PriceUnavailable(f'Exchange rate unavailable ({type(error).__name__})') from error
    finally:
        if owned:
            http.close()
    rate = (payload.get('rates') or {}).get(target)
    if rate is None:
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
