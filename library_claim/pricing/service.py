"""Decide what a book may be priced at, and withhold anything needing an appraiser."""
import logging

import httpx

from library_claim.config import CONFIG
from .markets import Market, market
from .cache import PriceCache
from .models import Price, Valuation
from .errors import PriceUnavailable
from .providers import PROVIDERS, USED_PROVIDERS, convert

logger = logging.getLogger(__name__)
APPRAISAL_REASON = 'Looks rare, signed or antiquarian; withheld from automatic pricing'
THRESHOLD_REASON = 'Above the automatic pricing threshold; withheld for human appraisal'


def appraisal_signals(book: dict) -> list[str]:
    """Find wording that marks a book as one an appraiser must value.

    Evidence is the text actually read or stated, never an inference about
    value drawn from the title alone.

    Args:
        book: Inventory entry carrying readings and observation text.

    Returns:
        The matched keywords, lowercased, in configuration order.
    """
    haystack = ' '.join(str(book.get(field) or '') for field in ('title', 'author', 'edition', 'notes')).lower()
    for observation in book.get('observations') or []:
        haystack += ' ' + str(observation.get('title') or '').lower()
    return [word for word in CONFIG.pricing.appraisal_keywords if word.lower() in haystack]


def _lookup(book: dict, chosen: Market, client: httpx.Client | None,
            cache: PriceCache | None = None, *, used: bool = False) -> tuple[Price | None, list[str], bool]:
    """Try each configured provider for the market until one returns a price.

    Args:
        book: Inventory entry being priced.
        chosen: Market to price in.
        client: Optional HTTP client, used by tests.
        cache: Today's remembered answers, or None to always ask the source.
        used: Look up a used value instead of a replacement cost.

    Returns:
        The first sourced price or None, the reasons lookups failed, and whether
        any failure was transient (worth retrying later).
    """
    reasons, transient = [], False
    catalogue = book.get('catalogue') or {}
    names, table = (chosen.used_providers, USED_PROVIDERS) if used else (chosen.providers, PROVIDERS)
    for name in names:
        provider = table.get(name)
        if provider is None:
            reasons.append(f'Unknown price provider {name}')
            continue
        key = (f'used:{name}' if used else name, book.get('title'), book.get('author'), catalogue.get('isbn'), chosen.country)
        cached = cache.get(key) if cache is not None else PriceCache.MISS
        if isinstance(cached, Price):
            return cached, reasons, transient
        if cached is None:
            reasons.append(cache.reason(key) or f'{name} had no price earlier today')
            continue
        try:
            price = provider(book.get('title') or '', book.get('author'), catalogue.get('isbn'),
                             chosen, client=client)
        except PriceUnavailable as error:
            reasons.append(str(error))
            transient = transient or error.transient
            # Only a definite "no listing" is remembered; a timeout is asked again next time.
            if cache is not None and not error.transient:
                cache.put(key, None, reason=str(error))
            continue
        if cache is not None:
            cache.put(key, price)
        return price, reasons, transient
    return None, reasons, transient


def price_book(book: dict, country: str | None = None, *, client: httpx.Client | None = None,
               cache: PriceCache | None = None) -> Valuation:
    """Value one identified book in its policyholder's market.

    A book with no readable title is never priced, and one that looks rare or
    signed is withheld rather than valued automatically. A price found only in
    another market is converted and labelled as converted.

    Args:
        book: Inventory entry from `library_claim.identification.storage.inventory`.
        country: Country to price in; the configured default when omitted.
        client: Optional HTTP client, used by tests.
        cache: Today's remembered price answers, or None to always ask the sources.

    Returns:
        The valuation, carrying any figure found and the reasons for what is missing.
    """
    code = (country or CONFIG.pricing.default_country)
    try:
        chosen = market(code)
    except KeyError as error:
        return Valuation(book_id=book['id'], country=str(code).upper(), currency='', reasons=(str(error),))
    base = {'book_id': book['id'], 'country': chosen.country, 'currency': chosen.currency}
    if not book.get('title'):
        return Valuation(**base, reasons=('Unidentified book; nothing to price',))
    # The appraisal gate runs before any lookup: these books are never auto-priced.
    signals = appraisal_signals(book)
    if signals:
        return Valuation(**base, needs_appraisal=True,
                         reasons=(f'{APPRAISAL_REASON} ({", ".join(signals)})',))
    price, reasons, transient = _find(book, chosen, client, cache, used=False)
    used, used_reasons, used_transient = _find(book, chosen, client, cache, used=True)
    reasons += used_reasons
    transient = transient or used_transient
    if price is None and used is None:
        return Valuation(**base, reasons=tuple(reasons) or ('No price source answered',), retry_later=transient)
    highest = max(figure.amount for figure in (price, used) if figure is not None)
    if highest > CONFIG.pricing.appraisal_threshold:
        # A high figure is reported but not trusted on its own; an appraiser confirms it.
        return Valuation(**base, needs_appraisal=True, replacement=price, used=used,
                         reasons=(f'{THRESHOLD_REASON} ({chosen.currency} {highest:.2f})',))
    if price is None:
        reasons.append('No replacement cost found; used value only')
    if used is None:
        reasons.append('No used value found; replacement cost only')
    # A figure missing because a source timed out may come back on the next run.
    return Valuation(**base, replacement=price, used=used, reasons=tuple(reasons), retry_later=transient)


def _find(book: dict, chosen: Market, client: httpx.Client | None, cache: PriceCache | None, *,
          used: bool) -> tuple[Price | None, list[str], bool]:
    """Look in the local market, then in the fallback markets, converting what they give.

    Args:
        book: Inventory entry being priced.
        chosen: The policyholder's market.
        client: Optional HTTP client, used by tests.
        cache: Today's remembered answers.
        used: Look up a used value instead of a replacement cost.

    Returns:
        The price (local, or converted and labelled), the failure reasons, and
        whether any failure was transient.
    """
    price, reasons, transient = _lookup(book, chosen, client, cache, used=used)
    if price is not None:
        return price, reasons, transient
    # No local price: try the configured fallback markets and label the result converted.
    for fallback in CONFIG.pricing.convert_from:
        if fallback.upper() == chosen.country:
            continue
        try:
            other = market(fallback)
        except KeyError:
            continue
        found, more, flaky = _lookup(book, other, client, cache, used=used)
        reasons += more
        transient = transient or flaky
        if found is not None:
            try:
                return convert(found, chosen.currency, client=client), reasons, transient
            except PriceUnavailable as error:
                reasons.append(str(error))
                transient = transient or error.transient
    return None, reasons, transient
