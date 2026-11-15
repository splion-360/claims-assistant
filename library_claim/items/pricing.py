"""Value non-book contents from store listings, or say why a figure is missing.

An object with a legible brand and model is priced at the first listing Drex accepts
as that same product. Without one it gets a sourced range: the lowest and highest of
the listings Drex accepts as comparable replacements. Framed art and mirrors are never
valued automatically.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import logging
from pathlib import Path

import httpx

from library_claim.config import CONFIG, CONFIG_DIGEST
from library_claim.decisions.drex import DrexError
from library_claim.decisions.listing_match import listing_probabilities
from library_claim.domain.models import ProcessingJob
from library_claim.items.models import APPRAISAL_CATEGORIES
from library_claim.pricing.errors import PriceUnavailable
from library_claim.pricing.markets import Market, market
from library_claim.pricing.models import Price, Valuation
from library_claim.pricing.nimble import amazon_search_url, fetch_page, parse_amazon
from library_claim.pricing.providers import convert
from library_claim.storage.repository import Repository

logger = logging.getLogger(__name__)
APPRAISAL_REASON = 'Framed art or a mirror; an appraiser values this unless it is a print'
UNIDENTIFIED_REASON = 'Brand and model are not legible, so only a sourced range can be given'


def _keyword(item: dict) -> str:
    """Build the store search for an object.

    Args:
        item: Inventory entry.

    Returns:
        The brand and model when legible, else the material and description.
    """
    if item.get('brand_model'):
        return item['brand_model']
    return ' '.join(part for part in (item.get('material'), item.get('description')) if part)


def amazon_item_price(item: dict, chosen: Market, *, client: httpx.Client | None = None,
                      drex: httpx.Client | None = None) -> tuple[Price, Price]:
    """Price an object from the market's Amazon search results, checked by Drex.

    Args:
        item: Inventory entry.
        chosen: Market naming the store and currency.
        client: Optional HTTP client for the store page, used by tests.
        drex: Optional Drex client, used by tests.

    Returns:
        The low and high replacement cost (equal when one product was matched).

    Raises:
        PriceUnavailable: No store, no listings, no accepted listing, or no decision.
    """
    if not chosen.amazon_domain:
        raise PriceUnavailable(f'No Amazon store configured for {chosen.country}')
    keyword = _keyword(item)
    if not keyword:
        raise PriceUnavailable('Nothing legible to search for')
    url = amazon_search_url(chosen.amazon_domain, keyword)
    listings = parse_amazon(fetch_page(url, chosen.country, client=client), chosen.amazon_domain,
                            chosen.currency)[:CONFIG.pricing.max_listings]
    if not listings:
        raise PriceUnavailable(f'No priced listings for "{keyword}" on {chosen.amazon_domain}')
    try:
        probabilities = listing_probabilities(item, [listing.title for listing in listings], client=drex)
    except DrexError as error:
        # Without a decision no listing is trusted; the lookup is repeated later.
        raise PriceUnavailable('Listing match could not be decided', transient=True) from error
    accepted = [listing for listing, value in zip(listings, probabilities)
                if value is not None and value >= CONFIG.drex.listing_match_min]
    if not accepted:
        raise PriceUnavailable(f'No listing for "{keyword}" on {chosen.amazon_domain} matched the object')
    now = datetime.now(timezone.utc)
    source = f'Amazon ({chosen.amazon_domain})'
    if item.get('brand_model'):
        best = accepted[0]
        price = Price(amount=best.amount, currency=best.currency, source=source, url=best.url, retrieved_at=now)
        return price, price
    low, high = min(accepted, key=lambda listing: listing.amount), max(accepted, key=lambda listing: listing.amount)
    # The range cites the search page and the two listings that bound it.
    note = f'{source}: range over {len(accepted)} comparable listings'
    return (Price(amount=low.amount, currency=low.currency, source=note, url=low.url, retrieved_at=now),
            Price(amount=high.amount, currency=high.currency, source=note, url=high.url, retrieved_at=now))


# Retail price lookups by provider name, returning (low, high) or raising PriceUnavailable.
PROVIDERS = {'amazon': amazon_item_price}


def _lookup(item: dict, chosen: Market, client, drex) -> tuple[tuple[Price, Price] | None, list[str], bool]:
    """Try each configured item provider in one market.

    Args:
        item: Inventory entry.
        chosen: Market to look in.
        client: HTTP client or None.
        drex: Drex client or None.

    Returns:
        The (low, high) figures or None, failure reasons, and whether a failure was transient.
    """
    reasons, transient = [], False
    for name in CONFIG.pricing.item_providers:
        provider = PROVIDERS.get(name)
        if provider is None:
            reasons.append(f'Unknown retail price provider {name}')
            continue
        try:
            return provider(item, chosen, client=client, drex=drex), reasons, transient
        except PriceUnavailable as error:
            reasons.append(str(error))
            transient = transient or error.transient
        except Exception as error:  # noqa: BLE001 - a provider failure is a missing price, not a crash
            reasons.append(f'{name} returned no price ({type(error).__name__})')
    return None, reasons, transient


def price_item(item: dict, country: str | None = None, *, client: httpx.Client | None = None,
               drex: httpx.Client | None = None) -> Valuation:
    """Value one non-book object in the policyholder's market.

    Args:
        item: Inventory entry from `library_claim.items.storage.inventory`.
        country: Market to price in; the configured default when omitted.
        client: Optional HTTP client, used by tests.
        drex: Optional Drex client, used by tests.

    Returns:
        The valuation: `replacement` is the low figure and `replacement_high` the high one.
    """
    code = (country or CONFIG.pricing.default_country)
    try:
        chosen = market(code)
    except KeyError as error:
        return Valuation(book_id=item['id'], country=str(code).upper(), currency='', reasons=(str(error),))
    base = {'book_id': item['id'], 'country': chosen.country, 'currency': chosen.currency}
    if item.get('category') in APPRAISAL_CATEGORIES and not item.get('stated_print'):
        return Valuation(**base, needs_appraisal=True, reasons=(APPRAISAL_REASON,))
    if not CONFIG.pricing.item_providers:
        return Valuation(**base, reasons=('No retail price provider is configured for non-book contents',))
    found, reasons, transient = _lookup(item, chosen, client, drex)
    if found is None:
        # No local listing: look in the fallback markets and label what they give as converted.
        for fallback in CONFIG.pricing.convert_from:
            if fallback.upper() == chosen.country:
                continue
            try:
                other = market(fallback)
            except KeyError:
                continue
            figures, more, flaky = _lookup(item, other, client, drex)
            reasons += more
            transient = transient or flaky
            if figures is not None:
                try:
                    found = tuple(convert(figure, chosen.currency, client=client) for figure in figures)
                    break
                except PriceUnavailable as error:
                    reasons.append(str(error))
                    transient = transient or error.transient
    if not item.get('brand_model'):
        reasons.append(UNIDENTIFIED_REASON)
    if found is None:
        return Valuation(**base, reasons=tuple(reasons), retry_later=transient)
    low, high = found
    if high.amount > CONFIG.pricing.appraisal_threshold * CONFIG.pricing.item_threshold_factor:
        return Valuation(**base, needs_appraisal=True, replacement=low, replacement_high=high,
                         reasons=(f'Above the automatic pricing threshold for contents '
                                  f'({chosen.currency} {high.amount:.2f}); withheld for human appraisal',))
    return Valuation(**base, replacement=low, replacement_high=high, reasons=tuple(reasons))


def _inputs(item: dict, country: str) -> dict:
    """Capture what an item valuation depended on, so a changed reading is re-priced.

    Args:
        item: Inventory entry.
        country: Market code.

    Returns:
        The identifying inputs.
    """
    return {key: item.get(key) or '' for key in ('category', 'description', 'material', 'brand_model')} | \
        {'stated_print': bool(item.get('stated_print')), 'country': country}


def value_item(database: Path, sweep_id: str, item: dict, country: str | None = None, *,
               client: httpx.Client | None = None, drex: httpx.Client | None = None) -> Valuation | None:
    """Price an object once and save the figure, reusing it until its reading changes.

    Args:
        database: Capture SQLite path.
        sweep_id: Owning sweep.
        item: Inventory entry.
        country: Market to price in; the configured default when omitted.
        client: Optional HTTP client, used by tests.
        drex: Optional Drex client, used by tests.

    Returns:
        The valuation, or None when the object has no frame to attribute it to.
    """
    frame_ids = item.get('frame_ids') or []
    if not frame_ids:
        return None
    code = (country or CONFIG.pricing.default_country).upper()
    inputs, job_id = _inputs(item, code), f'value_item_{item["id"]}'
    with Repository(database) as repo:
        try:
            existing = repo.get(ProcessingJob, job_id)
        except KeyError:
            existing = None
    if (existing is not None and existing.status == 'succeeded' and existing.result
            and existing.result.get('inputs') == inputs):
        return Valuation(**existing.result['valuation'])
    valuation = price_item(item, code, client=client, drex=drex)
    # A timed-out lookup is saved as failed so the next run tries again.
    record = ProcessingJob(id=job_id, sweep_id=sweep_id, frame_id=frame_ids[0], task='value_item',
                           status='failed' if valuation.retry_later else 'succeeded',
                           created_at=datetime.now(timezone.utc),
                           model=valuation.replacement.source if valuation.replacement else None,
                           configuration_version=CONFIG_DIGEST,
                           result={'inputs': inputs, 'valuation': valuation.model_dump(mode='json')})
    with Repository(database) as repo:
        if existing is None:
            repo.add(record)
        else:
            repo.update(record)
    return valuation


def value_items(database: Path, sweep_id: str, items: list[dict], country: str | None = None, *,
                client: httpx.Client | None = None, drex: httpx.Client | None = None) -> dict[str, Valuation]:
    """Value every kept object in a sweep, side by side, keeping saved figures.

    Args:
        database: Capture SQLite path.
        sweep_id: Owning sweep.
        items: Inventory entries.
        country: Market to price in.
        client: Optional HTTP client, used by tests.
        drex: Optional Drex client, used by tests.

    Returns:
        Valuations indexed by item ID.
    """
    def one(item: dict) -> tuple[str, Valuation | None]:
        try:
            return item['id'], value_item(database, sweep_id, item, country, client=client, drex=drex)
        except Exception as error:  # noqa: BLE001 - one object's failure must not stop the rest
            logger.warning('item valuation failed for %s: %s', item.get('id'), type(error).__name__)
            return item['id'], None

    with ThreadPoolExecutor(max_workers=CONFIG.pricing.workers) as pool:
        return {item_id: valuation for item_id, valuation in pool.map(one, items) if valuation is not None}
