"""Value non-book contents from retrievable sources, or say why a figure is missing."""
import logging

import httpx

from library_claim.config import CONFIG
from library_claim.items.models import APPRAISAL_CATEGORIES
from library_claim.pricing.markets import market
from library_claim.pricing.models import Valuation

logger = logging.getLogger(__name__)
APPRAISAL_REASON = 'Framed art or a mirror; an appraiser values this unless it is a print'
NO_PROVIDER_REASON = ('No retail price provider is configured for non-book contents; '
                      'set one under pricing.item_providers to value these lines')
UNIDENTIFIED_REASON = 'Brand and model are not legible; a sourced range needs a retail provider'

# Retail price lookups by provider name. A provider returns a low and high amount
# with the source behind them, or raises PriceUnavailable.
PROVIDERS: dict[str, object] = {}


def price_item(item: dict, country: str | None = None, *, client: httpx.Client | None = None) -> Valuation:
    """Value one non-book object in the policyholder's market.

    Framed art and mirrors are never valued automatically: the brief leaves them
    to an appraiser unless the policyholder says the piece is a print. Everything
    else needs a configured retail provider; without one the line stays empty and
    says so rather than carrying a figure with nothing behind it.

    Args:
        item: Inventory entry from `library_claim.items.storage.inventory`.
        country: Market to price in; the configured default when omitted.
        client: Optional HTTP client, used by tests.

    Returns:
        The valuation, carrying any figure found and the reasons for what is missing.
    """
    code = (country or CONFIG.pricing.default_country)
    try:
        chosen = market(code)
    except KeyError as error:
        return Valuation(book_id=item['id'], country=str(code).upper(), currency='', reasons=(str(error),))
    base = {'book_id': item['id'], 'country': chosen.country, 'currency': chosen.currency}
    if item.get('category') in APPRAISAL_CATEGORIES and not item.get('stated_print'):
        return Valuation(**base, needs_appraisal=True, reasons=(APPRAISAL_REASON,))
    reasons = []
    for name in CONFIG.pricing.item_providers:
        provider = PROVIDERS.get(name)
        if provider is None:
            reasons.append(f'Unknown retail price provider {name}')
            continue
        try:
            price = provider(item, chosen, client=client)
        except Exception as error:  # noqa: BLE001 - a provider failure is a missing price, not a crash
            reasons.append(f'{name} returned no price ({type(error).__name__})')
            continue
        return Valuation(**base, replacement=price, reasons=tuple(reasons))
    if not CONFIG.pricing.item_providers:
        reasons.append(NO_PROVIDER_REASON)
    if not item.get('brand_model'):
        reasons.append(UNIDENTIFIED_REASON)
    return Valuation(**base, reasons=tuple(reasons))
