"""Describe the markets a sweep can be priced in and how each one is reached."""
from pydantic import BaseModel, ConfigDict

from library_claim.config import CONFIG


class Market(BaseModel):
    """One country's pricing settings."""
    model_config = ConfigDict(extra='forbid', frozen=True)
    country: str
    currency: str
    # Providers are tried in order; the first with a sourced price wins.
    providers: tuple[str, ...]


def market(country: str) -> Market:
    """Look up the pricing settings for a country.

    Args:
        country: ISO 3166-1 alpha-2 country code, case-insensitive.

    Returns:
        The configured market.

    Raises:
        KeyError: No market is configured for that country.
    """
    code = (country or '').strip().upper()
    settings = CONFIG.pricing.markets.get(code)
    if settings is None:
        raise KeyError(f'No pricing market configured for {code or "an unset country"}')
    return Market(country=code, currency=settings.currency, providers=tuple(settings.providers))


def known_countries() -> list[str]:
    """List the countries a sweep can be priced in.

    Returns:
        Configured country codes in alphabetical order.
    """
    return sorted(CONFIG.pricing.markets)
