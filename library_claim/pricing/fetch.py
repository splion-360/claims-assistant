"""Call outside price sources with a bounded retry and a per-source circuit breaker.

A source that is down must cost the sweep seconds, not minutes: each call is retried
at most once, a Retry-After is capped, and a source that keeps failing is skipped
for a cooldown so the remaining lines are priced from other sources or left empty.
"""
import threading
import time

import httpx

from library_claim.config import CONFIG
from .errors import PriceUnavailable

# Statuses worth one more try: rate limiting and server-side failures.
TRANSIENT_STATUSES = {408, 425, 429, 500, 502, 503, 504, 529}

_lock = threading.Lock()
_failures: dict[str, int] = {}
_open_until: dict[str, float] = {}


def reset_breakers() -> None:
    """Close every breaker, e.g. between tests."""
    with _lock:
        _failures.clear()
        _open_until.clear()


def _record(source: str, ok: bool) -> None:
    """Count a transient failure, or clear the count after a success.

    Args:
        source: Source name.
        ok: Whether the call succeeded or failed definitely (both close the breaker).
    """
    with _lock:
        if ok:
            _failures.pop(source, None)
            return
        _failures[source] = _failures.get(source, 0) + 1
        if _failures[source] >= CONFIG.pricing.breaker_failures:
            _open_until[source] = time.monotonic() + CONFIG.pricing.breaker_cooldown_s


def _wait_s(response: httpx.Response | None) -> float:
    """Pick the pause before the retry, honouring a short Retry-After.

    Args:
        response: The failed response, or None after a network error.

    Returns:
        Seconds to wait, never above the configured cap.
    """
    wait = CONFIG.pricing.retry_backoff_s
    if response is not None:
        try:
            wait = float(response.headers.get('retry-after', wait))
        except ValueError:
            pass
    return max(0.0, min(wait, CONFIG.pricing.retry_max_wait_s))


def request_json(http: httpx.Client, method: str, url: str, *, source: str, **kwargs) -> dict | list:
    """Send one request and return its JSON body.

    Args:
        http: Client to send with.
        method: HTTP method.
        url: Request URL.
        source: Source name shown in messages and used for the breaker.
        **kwargs: Passed to `httpx.Client.request` (params, json, headers).

    Returns:
        The decoded JSON body.

    Raises:
        PriceUnavailable: The source failed; `transient` says whether a later try may work.
    """
    with _lock:
        skipped = _open_until.get(source, 0) > time.monotonic()
    if skipped:
        raise PriceUnavailable(f'{source} skipped: it failed repeatedly in the last '
                               f'{CONFIG.pricing.breaker_cooldown_s:.0f} s', transient=True)
    attempts = 1 + CONFIG.pricing.retries
    for attempt in range(attempts):
        response = None
        try:
            response = http.request(method, url, **kwargs)
        except httpx.TransportError as error:
            problem = f'{source} unreachable ({type(error).__name__})'
        else:
            if response.status_code in TRANSIENT_STATUSES:
                problem = f'{source} refused the request (HTTP {response.status_code})'
            elif response.is_error:
                _record(source, ok=True)
                raise PriceUnavailable(f'{source} rejected the request (HTTP {response.status_code})')
            else:
                _record(source, ok=True)
                try:
                    return response.json()
                except ValueError as error:
                    # A body that is not JSON will not improve on a retry.
                    raise PriceUnavailable(f'{source} returned unreadable data') from error
        if attempt + 1 < attempts:
            time.sleep(_wait_s(response))
    _record(source, ok=False)
    raise PriceUnavailable(problem, transient=True)
