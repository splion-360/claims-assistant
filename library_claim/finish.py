"""Finish a sweep's background work: catalogue lookups, room measurement and pricing.

The live loop starts this work while the policyholder walks, but a lookup can fail,
the server can restart, or the camera can stop before every book is priced. This
step completes whatever is missing and is safe to run any number of times: work
already saved is reused, and work that failed for a transient reason is retried.
It runs when the camera stops and again before every packet download, so the
packet never depends on the live loop having finished.
"""
import logging
from pathlib import Path
import time

import httpx

from library_claim.domain.models import Sweep
from library_claim.identification.catalogue import resolve_book
from library_claim.identification.storage import inventory
from library_claim.items.pricing import value_items
from library_claim.items.storage import inventory as item_inventory
from library_claim.measurement.room_storage import measure_room, stored_room
from library_claim.pricing.storage import value_sweep
from library_claim.storage.repository import Repository
from library_claim import telemetry

logger = logging.getLogger(__name__)
# Review reason resolve_book writes when a catalogue answered but nothing matched.
NO_MATCH = 'No confident catalogue match'


def _needs_catalogue(book: dict) -> bool:
    """Say whether a book still lacks a catalogue outcome.

    Args:
        book: Inventory entry.

    Returns:
        True for a titled, kept book with neither a match nor a recorded "no match".
    """
    if not book.get('title') or book.get('excluded') or book.get('catalogue'):
        return False
    return NO_MATCH not in (book.get('review_reasons') or [])


def _books(database: Path, sweep_id: str) -> list[dict]:
    """Read the sweep's kept books.

    Args:
        database: Capture SQLite path.
        sweep_id: Owning sweep.

    Returns:
        Inventory entries not excluded by the policyholder.
    """
    with Repository(database) as repo:
        return [book for book in inventory(repo, sweep_id) if not book.get('excluded')]


def finish_sweep(database: Path, data_dir: Path, sweep_id: str, country: str | None = None, *,
                 http: httpx.Client | None = None, drex: httpx.Client | None = None,
                 resolve=resolve_book, measure=measure_room) -> dict:
    """Complete every unfinished background step for one sweep.

    Each step is isolated: one that raises is recorded and the others still run,
    so a failing source costs its own lines, never the packet.

    Args:
        database: Capture SQLite path.
        data_dir: Capture data directory holding depth sidecars.
        sweep_id: Sweep to finish.
        country: Market to price in; the sweep's own when omitted.
        http: Optional HTTP client for catalogue and price sources, used by tests.
        drex: Optional Drex client, used by tests.
        resolve: Catalogue step, injectable for tests.
        measure: Room step, injectable for tests.

    Returns:
        Counts of what ran, the errors met, and the elapsed seconds.
    """
    started = time.monotonic()
    summary = {'resolved': 0, 'priced': 0, 'items_priced': 0, 'room': False, 'errors': []}
    books = _books(database, sweep_id)
    for book in [book for book in books if _needs_catalogue(book)]:
        try:
            resolve(database, sweep_id, book, http=http, drex=drex)
            summary['resolved'] += 1
        except Exception as error:  # noqa: BLE001 - one book's lookup must not stop the rest
            summary['errors'].append(f'catalogue {book["id"]}: {type(error).__name__}')
    try:
        with Repository(database) as repo:
            has_room = stored_room(repo, sweep_id) is not None
        if not has_room:
            summary['room'] = measure(database, data_dir, sweep_id, refresh=False) is not None
    except Exception as error:  # noqa: BLE001 - a room failure must not stop pricing
        summary['errors'].append(f'room: {type(error).__name__}')
    try:
        with Repository(database) as repo:
            chosen = country or repo.get(Sweep, sweep_id).country
        # Re-read: catalogue matches above may have added ISBNs that sharpen the price lookup.
        valued = value_sweep(database, sweep_id, _books(database, sweep_id), chosen, client=http)
        summary['priced'] = sum(1 for valuation in valued.values() if valuation.replacement)
    except Exception as error:  # noqa: BLE001 - the packet still builds from what was saved
        summary['errors'].append(f'pricing: {type(error).__name__}')
        chosen = country
    try:
        with Repository(database) as repo:
            objects = [entry for entry in item_inventory(repo, sweep_id) if not entry.get('excluded')]
        item_values = value_items(database, sweep_id, objects, chosen, client=http, drex=drex)
        summary['items_priced'] = sum(1 for valuation in item_values.values() if valuation.replacement)
    except Exception as error:  # noqa: BLE001 - the packet still builds from what was saved
        summary['errors'].append(f'item pricing: {type(error).__name__}')
    summary['elapsed_s'] = round(time.monotonic() - started, 2)
    telemetry.emit('finish', 'done', f"{summary['resolved']} looked up, {summary['priced']} priced",
                   ms=round(summary['elapsed_s'] * 1000))
    for problem in summary['errors']:
        logger.warning('finish sweep=%s %s', sweep_id, problem)
    return summary
