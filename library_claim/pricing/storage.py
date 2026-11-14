"""Persist one valuation per book so the panel and the packet quote the same figure."""
from datetime import datetime, timezone
import logging
from pathlib import Path

import httpx

from library_claim.config import CONFIG, CONFIG_DIGEST
from library_claim.domain.models import ProcessingJob
from library_claim.storage.repository import Repository
from .models import Valuation
from .service import price_book

logger = logging.getLogger(__name__)


def _inputs(book: dict, country: str) -> dict:
    """Capture what a valuation depended on, so a changed reading is re-priced.

    Args:
        book: Inventory entry being valued.
        country: Market the figure was fetched for.

    Returns:
        The identifying inputs of a price lookup.
    """
    catalogue = book.get('catalogue') or {}
    return {'title': book.get('title') or '', 'author': book.get('author') or '',
            'isbn': catalogue.get('isbn') or '', 'country': country}


def stored_valuations(repo: Repository, sweep_id: str) -> dict[str, dict]:
    """Read the saved valuation of every priced book in a sweep.

    Args:
        repo: Open repository on the caller's thread.
        sweep_id: Owning sweep.

    Returns:
        Valuation payloads indexed by book ID.
    """
    saved = {}
    for job in repo.list_for_sweep(ProcessingJob, sweep_id):
        if job.task == 'value' and job.status == 'succeeded' and job.result and job.book_id:
            saved[job.book_id] = job.result['valuation']
    return saved


def value_book(database: Path, sweep_id: str, book: dict, country: str | None = None, *,
               client: httpx.Client | None = None, refresh: bool = False) -> Valuation | None:
    """Price a book once and save the figure, reusing it until the reading changes.

    Pricing is kept off the capture path and out of packet downloads: the stored
    figure is what both report, so a book is not re-priced on every view and its
    retrieval time stays the moment the sweep actually looked it up.

    Args:
        database: Capture SQLite path.
        sweep_id: Owning sweep.
        book: Inventory entry to value.
        country: Market to price in; the sweep default when omitted.
        client: Optional HTTP client, used by tests.
        refresh: Price again even when a current figure is already saved.

    Returns:
        The valuation, or None when the book has no frame to attribute it to.
    """
    frame_ids = book.get('frame_ids') or []
    if not frame_ids:
        return None
    code = (country or CONFIG.pricing.default_country).upper()
    inputs, job_id = _inputs(book, code), f'value_{book["id"]}'
    with Repository(database) as repo:
        try:
            existing = repo.get(ProcessingJob, job_id)
        except KeyError:
            existing = None
    # A saved figure stands until the reading, the catalogue match or the market changes.
    if existing is not None and not refresh and existing.result and existing.result.get('inputs') == inputs:
        return Valuation(**existing.result['valuation'])
    valuation = price_book(book, code, client=client)
    record = ProcessingJob(id=job_id, sweep_id=sweep_id, frame_id=frame_ids[0], book_id=book['id'],
                           task='value', status='succeeded', created_at=datetime.now(timezone.utc),
                           model=valuation.replacement.source if valuation.replacement else None,
                           configuration_version=CONFIG_DIGEST,
                           result={'inputs': inputs, 'valuation': valuation.model_dump(mode='json')})
    with Repository(database) as repo:
        if existing is None:
            repo.add(record)
        else:
            repo.update(record)
    return valuation


def value_sweep(database: Path, sweep_id: str, books: list[dict], country: str | None = None, *,
                client: httpx.Client | None = None) -> dict[str, Valuation]:
    """Value every identified book in a sweep, keeping already-priced ones.

    Args:
        database: Capture SQLite path.
        sweep_id: Owning sweep.
        books: Inventory entries to value.
        country: Market to price in; the sweep default when omitted.
        client: Optional HTTP client, used by tests.

    Returns:
        Valuations indexed by book ID, omitting books that could not be valued.
    """
    valued = {}
    for book in books:
        try:
            valuation = value_book(database, sweep_id, book, country, client=client)
        except (httpx.HTTPError, ValueError, KeyError) as error:
            # One unreachable source must not stop the rest of the sweep being priced.
            logger.warning('valuation failed for %s: %s', book.get('id'), type(error).__name__)
            continue
        if valuation is not None:
            valued[book['id']] = valuation
    return valued
