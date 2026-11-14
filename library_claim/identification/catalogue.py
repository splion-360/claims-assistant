"""Resolve legible spine readings to a catalogue entry (Google Books, then Open Library), with Drex choosing."""
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path

import httpx

from library_claim import telemetry
from library_claim.config import CONFIG
from library_claim.decisions.drex import DrexError, choice, decide
from library_claim.domain.models import BookObservation, ReviewIssue
from library_claim.sources import GOOGLE_BOOKS, HEADERS, google_books_params, volume_isbn
from library_claim.storage.repository import Repository
from .matching import normalize

OPEN_LIBRARY = 'https://openlibrary.org'
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkCandidate:
    """One catalogue entry returned for a spine reading."""
    key: str
    title: str
    authors: list[str]
    first_publish_year: int | None
    source: str = 'googlebooks'
    url: str | None = None
    publisher: str | None = None
    isbn: str | None = None
    edition: str | None = None


def _year(published: str | None) -> int | None:
    """Read the publication year from a catalogue date of any precision.

    Args:
        published: A date such as '2016', '2016-09' or '2016-09-27'.

    Returns:
        The year, or None when the record carries no usable date.
    """
    try:
        return int(str(published or '')[:4])
    except ValueError:
        return None


def search_google_books(title: str, author: str | None, *,
                        client: httpx.Client | None = None) -> list[WorkCandidate]:
    """Search Google Books by title and, when read, author.

    Args:
        title: Title read from the spine.
        author: Author read from the spine, if any.
        client: Optional HTTP client, used by tests.

    Returns:
        Up to five candidate volumes.

    Raises:
        httpx.HTTPError: The request failed or returned an error status.
    """
    query = f'intitle:"{title}"' + (f' inauthor:"{author}"' if author else '')
    owned = client is None
    client = client or httpx.Client(timeout=15)
    try:
        response = client.get(GOOGLE_BOOKS, params=google_books_params(q=query, maxResults=5), headers=HEADERS)
        response.raise_for_status()
        payload = response.json()
    finally:
        if owned:
            client.close()
    candidates = []
    for item in (payload.get('items') or [])[:5]:
        info = item.get('volumeInfo') or {}
        if not item.get('id') or not info.get('title'):
            continue
        candidates.append(WorkCandidate(
            key=item['id'], title=info['title'], authors=info.get('authors', []),
            first_publish_year=_year(info.get('publishedDate')),
            url=info.get('canonicalVolumeLink') or info.get('infoLink'),
            publisher=info.get('publisher'), isbn=volume_isbn(info)))
    return candidates


def _open_library(path: str, params: dict, client: httpx.Client | None) -> dict:
    """Fetch JSON from Open Library.

    Args:
        path: URL path appended to the Open Library base.
        params: Query parameters.
        client: Optional HTTP client, used by tests.

    Returns:
        The decoded JSON body.

    Raises:
        httpx.HTTPError: The request failed or returned an error status.
    """
    owned = client is None
    client = client or httpx.Client(timeout=15)
    try:
        response = client.get(OPEN_LIBRARY + path, params=params, headers=HEADERS)
        response.raise_for_status()
        return response.json()
    finally:
        if owned:
            client.close()


def search_works(title: str, author: str | None, *, client: httpx.Client | None = None) -> list[WorkCandidate]:
    """Search Open Library works by title and, when read, author.

    Args:
        title: Title read from the spine.
        author: Author read from the spine, if any.
        client: Optional HTTP client, used by tests.

    Returns:
        Up to five candidate works.

    Raises:
        httpx.HTTPError: The request failed or returned an error status.
    """
    params = {'title': title, 'limit': 5, 'fields': 'key,title,author_name,first_publish_year'}
    if author:
        params['author'] = author
    docs = _open_library('/search.json', params, client).get('docs', [])
    return [WorkCandidate(doc['key'], doc.get('title', ''), doc.get('author_name', []),
                          doc.get('first_publish_year'), source='openlibrary',
                          url=OPEN_LIBRARY + doc['key'])
            for doc in docs[:5] if doc.get('key')]


def matching_edition(work_key: str, publisher: str | None, *, client: httpx.Client | None = None) -> dict | None:
    """Pick an Open Library edition only when the spine's publisher identifies exactly one.

    Args:
        work_key: Open Library work key, e.g. '/works/OL1W'.
        publisher: Publisher read from the spine.
        client: Optional HTTP client, used by tests.

    Returns:
        Edition label and ISBN, or None when the evidence does not single out one edition.

    Raises:
        httpx.HTTPError: The request failed or returned an error status.
    """
    if not publisher:
        return None
    entries = _open_library(f'{work_key}/editions.json', {'limit': 50}, client).get('entries', [])
    matches = [entry for entry in entries
               if any(normalize(name) == normalize(publisher) for name in entry.get('publishers', []))]
    if len(matches) != 1:
        return None
    edition = matches[0]
    isbn = (edition.get('isbn_13') or edition.get('isbn_10') or [None])[0]
    return {'edition': f"{edition.get('title', 'Edition')} ({edition['publishers'][0]})", 'isbn': isbn}


def catalogue_sources() -> list[tuple[str, object]]:
    """Catalogues to try, in order.

    Google Books answers first because the same record also prices the book;
    Open Library needs no key, so it still answers when the key is missing,
    over quota, or the volume is not listed.

    Returns:
        (provider name, search function) pairs.
    """
    return [('Google Books', search_google_books), ('Open Library', search_works)]


def choose_work(readings: dict, candidates: list[WorkCandidate], *, client: httpx.Client | None = None) -> WorkCandidate | None:
    """Let Drex pick the work the spine shows, or none.

    Args:
        readings: Spine readings (title, author, publisher, visible_text).
        candidates: Catalogue candidates.
        client: Optional HTTP client, used by tests.

    Returns:
        The chosen work when its probability reaches `drex.catalogue_min`, else None.

    Raises:
        DrexError: The decision could not be obtained.
    """
    if not candidates:
        return None
    criteria = {f'c{index}': f"{item.title} by {', '.join(item.authors) or 'unknown author'}"
                             f" ({item.first_publish_year or 'year unknown'})" for index, item in enumerate(candidates)}
    criteria['none'] = 'None of these works matches the spine'
    result = decide({'match': choice('Which catalogue work is the book whose spine was read?', criteria)},
                    state={'spine_readings': readings}, client=client, stage='d3')
    answer = result.answers['match']
    if answer.choice == 'none' or answer.choice not in criteria:
        return None
    if answer.probabilities.get(answer.choice, 0) < CONFIG.drex.catalogue_min:
        return None
    return candidates[int(answer.choice[1:])]


def _edition(work: WorkCandidate, publisher: str | None, http: httpx.Client | None = None) -> dict | None:
    """Edition and ISBN for a matched entry, only when the spine's publisher singles one out.

    The publisher read from the spine must match the record's, so an ISBN is
    never attached to a printing the sweep did not actually see. A Google Books
    volume carries its own publisher; an Open Library work needs its editions
    walked to find the one printing that publisher names.

    Args:
        work: Matched catalogue entry.
        publisher: Publisher read from the spine.
        http: Optional HTTP client, used by tests.

    Returns:
        Edition label and ISBN, or None when the evidence does not single out one edition.

    Raises:
        httpx.HTTPError: The Open Library editions request failed.
    """
    if work.source == 'openlibrary':
        return matching_edition(work.key, publisher, client=http)
    if publisher and work.publisher and normalize(publisher) == normalize(work.publisher):
        return {'edition': work.edition or f'{work.title} ({work.publisher})', 'isbn': work.isbn}
    return None


def resolve_book(database: Path, sweep_id: str, book: dict, *, http: httpx.Client | None = None,
                 drex: httpx.Client | None = None) -> None:
    """Append a catalogue observation for a legible book, or a review issue when none fits.

    Args:
        database: Capture SQLite path.
        sweep_id: Owning sweep.
        book: Inventory entry with a title and at least one visual observation.
        http: Optional catalogue client, used by tests.
        drex: Optional Drex client, used by tests.
    """
    if not book.get('title') or not book.get('observations'):
        return
    latest = book['observations'][-1]
    readings = {key: book.get(key) for key in ('title', 'author', 'publisher')}
    work, answered = None, False
    # Each catalogue is tried in turn until Drex accepts a match; a failing catalogue is skipped.
    for provider, search in catalogue_sources():
        try:
            # One catalogue block in the flow view; its provider line names who answered.
            with telemetry.span('catalogue', book['title'][:40]) as reported:
                reported.model = provider
                candidates = search(book['title'], book.get('author'), client=http)
                reported.detail = f'{len(candidates)} candidate(s)'
            work = choose_work(readings, candidates, client=drex)
            answered = True
        except DrexError as error:
            # Without a decision nothing is recorded; the lookup repeats on the next reading change.
            logger.warning('catalogue decision failed: %s', type(error).__name__)
            return
        except (httpx.HTTPError, ValueError, KeyError) as error:
            logger.warning('%s lookup failed: %s', provider, type(error).__name__)
            continue
        if work is not None:
            break
    if not answered:
        return
    try:
        edition = _edition(work, book.get('publisher'), http) if work else None
    except (httpx.HTTPError, ValueError, KeyError):
        edition = None
    with Repository(database) as repo:
        if work is None:
            repo.add(ReviewIssue(sweep_id=sweep_id, book_id=book['id'], field='catalogue',
                                 reason='No confident catalogue match'))
            return
        repo.add(BookObservation(sweep_id=sweep_id, book_id=book['id'], frame_id=latest['frame_id'],
                                 bbox=latest['bbox'], provenance='catalogue', association_method='catalogue',
                                 title=work.title, author=', '.join(work.authors) or None,
                                 edition=edition['edition'] if edition else None,
                                 isbn=edition['isbn'] if edition else None,
                                 catalogue_url=work.url, retrieved_at=datetime.now(timezone.utc),
                                 source_model=CONFIG.drex.model))
