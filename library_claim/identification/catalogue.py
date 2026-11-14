"""Resolve legible spine readings to catalogue entries (Open Library, then Google Books), with Drex choosing the match."""
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import os
from pathlib import Path

import httpx

from library_claim import telemetry
from library_claim.config import CONFIG
from library_claim.decisions.drex import DrexError, choice, decide
from library_claim.domain.models import BookObservation, ReviewIssue
from library_claim.storage.repository import Repository
from .matching import normalize

OPEN_LIBRARY = 'https://openlibrary.org'
GOOGLE_BOOKS = 'https://www.googleapis.com/books/v1/volumes'
# Generic client name only; no user or project details go to either catalogue.
HEADERS = {'User-Agent': 'library-claim/1'}
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WorkCandidate:
    """One catalogue entry returned for a spine reading."""
    key: str
    title: str
    authors: list[str]
    first_publish_year: int | None
    source: str = 'openlibrary'
    # Google Books volumes carry their own link, publisher and ISBN; Open Library editions are fetched separately.
    url: str | None = None
    publisher: str | None = None
    isbn: str | None = None


def _get(path: str, params: dict | None, client: httpx.Client | None, base: str = OPEN_LIBRARY) -> dict:
    """Fetch JSON from a catalogue.

    Args:
        path: URL path appended to `base`.
        params: Query parameters.
        client: Optional HTTP client, used by tests.
        base: Catalogue base URL.

    Returns:
        The decoded JSON body.

    Raises:
        httpx.HTTPError: The request failed or returned an error status.
    """
    owned = client is None
    client = client or httpx.Client(timeout=15)
    try:
        response = client.get(base + path, params=params, headers=HEADERS)
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
    """
    params = {'title': title, 'limit': 5, 'fields': 'key,title,author_name,first_publish_year'}
    if author:
        params['author'] = author
    docs = _get('/search.json', params, client).get('docs', [])
    return [WorkCandidate(doc['key'], doc.get('title', ''), doc.get('author_name', []), doc.get('first_publish_year'))
            for doc in docs[:5] if doc.get('key')]


def matching_edition(work_key: str, publisher: str | None, *, client: httpx.Client | None = None) -> dict | None:
    """Pick an edition only when the spine's publisher identifies exactly one.

    Args:
        work_key: Open Library work key, e.g. '/works/OL1W'.
        publisher: Publisher read from the spine.
        client: Optional HTTP client, used by tests.

    Returns:
        Edition label and ISBN, or None when the evidence does not single out one edition.
    """
    if not publisher:
        return None
    entries = _get(f'{work_key}/editions.json', {'limit': 50}, client).get('entries', [])
    matches = [entry for entry in entries
               if any(normalize(name) == normalize(publisher) for name in entry.get('publishers', []))]
    if len(matches) != 1:
        return None
    edition = matches[0]
    isbn = (edition.get('isbn_13') or edition.get('isbn_10') or [None])[0]
    return {'edition': f"{edition.get('title', 'Edition')} ({edition['publishers'][0]})", 'isbn': isbn}


def search_google_books(title: str, author: str | None, *, client: httpx.Client | None = None) -> list[WorkCandidate]:
    """Search Google Books volumes by title and, when read, author.

    Args:
        title: Title read from the spine.
        author: Author read from the spine, if any.
        client: Optional HTTP client, used by tests.

    Returns:
        Up to five candidate volumes.
    """
    query = f'intitle:"{title}"' + (f' inauthor:"{author}"' if author else '')
    params = {'q': query, 'maxResults': 5, 'printType': 'books'}
    # An optional key only raises Google's anonymous rate limit.
    if os.getenv('GOOGLE_BOOKS_API_KEY'):
        params['key'] = os.environ['GOOGLE_BOOKS_API_KEY']
    candidates = []
    for item in _get('', params, client, GOOGLE_BOOKS).get('items', [])[:5]:
        info = item.get('volumeInfo', {})
        identifiers = {entry.get('type'): entry.get('identifier') for entry in info.get('industryIdentifiers', [])}
        year = (info.get('publishedDate') or '')[:4]
        candidates.append(WorkCandidate(
            key=item['id'], title=info.get('title', ''), authors=info.get('authors', []),
            first_publish_year=int(year) if year.isdigit() else None, source='googlebooks',
            url=info.get('infoLink') or f"https://books.google.com/books?id={item['id']}",
            publisher=info.get('publisher'), isbn=identifiers.get('ISBN_13') or identifiers.get('ISBN_10')))
    return candidates


def choose_work(readings: dict, candidates: list[WorkCandidate], *, client: httpx.Client | None = None) -> WorkCandidate | None:
    """Let Drex pick the work the spine shows, or none.

    Args:
        readings: Spine readings (title, author, publisher, visible_text).
        candidates: Open Library candidates.
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


def _edition(work: WorkCandidate, publisher: str | None, http: httpx.Client | None) -> dict | None:
    """Edition and ISBN for a matched entry, only when the spine's publisher singles one out.

    Args:
        work: Matched catalogue entry.
        publisher: Publisher read from the spine.
        http: Optional HTTP client, used by tests.

    Returns:
        Edition label and ISBN, or None.
    """
    if work.source == 'googlebooks':
        if publisher and work.publisher and normalize(publisher) == normalize(work.publisher):
            return {'edition': f'{work.title} ({work.publisher})', 'isbn': work.isbn}
        return None
    return matching_edition(work.key, publisher, client=http)


def resolve_book(database: Path, sweep_id: str, book: dict, *, http: httpx.Client | None = None,
                 drex: httpx.Client | None = None) -> None:
    """Append a catalogue observation for a legible book, or a review issue when none fits.

    Args:
        database: Capture SQLite path.
        sweep_id: Owning sweep.
        book: Inventory entry with a title and at least one visual observation.
        http: Optional Open Library client, used by tests.
        drex: Optional Drex client, used by tests.
    """
    if not book.get('title') or not book.get('observations'):
        return
    latest = book['observations'][-1]
    readings = {key: book.get(key) for key in ('title', 'author', 'publisher')}
    work, answered = None, False
    # Open Library first; Google Books when it fails, returns nothing, or Drex finds no match there.
    for stage, search in (('openlibrary', search_works), ('googlebooks', search_google_books)):
        try:
            with telemetry.span(stage, book['title'][:40]) as reported:
                candidates = search(book['title'], book.get('author'), client=http)
                reported.detail = f'{len(candidates)} candidate(s)'
            work = choose_work(readings, candidates, client=drex)
            answered = True
        except DrexError as error:
            # Without a decision nothing is recorded; the lookup repeats on the next reading change.
            logger.warning('catalogue decision failed: %s', type(error).__name__)
            return
        except (httpx.HTTPError, ValueError, KeyError) as error:
            logger.warning('%s lookup failed: %s', stage, type(error).__name__)
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
                                 catalogue_url=work.url or OPEN_LIBRARY + work.key, retrieved_at=datetime.now(timezone.utc),
                                 source_model=CONFIG.drex.model))
