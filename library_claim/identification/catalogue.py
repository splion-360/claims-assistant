"""Resolve legible spine readings to catalogue entries (Open Library, Library of Congress, Google Books), with Drex choosing."""
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import re
from xml.etree import ElementTree

import httpx

from library_claim import telemetry
from library_claim.config import CONFIG
from library_claim.decisions.drex import DrexError, choice, decide
from library_claim.domain.models import BookObservation, ReviewIssue
from library_claim.storage.repository import Repository
from .matching import normalize

OPEN_LIBRARY = 'https://openlibrary.org'
GOOGLE_BOOKS = 'https://www.googleapis.com/books/v1/volumes'
LIBRARY_OF_CONGRESS = 'https://lx2.loc.gov/sru/lcdb'
MODS = {'m': 'http://www.loc.gov/mods/v3'}
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
    # Library of Congress and Google Books records carry their own link, publisher, edition and ISBN;
    # Open Library editions are fetched separately.
    url: str | None = None
    publisher: str | None = None
    isbn: str | None = None
    edition: str | None = None


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


def search_library_of_congress(title: str, author: str | None, *,
                               client: httpx.Client | None = None) -> list[WorkCandidate]:
    """Search the Library of Congress catalogue (SRU, MODS records) by title and, when read, author.

    Args:
        title: Title read from the spine.
        author: Author read from the spine, if any.
        client: Optional HTTP client, used by tests.

    Returns:
        Up to five catalogue records.

    Raises:
        ValueError: The catalogue returned something other than MODS XML.
    """
    # Quotes would end the CQL phrase early, so they are dropped from spine readings.
    query = f'dc.title="{title.replace(chr(34), "")}"'
    if author:
        query += f' and dc.creator="{author.replace(chr(34), "")}"'
    params = {'version': '1.1', 'operation': 'searchRetrieve', 'maximumRecords': 5, 'recordSchema': 'mods', 'query': query}
    owned = client is None
    client = client or httpx.Client(timeout=15)
    try:
        response = client.get(LIBRARY_OF_CONGRESS, params=params, headers=HEADERS)
        response.raise_for_status()
        root = ElementTree.fromstring(response.text)
    except ElementTree.ParseError:
        raise ValueError('Library of Congress returned invalid XML') from None
    finally:
        if owned:
            client.close()

    def text(node, path):
        """First matching text, with trailing catalogue punctuation removed.

        Args:
            node: MODS element.
            path: Namespaced XPath.

        Returns:
            Cleaned text, or None.
        """
        found = node.find(path, MODS)
        return found.text.strip().rstrip(',:;/ ').strip() if found is not None and found.text else None

    candidates = []
    for record in root.iter('{http://www.loc.gov/mods/v3}mods'):
        lccn = text(record, "m:identifier[@type='lccn']")
        isbn = text(record, "m:identifier[@type='isbn']")
        primary = record.find("m:name[@usage='primary']/m:namePart", MODS)
        year = re.search(r'\d{4}', text(record, 'm:originInfo/m:dateIssued') or '')
        # Catalogues file leading articles ("The") separately so they do not affect sorting.
        title = ' '.join(part for part in (text(record, 'm:titleInfo/m:nonSort'), text(record, 'm:titleInfo/m:title')) if part)
        candidates.append(WorkCandidate(
            key=lccn or title, title=title,
            authors=[primary.text.strip().rstrip(',. ')] if primary is not None and primary.text else [],
            first_publish_year=int(year.group()) if year else None, source='loc',
            url=f'https://lccn.loc.gov/{lccn}' if lccn else None,
            publisher=_publisher(record),
            isbn=isbn.split()[0] if isbn else None, edition=text(record, 'm:originInfo/m:edition')))
    return candidates[:5]


def _publisher(record) -> str | None:
    """Publisher of a MODS record, from a publisher agent or the older publisher element.

    Args:
        record: MODS record element.

    Returns:
        Publisher name without trailing catalogue punctuation, or None.
    """
    for agent in record.iterfind('m:originInfo/m:agent', MODS):
        role = agent.find('m:role/m:roleTerm', MODS)
        name = agent.find('m:namePart', MODS)
        if role is not None and (role.text or '').strip().lower() == 'publisher' and name is not None and name.text:
            return name.text.strip().rstrip(',:;/ ').strip()
    found = record.find('m:originInfo/m:publisher', MODS)
    return found.text.strip().rstrip(',:;/ ').strip() if found is not None and found.text else None


def catalogue_sources() -> list[tuple[str, object]]:
    """Catalogues to try, in order; Google Books only with a key, since it throttles anonymous use.

    Returns:
        (stage name, search function) pairs.
    """
    sources = [('openlibrary', search_works), ('loc', search_library_of_congress)]
    if os.getenv('GOOGLE_BOOKS_API_KEY'):
        sources.append(('googlebooks', search_google_books))
    return sources


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
    if work.source != 'openlibrary':
        if publisher and work.publisher and normalize(publisher) == normalize(work.publisher):
            return {'edition': work.edition or f'{work.title} ({work.publisher})', 'isbn': work.isbn}
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
    # Each catalogue is tried in turn until Drex accepts a match; a failing catalogue is skipped.
    for stage, search in catalogue_sources():
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
