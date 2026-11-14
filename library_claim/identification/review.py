"""Record book association decisions and reconcile immutable visual observations."""
from pathlib import Path

from library_claim.domain.models import Book, BookObservation, Frame, FrameAssociation
from library_claim.storage.repository import Repository
from library_claim.storage.evidence import EvidenceStore
from .models import BookDetection, BookFrameAnalysis


def identities(repo: Repository, sweep_id: str) -> tuple[dict, set]:
    """Resolve append-only review decisions into canonical IDs and separate pairs.

    Args:
        repo: Open repository.
        sweep_id: Owning sweep.

    Returns:
        Canonical book IDs and explicitly distinct pairs of canonical IDs.
    """
    roots = {book.id: book.id for book in repo.list_for_sweep(Book, sweep_id)}
    separate = []
    for source, target, action in repo.connection.execute(
            'SELECT source_id, target_id, action FROM book_reviews WHERE sweep_id=? ORDER BY id', (sweep_id,)):
        if action == 'separate':
            separate.append((source, target))
        else:
            old, new = roots[source], roots[target]
            roots = {key: new if value == old else value for key, value in roots.items()}
    return roots, {frozenset((roots[a], roots[b])) for a, b in separate}


def decide(repo: Repository, sweep_id: str, source: str, target: str, action: str, origin: str = 'user') -> None:
    """Append a validated identity decision without rewriting source observations.

    Args:
        repo: Open repository.
        sweep_id: Owning sweep.
        source: Source book ID.
        target: Target book ID.
        action: Same or separate.
        origin: User confirmation or automatic geometric/visual reconciliation.

    Raises:
        ValueError: IDs, action, or evidence contradict the requested decision.
    """
    if action not in ('same', 'separate') or origin not in ('user', 'automatic'):
        raise ValueError('Invalid review decision')
    # Serialize validation and insertion so overlapping reviews cannot create conflicts.
    with repo.connection:
        repo.connection.execute('BEGIN IMMEDIATE')
        roots, separate = identities(repo, sweep_id)
        if source not in roots or target not in roots:
            raise ValueError('Book does not belong to this sweep')
        a, b = roots[source], roots[target]
        if a == b:
            if action == 'same':
                return
            raise ValueError('These observations already belong to one book')
        observations = repo.list_for_sweep(BookObservation, sweep_id)
        left = {row.frame_id for row in observations if roots[row.book_id] == a}
        right = {row.frame_id for row in observations if roots[row.book_id] == b}
        if action == 'same' and (left & right or frozenset((a, b)) in separate):
            raise ValueError('Separate books cannot be merged: co-visible evidence or a previous decision')
        repo.connection.execute(
            'INSERT INTO book_reviews(sweep_id,source_id,target_id,action,origin) VALUES (?,?,?,?,?)',
            (sweep_id, a, b, action, origin))


def reconcile(repo: Repository, sweep_id: str, frame: Frame, image, database: Path, decide_same=None) -> None:
    """Revisit earlier fragmented records using a newly analyzed frame.

    Args:
        repo: Open repository containing the newly committed analysis.
        sweep_id: Owning sweep.
        frame: Newly analyzed source frame.
        image: Decoded image used for crop verification.
        database: SQLite path beside the evidence directory.
        decide_same: D2 decision function; defaults to the Drex same-book decision.
    """
    from .storage import association_choices, same_book_probabilities

    decide_same = decide_same or same_book_probabilities

    observations = [row for row in repo.list_for_sweep(BookObservation, sweep_id) if row.provenance == 'visual']
    frames = {row.id: row for row in repo.list_for_sweep(Frame, sweep_id)}
    mappings = {row.frame_id: row for row in repo.list_for_sweep(FrameAssociation, sweep_id)}
    for current in [row for row in observations if row.frame_id == frame.id]:
        if not current.title or current.visual_issues:
            continue
        roots, separate = identities(repo, sweep_id)
        root = roots[current.book_id]
        co_visible = {roots[row.book_id] for row in observations if row.frame_id == frame.id}
        candidates = [row.model_copy(update={'book_id': roots[row.book_id], 'association_method': 'review_candidate'})
                      for row in observations if roots[row.book_id] not in co_visible
                      and frozenset((root, roots[row.book_id])) not in separate]
        analysis = BookFrameAnalysis(books=[BookDetection(bbox=current.bbox, view=current.view or 'partial',
                                                         title=current.title, author=current.author,
                                                         issues=current.visual_issues)])
        matches, _ = association_choices(frame, analysis, frames, candidates, mappings, image,
                                         EvidenceStore(database.parent / 'frames'), decide_same)
        if matches[0] is not None:
            try:
                decide(repo, sweep_id, matches[0], root, 'same', 'automatic')
            except ValueError:
                # Historical co-visibility still takes precedence over a newer match.
                continue


def review_books(database: Path, sweep_id: str, source: str, target: str, action: str) -> list[dict]:
    """Persist a user's review and return refreshed inventory.

    Args:
        database: Capture SQLite path.
        sweep_id: Owning sweep.
        source: Selected book.
        target: Compared book.
        action: Same or separate.

    Returns:
        Updated canonical inventory.
    """
    from .storage import inventory

    with Repository(database) as repo:
        decide(repo, sweep_id, source, target, action)
        return inventory(repo, sweep_id)
