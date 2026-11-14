"""Apply spoken corrections as user-statement evidence without overwriting what the camera saw."""
from pathlib import Path

from library_claim.domain.models import Book, BookObservation, ReviewIssue
from library_claim.storage.repository import Repository


def apply_correction(database: Path, sweep_id: str, intent: str, book_ids: list[str], frame_id: str,
                     statement: str, stated_title: str | None = None) -> None:
    """Record a correction and update the flags it is allowed to change.

    Args:
        database: Capture SQLite path.
        sweep_id: Owning sweep.
        intent: D5 intent other than 'question_or_chat'.
        book_ids: Books the correction refers to.
        frame_id: Frame the user was looking at.
        statement: The user's words.
        stated_title: Title extracted from a 'correct_title' statement.
    """
    with Repository(database) as repo:
        visual = [row for row in repo.list_for_sweep(BookObservation, sweep_id)
                  if row.provenance == 'visual' and row.book_id in book_ids]
        for book_id in book_ids:
            rows = [row for row in visual if row.book_id == book_id]
            if not rows:
                continue
            # Anchor the statement to the book's box in the frame the user meant, if it was seen there.
            anchor = next((row for row in reversed(rows) if row.frame_id == frame_id), rows[-1])
            title = stated_title if intent == 'correct_title' else None
            records = [BookObservation(sweep_id=sweep_id, book_id=book_id, frame_id=anchor.frame_id, bbox=anchor.bbox,
                                       provenance='user_statement', association_method='user_statement',
                                       visible_text=statement[:500], title=title)]
            if intent == 'correct_title':
                # The user may be wrong; keep the spine reading and let a reviewer decide.
                records.append(ReviewIssue(sweep_id=sweep_id, book_id=book_id, field='title',
                                           reason='User-stated title differs from spine reading'))
            repo.add_many(records)
            book = repo.get(Book, book_id)
            if intent == 'exclude_books':
                repo.update(book.model_copy(update={'excluded': True}))
            elif intent in ('mark_first_edition', 'mark_signed'):
                repo.update(book.model_copy(update={'needs_appraisal': True}))
