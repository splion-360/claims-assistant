"""Decide whether a camera frame shows books and what limits reading them (D1)."""
from datetime import datetime, timezone
from pathlib import Path

import httpx
from pydantic import BaseModel

from library_claim.config import CONFIG
from library_claim.decisions.drex import choice, decide, noul
from library_claim.domain.models import ProcessingJob
from library_claim.storage.repository import Repository

ISSUES = {
    'none': 'Nothing; text is clear',
    'blur': 'Motion or focus blur',
    'glare': 'Reflections or glare',
    'dark': 'Too dark',
    'far': 'Books too small or far away',
    'occluded': 'Books blocked or cut off',
    'no_books': 'No books present',
}
QUESTIONS = {
    'book_visible': noul('Is at least one physical book visible in the image?'),
    'spine_legible': noul('Could a person read the title text on a visible book spine or cover in this image?'),
    'issue': choice('What most limits reading book text in this image?', ISSUES),
}
STATE = 'Live camera frame from a home-library inventory sweep.'


class FrameCheck(BaseModel):
    """Drex answers for one frame with the provenance to trace and cost them."""
    book_visible: float
    spine_legible: float
    issue: str
    issue_probabilities: dict[str, float]
    model: str
    request_id: str | None
    latency_ms: float
    input_tokens: int


def check_frame(jpeg: bytes, *, client: httpx.Client | None = None) -> FrameCheck:
    """Ask Drex the three D1 questions about one JPEG frame.

    Args:
        jpeg: Encoded camera frame.
        client: Optional HTTP client, used by tests.

    Returns:
        The frame check.

    Raises:
        DrexError: The decision could not be obtained.
    """
    result = decide(QUESTIONS, state=STATE, images=[jpeg], client=client, stage='d1')
    answers = result.answers
    return FrameCheck(book_visible=answers['book_visible'].noul, spine_legible=answers['spine_legible'].noul,
                      issue=answers['issue'].choice, issue_probabilities=answers['issue'].probabilities,
                      model=result.model, request_id=result.request_id, latency_ms=result.latency_ms,
                      input_tokens=result.input_tokens)


def is_book_candidate(check: FrameCheck | None) -> bool:
    """Admit a frame to spine reading only when Drex sees a book.

    Args:
        check: The frame's D1 result, or None when no decision exists.

    Returns:
        Whether the frame should go to text extraction.
    """
    return check is not None and check.book_visible >= CONFIG.drex.book_visible_min


def save_frame_check(database: Path, sweep_id: str, frame_id: str, check: FrameCheck) -> None:
    """Persist a D1 result beside its immutable saved frame.

    Args:
        database: SQLite database holding the frame.
        sweep_id: Owning sweep.
        frame_id: Saved frame the check describes.
        check: The frame check to store.
    """
    job_id = f'check_{frame_id}'
    with Repository(database) as repo:
        try:
            repo.get(ProcessingJob, job_id)
            return
        except KeyError:
            pass
        repo.add(ProcessingJob(id=job_id, sweep_id=sweep_id, frame_id=frame_id, task='frame_check',
                               status='succeeded', created_at=datetime.now(timezone.utc), model=check.model,
                               result=check.model_dump()))
