"""Decide what a spoken correction means and which books it refers to (D5)."""
import httpx
from pydantic import BaseModel

from library_claim.config import CONFIG
from library_claim.decisions.drex import DrexError, choice, decide

INTENTS = {
    'exclude_books': 'The user says some visible books are not theirs or should be skipped',
    'mark_first_edition': 'The user says a book is a first edition',
    'mark_signed': 'The user says a book is signed',
    'correct_title': 'The user says a book has a different title than the agent read',
    'question_or_chat': 'Anything else: a question, an answer, or conversation',
}
# A correction refers to what the user was just looking at; older frames are not "that".
TARGET_WINDOW_S = 30.0


class StatedTitle(BaseModel):
    """Title the user stated in a correction, or None when they did not say one."""
    title: str | None


def classify_intent(text: str, *, client: httpx.Client | None = None) -> str:
    """Classify a user utterance during a sweep.

    Args:
        text: Final transcript of the user's turn.
        client: Optional HTTP client, used by tests.

    Returns:
        One of INTENTS; 'question_or_chat' when Drex fails.
    """
    try:
        result = decide({'intent': choice('What does the user want the inventory agent to do?', INTENTS)},
                        state={'utterance': text}, client=client, stage='d5')
    except DrexError:
        return 'question_or_chat'
    intent = result.answers['intent'].choice
    return intent if intent in INTENTS else 'question_or_chat'


def resolve_target(text: str, books: list[dict], *, allow_all: bool,
                   client: httpx.Client | None = None) -> list[str] | None:
    """Decide which of the books in view a correction refers to.

    Args:
        text: The user's utterance.
        books: Inventory entries seen in the target frame.
        allow_all: Whether "all visible books" is a valid answer (exclusions).
        client: Optional HTTP client, used by tests.

    Returns:
        Book IDs, or None when the reference is ambiguous and the agent should ask.
    """
    if len(books) == 1:
        return [books[0]['id']]
    if not books:
        return None
    # Describe books by left-to-right position so "the left one" can be resolved.
    ordered = sorted(books, key=lambda book: book['observations'][-1]['bbox']['x_min'] if book.get('observations') else 0)
    criteria = {book['id']: f"position {index + 1} from the left; title {book.get('title') or 'unreadable'}; "
                            f"author {book.get('author') or 'unreadable'}" for index, book in enumerate(ordered)}
    if allow_all:
        criteria['all_visible'] = 'Every book currently in view'
    try:
        result = decide({'target': choice('Which book or books does the user mean?', criteria)},
                        state={'utterance': text}, client=client, stage='d5')
    except DrexError:
        return None
    answer = result.answers['target']
    if answer.choice not in criteria or answer.probabilities.get(answer.choice, 0) < CONFIG.drex.target_min:
        return None
    return [book['id'] for book in books] if answer.choice == 'all_visible' else [answer.choice]


def target_frame(frames: list[tuple[str, float]], now: float) -> str | None:
    """Pick the most recent frame with detections that is recent enough to be "that".

    Args:
        frames: (frame ID, monotonic time) pairs for saved frames with detections.
        now: Current monotonic time.

    Returns:
        The frame ID, or None when no frame is within TARGET_WINDOW_S.
    """
    recent = [(seen, frame_id) for frame_id, seen in frames if now - seen <= TARGET_WINDOW_S]
    return max(recent)[1] if recent else None


async def extract_stated_title(text: str) -> str | None:
    """Extract the title a user stated, using the conversation model.

    Args:
        text: The user's correction.

    Returns:
        The stated title, or None.

    Raises:
        provider.InferenceError: The model call failed.
    """
    import provider

    result = await provider.structured('Extract the book title the user states in this correction. '
                                       'Return null when they do not state a title.', text, StatedTitle)
    return result.title
