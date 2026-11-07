"""Gate automatic guidance on missing titles and remembered evidence requests."""
from library_claim.config import CONFIG
from library_claim.identification.matching import text_relation


def needs_capture(book: dict) -> bool:
    """Decide whether another view can fill the current identification objective.

    Args:
        book: Canonical inventory record.

    Returns:
        True only for a missing title; association and optional metadata stay in review.
    """
    return not bool(book.get('title'))


def capture_instruction(book: dict) -> str:
    """Ask for one camera adjustment without asking the user to operate capture controls.

    Args:
        book: Selected inventory record.

    Returns:
        A brief evidence-focused camera instruction.
    """
    observations = book.get('observations', [])
    issues = observations[-1].get('issues', []) if observations else []
    if 'blur' in issues:
        return 'Hold the camera steady for a moment so I can read the title.'
    if 'glare' in issues:
        return 'Shift the camera slightly to the side so I can read the title without glare.'
    if 'occlusion' in issues or 'cut_off' in issues:
        return 'Adjust the camera angle to show the full spine without moving the books.'
    return 'Move a little closer to this shelf so I can read the unclear title.'


class GuidancePolicy:
    """Limit interventions across canonical IDs, candidate fragments, and time."""

    def __init__(self):
        """Initialize session-local request memory."""
        self.requests = []
        self.last_request_at = float('-inf')

    def remember(self, book: dict, now: float) -> None:
        """Remember a request across known aliases and candidate sightings.

        Args:
            book: Requested book and its possible aliases.
            now: Monotonic request time.
        """
        self.requests.append({'ids': set(book.get('member_ids', [])) | {book['id']} | set(book.get('candidates', [])),
                              'title': book.get('title')})
        self.last_request_at = now

    def choose(self, books: list[dict], now: float, frame_id: str) -> dict | None:
        """Select a visible missing title only when another intervention is warranted.

        Args:
            books: Reconciled inventory.
            now: Monotonic time.
            frame_id: Frame currently being inspected; avoid requests for offscreen books.

        Returns:
            One eligible book, or None to continue silently.
        """
        if now - self.last_request_at < CONFIG.identification.guidance_cooldown_s:
            return None
        if len(self.requests) >= CONFIG.identification.guidance_max_requests:
            return None
        # Expand remembered aliases as reconciliation discovers additional connections.
        for book in books:
            ids = set(book.get('member_ids', [])) | {book['id']} | set(book.get('candidates', []))
            for request in self.requests:
                if ids & request['ids']:
                    request['ids'].update(ids)
                    request['title'] = book.get('title') or request['title']
        for book in books:
            if not needs_capture(book) or frame_id not in book.get('frame_ids', []):
                continue
            ids = set(book.get('member_ids', [])) | {book['id']} | set(book.get('candidates', []))
            repeated = any(ids & request['ids'] or text_relation(book.get('title'), request['title']) in
                           ('exact', 'partial', 'fuzzy') for request in self.requests)
            # Unidentifiable fragments cannot reliably be distinguished, so don't keep asking.
            repeated |= any(not request['title'] for request in self.requests)
            if not repeated:
                return book
        return None
