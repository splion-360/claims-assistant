"""Compose what the agent says out loud when it reads new book titles."""
from library_claim.identification.matching import normalize


def _spoken(title: str) -> str:
    """Make an all-caps spine reading sound natural without changing its words.

    Args:
        title: Title as read from the spine.

    Returns:
        The title in title case if it was all caps, otherwise unchanged.
    """
    return title.title() if title.isupper() else title


def announcement(books: list[dict], announced: set[str]) -> tuple[str | None, set[str]]:
    """Build one sentence naming titles not announced yet.

    Args:
        books: Inventory entries.
        announced: Normalized titles already spoken this session.

    Returns:
        The sentence (or None when nothing is new) and the normalized titles it names.
    """
    new = {}
    for book in books:
        title = book.get('title')
        if not title or book.get('excluded'):
            continue
        key = normalize(title)
        # One owned copy per title: a repeat sighting or unmerged duplicate is not news.
        if key and key not in announced and key not in new:
            new[key] = _spoken(title)
    if not new:
        return None, set()
    titles = list(new.values())
    if len(titles) == 1:
        text = f'I found a book called {titles[0]}.'
    else:
        listed = ', '.join(titles[:-1]) + f' and {titles[-1]}'
        text = f"I found {'two' if len(titles) == 2 else len(titles)} books: {listed}."
    return text, set(new)
