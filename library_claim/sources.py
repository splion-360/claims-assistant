"""Share one Google Books access point between catalogue and price lookups."""
import os

GOOGLE_BOOKS = 'https://www.googleapis.com/books/v1/volumes'
# Generic client name only; no policyholder or project details reach an outside source.
HEADERS = {'User-Agent': 'library-claim/1'}


def google_books_params(**params) -> dict:
    """Add the configured API key to Google Books query parameters.

    The key lifts the shared anonymous quota. Without one the source refuses
    most requests, which callers report rather than work around.

    Args:
        **params: Query parameters for the volumes endpoint.

    Returns:
        The parameters, carrying the key when one is configured.
    """
    key = os.environ.get('GOOGLE_BOOKS_API_KEY')
    return {**params, 'key': key} if key else dict(params)


def volume_isbn(volume_info: dict) -> str | None:
    """Pick the most specific ISBN a volume record carries.

    Args:
        volume_info: The `volumeInfo` block of a Google Books volume.

    Returns:
        An ISBN-13, else an ISBN-10, else None.
    """
    identifiers = {entry.get('type'): entry.get('identifier')
                   for entry in volume_info.get('industryIdentifiers') or []}
    return identifiers.get('ISBN_13') or identifiers.get('ISBN_10')
