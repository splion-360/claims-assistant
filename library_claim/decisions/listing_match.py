"""Decide which store listings are a fair replacement for an object seen in the room."""
import httpx

from .drex import decide, noul

STATE_NOTE = ('The object was seen in a room during an insurance contents sweep. Each listing is a '
              'store search result. Judge only whether the listing is the thing described, not its '
              'price. Accessories, spare parts, bundles, toys and miniatures do not count.')


def listing_probabilities(item: dict, titles: list[str], *, client: httpx.Client | None = None) -> list[float]:
    """Ask Drex, in one request, whether each listing would replace the object.

    With a legible brand and model the question is "same product"; without one it is
    "comparable replacement" (same kind of object, similar size and class).

    Args:
        item: Inventory entry with category, description, material and brand_model.
        titles: Listing titles, in store order.
        client: Optional HTTP client, used by tests.

    Returns:
        One probability per listing, in order.

    Raises:
        DrexError: The decision could not be obtained.
    """
    exact = bool(item.get('brand_model'))
    state = {'note': STATE_NOTE,
             'object': {key: item.get(key) for key in ('category', 'description', 'material', 'brand_model')},
             'listings': [{'listing': index + 1, 'title': title} for index, title in enumerate(titles)]}
    wording = ('the same product, same brand and model, as the object' if exact else
               'a comparable replacement for the object: the same kind of item, of similar size and class')
    questions = {f'l{index + 1}': noul(f'Is listing {index + 1} {wording}?') for index in range(len(titles))}
    result = decide(questions, state=state, client=client, stage='listing_match')
    return [result.answers[f'l{index + 1}'].noul for index in range(len(titles))]
