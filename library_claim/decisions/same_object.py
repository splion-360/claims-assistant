"""Decide whether two sightings show the same physical non-book object."""
from dataclasses import dataclass
import logging

import httpx

from library_claim.config import CONFIG
from .drex import decide, noul

logger = logging.getLogger(__name__)
STATE_NOTE = ('Each image is one detected object. Decide whether they are the same physical object '
              'seen again, not whether they are the same kind of thing. Two identical chairs in one '
              'room are different objects; one laptop seen from the front and from the side is one '
              'object. Judge the object itself, not the surroundings.')


@dataclass(frozen=True)
class SameObjectCandidate:
    """One saved object a new detection might be another sighting of."""
    item_id: str
    category: str
    description: str
    crop_jpeg: bytes
    # How much the two boxes overlap, when a shared coordinate basis exists.
    score: float = 0.0


def same_object_probabilities(detection_crop: bytes, category: str, description: str,
                              candidates: list[SameObjectCandidate], *,
                              client: httpx.Client | None = None) -> list[float]:
    """Ask Drex, in one request, whether the detection is each candidate object.

    Args:
        detection_crop: JPEG crop of the new detection.
        category: The new detection's category.
        description: The new detection's description.
        candidates: Up to three surviving candidates.
        client: Optional HTTP client, used by tests.

    Returns:
        One probability per candidate, in order.

    Raises:
        DrexError: The decision could not be obtained.
    """
    state = {'note': STATE_NOTE,
             'detection': {'category': category, 'description': description},
             'candidates': [{'image': index + 2, 'category': item.category,
                             'description': item.description, 'overlap': round(item.score, 3)}
                            for index, item in enumerate(candidates)]}
    questions = {f'same_{index + 1}':
                 noul(f'Is the object in image 1 the same physical object as the object in image {index + 2}?')
                 for index in range(len(candidates))}
    result = decide(questions, state=state, images=[detection_crop, *(item.crop_jpeg for item in candidates)],
                    client=client, stage='d2')
    return [result.answers[f'same_{index + 1}'].noul for index in range(len(candidates))]


def choose_same_object(probabilities: list[float]) -> int | None:
    """Turn same-object probabilities into one decision.

    Exactly one candidate must be confidently the same object. Competing
    confident answers leave the sighting unmatched rather than guessing which.

    Args:
        probabilities: One probability per candidate, in order.

    Returns:
        The index of the matching candidate, or None when none is established.
    """
    confident = [index for index, value in enumerate(probabilities)
                 if value is not None and value >= CONFIG.drex.same_book_min]
    return confident[0] if len(confident) == 1 else None
