"""Decide whether a new detection is an already-known physical book (D2)."""
from dataclasses import dataclass
from typing import Literal

import cv2
import httpx
import numpy as np

from library_claim.config import CONFIG
from library_claim.decisions.drex import decide, noul
from library_claim.domain.models import Box
from library_claim.identification.models import BookDetection

STATE_NOTE = ('Book crops from a home-library camera sweep. Image 1 is a new detection; '
              'each later image is an existing book. Copies of the same title can be different physical books.')


@dataclass(frozen=True)
class SameBookCandidate:
    """An existing book that survived the hard association vetoes."""
    book_id: str
    crop_jpeg: bytes
    title: str | None
    author: str | None
    view: str | None
    score: float
    method: str


def crop_jpeg(image: np.ndarray, box: Box) -> bytes:
    """Encode the region of a book detection as a JPEG.

    Args:
        image: Decoded BGR source image.
        box: Normalized detection box.

    Returns:
        JPEG bytes of the crop (at least one pixel each way).

    Raises:
        ValueError: The crop cannot be encoded.
    """
    height, width = image.shape[:2]
    x0, x1 = int(box.x_min * width), max(int(box.x_max * width), int(box.x_min * width) + 1)
    y0, y1 = int(box.y_min * height), max(int(box.y_max * height), int(box.y_min * height) + 1)
    success, encoded = cv2.imencode('.jpg', image[y0:y1, x0:x1])
    if not success:
        raise ValueError('Cannot encode book crop')
    return encoded.tobytes()


def same_book_probabilities(detection_crop: bytes, detection: BookDetection, candidates: list[SameBookCandidate],
                            *, client: httpx.Client | None = None) -> list[float]:
    """Ask Drex, in one request, whether the detection is each candidate book.

    Args:
        detection_crop: JPEG crop of the new detection.
        detection: The new detection's readings.
        candidates: Up to three surviving candidates.
        client: Optional HTTP client, used by tests.

    Returns:
        One probability per candidate, in order.

    Raises:
        DrexError: The decision could not be obtained.
    """
    state = {'note': STATE_NOTE,
             'detection': {'title': detection.title, 'author': detection.author, 'view': detection.view},
             'candidates': [{'image': index + 2, 'title': item.title, 'author': item.author, 'view': item.view,
                             'evidence': item.method, 'score': round(item.score, 3)}
                            for index, item in enumerate(candidates)]}
    questions = {f'same_{index + 1}': noul(f'Is the book in image 1 the same physical book as the book in image {index + 2}?')
                 for index in range(len(candidates))}
    result = decide(questions, state=state, images=[detection_crop, *(item.crop_jpeg for item in candidates)],
                    client=client)
    return [result.answers[f'same_{index + 1}'].noul for index in range(len(candidates))]


def choose_same_book(probabilities: list[float]) -> int | None | Literal['unresolved']:
    """Turn same-book probabilities into a decision.

    Args:
        probabilities: One probability per candidate.

    Returns:
        The index of the single candidate at or above `same_book_min`; None when every
        candidate is at or below `different_book_max` (a new book); otherwise 'unresolved'.
    """
    above = [index for index, value in enumerate(probabilities) if value >= CONFIG.drex.same_book_min]
    if len(above) == 1:
        return above[0]
    if all(value <= CONFIG.drex.different_book_max for value in probabilities):
        return None
    # Two strong matches or a middling one are ambiguous; leave them for review.
    return 'unresolved'
