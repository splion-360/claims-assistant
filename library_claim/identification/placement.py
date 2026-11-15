"""Place books on shelves and number them along each shelf from tracked frame geometry.

Frames tracked to the same reference share one coordinate basis, so spines seen in
different frames of one pan can be ordered together. A view with no tracked link to
an earlier one starts a new unit, since nothing measured relates the two. Within a
unit, spines are split into shelves by height in the image and ordered left to right.
"""
import statistics

import numpy as np

from library_claim.capture.tracking.geometry import transform
from library_claim.domain.models import BookObservation, Frame, FrameAssociation
from library_claim.storage.repository import Repository
from .storage import polygon, reference_geometry

# A vertical gap larger than this share of a typical spine height starts a new shelf.
ROW_GAP = 0.6


def place_books(repo: Repository, sweep_id: str, book_ids: list[str]) -> dict[str, tuple[str, int | None]]:
    """Give each book a shelf label and a position along that shelf.

    Args:
        repo: Open repository.
        sweep_id: Owning sweep.
        book_ids: Books to place.

    Returns:
        (shelf label, 1-based position) by book ID; ('', None) for a book with no usable sighting.
    """
    associations = {row.frame_id: row for row in repo.list_for_sweep(FrameAssociation, sweep_id)}
    captured = {frame.id: frame.captured_at for frame in repo.list_for_sweep(Frame, sweep_id)}
    sightings = {}
    for row in repo.list_for_sweep(BookObservation, sweep_id):
        # The first camera sighting places the book; catalogue and spoken rows carry no geometry.
        if row.provenance != 'visual' or row.book_id in sightings:
            continue
        try:
            root, matrix = reference_geometry(row.frame_id, associations)
        except ValueError:
            continue
        corners = transform(polygon(row.bbox), matrix)
        x, y = corners.mean(axis=0)
        height = float(np.ptp(corners[:, 1]))
        sightings[row.book_id] = (root, float(x), float(y), height)
    units = sorted({root for root, *_ in sightings.values()}, key=lambda root: captured.get(root) or 0)
    places = {book_id: ('', None) for book_id in book_ids}
    for unit_number, root in enumerate(units, start=1):
        members = sorted(((book_id, x, y, h) for book_id, (r, x, y, h) in sightings.items()
                          if r == root and book_id in places), key=lambda entry: entry[2])
        if not members:
            continue
        gap = ROW_GAP * statistics.median(h for *_, h in members)
        rows, current = [], [members[0]]
        for entry in members[1:]:
            if entry[2] - current[-1][2] > gap:
                rows.append(current)
                current = []
            current.append(entry)
        rows.append(current)
        for row_number, row in enumerate(rows, start=1):
            for position, (book_id, *_) in enumerate(sorted(row, key=lambda entry: entry[1]), start=1):
                places[book_id] = (f'unit {unit_number}, shelf {row_number}', position)
    return places
