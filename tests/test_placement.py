"""Verify books are placed on shelves, in order, from tracked frame geometry."""
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from library_claim.domain.models import Book, BookObservation, Box, Frame, FrameAssociation, Sweep
from library_claim.identification.placement import place_books
from library_claim.storage.repository import Repository

START = datetime(2026, 11, 14, 9, tzinfo=timezone.utc)


class PlacementTests(unittest.TestCase):
    """Rows come from vertical position, order from horizontal position in a shared reference."""

    def setUp(self):
        """Save two shelf rows seen across two tracked frames, and a second unit."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.database = Path(directory.name) / 'claims.sqlite3'
        frames = [Frame(id=name, sweep_id='s', captured_at=START + timedelta(seconds=n),
                        evidence_key=f'{str(n) * 64}.jpg', width_px=640, height_px=480)
                  for n, name in enumerate(['f1', 'f2', 'f3'], start=1)]
        # f2 is f1 panned right: its x=0.4 is the reference's x=0.9.
        link = FrameAssociation(id='f2', sweep_id='s', frame_id='f2', reference_frame_id='f1',
                                matrix=((1, 0, 0.5), (0, 1, 0), (0, 0, 1)), method='optical_flow',
                                matched_points=30, reprojection_error_px=0.5)
        spines = {'A': ('f1', 0.2, 0.3), 'B': ('f1', 0.6, 0.3), 'C': ('f2', 0.4, 0.3),
                  'D': ('f1', 0.3, 0.8), 'E': ('f3', 0.5, 0.5)}
        records = [Sweep(id='s', captured_at=START), *frames, link]
        for name, (frame, x, y) in spines.items():
            records.append(Book(id=name, sweep_id='s'))
            records.append(BookObservation(sweep_id='s', book_id=name, frame_id=frame,
                                           bbox=Box(x_min=x - 0.03, y_min=y - 0.15, x_max=x + 0.03, y_max=y + 0.15),
                                           association_method='new_provisional', title=name))
        with Repository(self.database) as repo:
            repo.add_many(records)

    def placements(self):
        """Run placement over every book.

        Returns:
            (shelf, position) by book ID.
        """
        with Repository(self.database) as repo:
            return place_books(repo, 's', ['A', 'B', 'C', 'D', 'E', 'missing'])

    def test_order_along_a_shelf_follows_the_reference(self):
        """A, B and C (seen in a panned frame) are positions 1 to 3 on the top shelf."""
        places = self.placements()
        self.assertEqual([places[name] for name in 'ABC'], [('unit 1, shelf 1', n) for n in (1, 2, 3)])

    def test_lower_row_is_a_separate_shelf(self):
        """D sits lower, so it is the first book of shelf 2."""
        self.assertEqual(self.placements()['D'], ('unit 1, shelf 2', 1))

    def test_untracked_view_is_another_unit(self):
        """A frame with no tracked link to the first view starts unit 2."""
        self.assertEqual(self.placements()['E'], ('unit 2, shelf 1', 1))

    def test_book_without_sightings_is_left_empty(self):
        """No evidence, no placement."""
        self.assertEqual(self.placements()['missing'], ('', None))


if __name__ == '__main__':
    unittest.main()
