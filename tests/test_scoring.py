"""Verify the packet scorer against hand-built ground truth on every pass bar."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from library_claim.evaluation import load_truth, results_markdown, score

ROOT = Path(__file__).resolve().parent.parent


def line(identifier, title, status='identified', height=None, thickness=None, price=None):
    """Build a minimal packet book line.

    Args:
        identifier: Book ID.
        title: Title read by the system, or '' when unread.
        status: Packet status.
        height: Spine height in cm.
        thickness: Spine thickness in cm.
        price: Replacement cost amount.

    Returns:
        A packet book line.
    """
    return {'id': identifier, 'title': title, 'status': status, 'spine_height_cm': height,
            'spine_thickness_cm': thickness, 'replacement_cost': {'amount': price}}


def packet(books, items=(), room=None, count=None, review=(), time_to_packet=None):
    """Build a minimal claim packet.

    Args:
        books: Book lines.
        items: Item lines.
        room: Room block.
        count: Reported book count; defaults to the number of lines.
        review: Review queue entries.
        time_to_packet: Seconds from sweep end to packet, or None when not recorded.

    Returns:
        A claim packet dictionary.
    """
    sweep = {'id': 's'} if time_to_packet is None else {'id': 's', 'time_to_packet_s': time_to_packet}
    return {'sweep': sweep, 'books': list(books), 'items': list(items),
            'room': room or {'floor_area_m2': None, 'wall_area_m2': None},
            'totals': {'book_count': len(books) if count is None else count}, 'review_queue': list(review)}


TRUTH = {
    'books': {'count': 20, 'titles': [{'title': f'Book {n}'} for n in range(10)],
              'spines': {'method': 'ruler', 'sample': [{'title': 'Book 0', 'height_cm': 20, 'thickness_cm': 2}]}},
    'prices': [{'title': 'Book 0', 'replacement': 10.0, 'currency': 'USD', 'source_type': 'google_books'}],
    'items': [{'category': 'lighting'}, {'category': 'lighting'}, {'category': 'rug'}, {'category': 'art'},
              {'category': 'seating'}],
    'room': {'method': 'tape', 'length_m': 4, 'width_m': 3, 'height_m': 2.5},
}


def bars(results):
    """Index bar results by name.

    Args:
        results: Scorer output.

    Returns:
        Results keyed by bar name.
    """
    return {result.name: result for result in results}


class ScoreTests(unittest.TestCase):
    """Each pass bar is scored from the packet against the ground truth."""

    def test_book_count_within_five_percent(self):
        """19 of 20 is a 5% miss and passes; 18 fails."""
        self.assertEqual(bars(score(packet([], count=19), TRUTH))['Book count'].status, 'pass')
        self.assertEqual(bars(score(packet([], count=18), TRUTH))['Book count'].status, 'fail')

    def test_title_rate_counts_each_true_title_once(self):
        """Seven correct legible titles out of ten pass, even when one is read twice."""
        books = [line(f'b{n}', f'BOOK {n}') for n in range(7)] + [line('dup', 'Book 0')]
        result = bars(score(packet(books), TRUTH))['Title identification']
        self.assertEqual(result.status, 'pass')
        self.assertIn('7/10', result.value)

    def test_confidently_wrong_titles_fail_the_bar(self):
        """A wrong title not sent to review counts against the 3% limit; a blank does not."""
        books = [line(f'b{n}', f'Book {n}') for n in range(8)] + [line('x', 'Invented Title'), line('u', '', 'unidentified')]
        result = bars(score(packet(books), TRUTH))['Title identification']
        self.assertEqual(result.status, 'fail')
        self.assertTrue(any('Invented Title' in detail for detail in result.details))

    def test_wrong_title_sent_to_review_is_not_confident(self):
        """A wrong title the system flagged for review is reported but not counted as confident."""
        books = [line(f'b{n}', f'Book {n}') for n in range(8)] + [line('x', 'Invented Title')]
        result = bars(score(packet(books, review=[{'ref_id': 'x', 'reason': 'unsure'}]), TRUTH))['Title identification']
        self.assertEqual(result.status, 'pass')

    def test_unmeasured_spine_counts_as_full_error(self):
        """A sampled spine the system did not measure cannot be skipped to raise the score."""
        self.assertEqual(bars(score(packet([line('b', 'Book 0')]), TRUTH))['Spine dimensions'].status, 'fail')
        close = packet([line('b', 'Book 0', height=21, thickness=2.2)])
        self.assertEqual(bars(score(close, TRUTH))['Spine dimensions'].status, 'pass')

    def test_price_within_twenty_five_percent(self):
        """12 against a hand-checked 10 passes; 13 fails."""
        self.assertEqual(bars(score(packet([line('b', 'Book 0', price=12)]), TRUTH))['Book prices'].status, 'pass')
        self.assertEqual(bars(score(packet([line('b', 'Book 0', price=13)]), TRUTH))['Book prices'].status, 'fail')

    def test_items_match_by_category_one_to_one(self):
        """Four of five items with the right category pass; extra lamps do not count twice."""
        items = [{'id': f'i{n}', 'category': category} for n, category in
                 enumerate(['lighting', 'lighting', 'lighting', 'rug', 'art'])]
        result = bars(score(packet([], items=items), TRUTH))['Non-book items']
        self.assertEqual(result.status, 'pass')
        self.assertIn('4/5', result.value)

    def test_room_areas_from_tape_dimensions(self):
        """Floor 12 m² and wall 35 m² come from length, width and height."""
        room = {'floor_area_m2': 11.0, 'wall_area_m2': 41.0}
        results = bars(score(packet([], room=room), TRUTH))
        self.assertEqual(results['Floor area'].status, 'pass')
        self.assertEqual(results['Wall area'].status, 'fail')

    def test_missing_system_value_fails_and_missing_truth_is_not_measured(self):
        """No room from the system fails; no truth for a bar marks it not measured."""
        self.assertEqual(bars(score(packet([]), TRUTH))['Floor area'].status, 'fail')
        results = bars(score(packet([]), {'books': {'count': 3}}))
        self.assertEqual(results['Floor area'].status, 'not measured')
        self.assertEqual(results['Book prices'].status, 'not measured')

    def test_time_to_packet(self):
        """Under 300 s passes; an unrecorded time fails rather than passing silently."""
        self.assertEqual(bars(score(packet([], time_to_packet=240), TRUTH))['Time to packet'].status, 'pass')
        self.assertEqual(bars(score(packet([]), TRUTH))['Time to packet'].status, 'fail')

    def test_markdown_lists_every_bar(self):
        """The results table has one row per pass bar."""
        text = results_markdown(score(packet([]), TRUTH))
        for name in ('Book count', 'Title identification', 'Spine dimensions', 'Book prices',
                     'Non-book items', 'Floor area', 'Wall area', 'Time to packet'):
            self.assertIn(f'| {name} |', text)


class TruthFileTests(unittest.TestCase):
    """The shipped template loads and the command line scores a packet file."""

    def test_template_and_example_load(self):
        """Both ground-truth files parse into the shape the scorer reads."""
        for name in ('template.yml', 'example.yml'):
            truth = load_truth(ROOT / 'ground_truth' / name)
            self.assertIn('books', truth)

    def test_command_line_writes_results(self):
        """The script reads a packet and a truth file and writes the Markdown table."""
        with tempfile.TemporaryDirectory() as directory:
            packet_path, truth_path, out = (Path(directory) / name for name in ('p.json', 't.yml', 'r.md'))
            packet_path.write_text(json.dumps(packet([line('b', 'Book 0', price=10)])))
            truth_path.write_text(yaml.safe_dump(TRUTH))
            subprocess.run([sys.executable, str(ROOT / 'scripts' / 'score_packet.py'), str(packet_path),
                            str(truth_path), '--out', str(out)], check=True, cwd=ROOT, capture_output=True)
            self.assertIn('| Book prices |', out.read_text())


if __name__ == '__main__':
    unittest.main()
