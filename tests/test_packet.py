"""Verify the claim packet contract, its code-computed totals and its readable report."""
import io
import json
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from library_claim.config import CONFIG
from library_claim.domain.models import Book, BookObservation, Box, Frame, ReviewIssue, Sweep
from library_claim.packet import build_packet, packet_archive, render_report
from library_claim.storage.evidence import EvidenceStore
from library_claim.storage.repository import Repository

CONTRACT_BOOK_FIELDS = {'id', 'shelf', 'position', 'frame_ref', 'status', 'title', 'author', 'edition',
                        'isbn', 'spine_height_cm', 'spine_thickness_cm', 'id_confidence',
                        'replacement_cost', 'used_value'}
CONTRACT_ROOM_FIELDS = {'length_m', 'width_m', 'height_m', 'floor_area_m2', 'wall_area_m2',
                        'shelved_wall_area_m2', 'scale_method', 'confidence'}
CONTRACT_TOTALS = {'book_count', 'books_identified', 'books_unidentified', 'shelf_run_m',
                   'books_replacement_cost', 'books_used_value', 'items_replacement_cost_low',
                   'items_replacement_cost_high', 'excluded_from_totals'}


class PacketTests(unittest.TestCase):
    """Build packets from saved evidence and check what each field may claim."""

    def setUp(self):
        """Create an empty capture database for one sweep."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Path(self.directory.name) / 'claims.sqlite3'
        self.start = datetime(2026, 11, 14, 9, 0, tzinfo=timezone.utc)
        self.sweep_id = 'sweep1'

    def write(self, records):
        """Persist records into the capture database.

        Args:
            records: Domain records to insert.
        """
        with Repository(self.database) as repo:
            repo.add_many(records)

    def frame(self, identifier, key_byte, offset_s=0):
        """Build a saved frame record.

        Args:
            identifier: Frame ID.
            key_byte: Hex character repeated to form the evidence key.
            offset_s: Seconds after the sweep start.

        Returns:
            A frame record.
        """
        return Frame(id=identifier, sweep_id=self.sweep_id, captured_at=self.start + timedelta(seconds=offset_s),
                     evidence_key=f'{key_byte * 64}.jpg', width_px=640, height_px=480)

    def observation(self, book_id, frame_id, **fields):
        """Build a visual observation of one book.

        Args:
            book_id: Owning book.
            frame_id: Frame the reading came from.
            **fields: Overrides such as title or spine dimensions.

        Returns:
            An observation record.
        """
        return BookObservation(sweep_id=self.sweep_id, book_id=book_id, frame_id=frame_id,
                               bbox=Box(x_min=.1, y_min=.1, x_max=.5, y_max=.9),
                               association_method='new_provisional', **fields)

    def seed_one_identified_book(self):
        """Save a sweep whose single book was read from two frames."""
        self.write([
            Sweep(id=self.sweep_id, captured_at=self.start, device='browser camera'),
            self.frame('f1', 'a'), self.frame('f2', 'b', offset_s=42),
            Book(id='b1', sweep_id=self.sweep_id),
            self.observation('b1', 'f1', title='THE GOOD PEOPLE', author='HANNAH KENT',
                             spine_height_cm=19.5, spine_thickness_cm=3.2, scale_method='webxr_depth'),
            self.observation('b1', 'f2', title='THE GOOD PEOPLE'),
        ])

    def test_packet_keeps_every_contract_field(self):
        """Emit the sweep, room, book, totals and review sections the contract fixes."""
        self.seed_one_identified_book()
        packet = build_packet(self.database, self.sweep_id, price=False)
        self.assertEqual(set(packet), {'sweep', 'room', 'books', 'items', 'totals', 'review_queue'})
        self.assertLessEqual({'id', 'captured_at', 'device', 'duration_s', 'country', 'currency'}, set(packet['sweep']))
        self.assertLessEqual(CONTRACT_ROOM_FIELDS, set(packet['room']))
        self.assertLessEqual(CONTRACT_TOTALS, set(packet['totals']))
        self.assertLessEqual(CONTRACT_BOOK_FIELDS, set(packet['books'][0]))

    def test_frame_ref_names_a_saved_frame(self):
        """Point every frame reference at the stored JPEG for that frame."""
        self.seed_one_identified_book()
        packet = build_packet(self.database, self.sweep_id, price=False)
        line = packet['books'][0]
        self.assertEqual(line['frame_ref'], f'frames/{"a" * 64}.jpg')
        self.assertEqual(line['frame_refs'], [f'frames/{"a" * 64}.jpg', f'frames/{"b" * 64}.jpg'])

    def test_unpriced_books_stay_empty_and_leave_the_totals(self):
        """Leave an unknown price empty rather than writing a figure with no source."""
        self.seed_one_identified_book()
        packet = build_packet(self.database, self.sweep_id, price=False)
        line = packet['books'][0]
        self.assertIsNone(line['replacement_cost']['amount'])
        self.assertEqual(line['replacement_cost']['source'], '')
        self.assertIsNone(line['used_value']['amount'])
        self.assertEqual(packet['totals']['books_replacement_cost'], 0)
        self.assertEqual(packet['totals']['excluded_from_totals'], 1)

    def test_totals_are_summed_from_the_lines(self):
        """Count books and shelf run from the packet's own lines."""
        self.write([
            Sweep(id=self.sweep_id, captured_at=self.start, device='browser camera'),
            self.frame('f1', 'a'), self.frame('f2', 'b', offset_s=90),
            Book(id='b1', sweep_id=self.sweep_id), Book(id='b2', sweep_id=self.sweep_id),
            self.observation('b1', 'f1', title='READ SPINE', spine_height_cm=20, spine_thickness_cm=2.5),
            self.observation('b2', 'f2', spine_height_cm=18, spine_thickness_cm=4.5),
        ])
        totals = build_packet(self.database, self.sweep_id, price=False)['totals']
        self.assertEqual(totals['book_count'], 2)
        self.assertEqual(totals['books_identified'], 1)
        self.assertEqual(totals['books_unidentified'], 1)
        # 2.5 cm and 4.5 cm of spine stand 7 cm along the shelf.
        self.assertEqual(totals['shelf_run_m'], 0.07)

    def test_unreadable_spine_is_unidentified_without_a_guessed_title(self):
        """Record a book with no readable title as unidentified and leave the title blank."""
        self.write([
            Sweep(id=self.sweep_id, captured_at=self.start, device='browser camera'),
            self.frame('f1', 'a'), Book(id='b1', sweep_id=self.sweep_id),
            self.observation('b1', 'f1', spine_height_cm=21, spine_thickness_cm=3),
        ])
        line = build_packet(self.database, self.sweep_id, price=False)['books'][0]
        self.assertEqual(line['status'], 'unidentified')
        self.assertEqual(line['title'], '')
        self.assertEqual(line['spine_height_cm'], 21)

    def test_review_queue_reports_each_concern_once(self):
        """Carry saved review reasons, the unpriced line and the unmeasured room, without repeats."""
        self.seed_one_identified_book()
        self.write([ReviewIssue(sweep_id=self.sweep_id, book_id='b1', field='catalogue',
                                reason='No confident catalogue match')])
        queue = build_packet(self.database, self.sweep_id, price=False)['review_queue']
        self.assertEqual(len(queue), len({(entry['ref_id'], entry['reason']) for entry in queue}))
        reasons = {entry['reason'] for entry in queue}
        self.assertIn('No confident catalogue match', reasons)
        self.assertTrue(any(entry['ref_id'] == 'room' for entry in queue))
        self.assertTrue(any(entry['ref_id'] == 'items' for entry in queue))
        self.assertTrue(any('price source' in reason for reason in reasons))

    def test_unmeasured_room_and_items_stay_empty(self):
        """Leave the room and item sections unfilled while no pass measures them."""
        self.seed_one_identified_book()
        packet = build_packet(self.database, self.sweep_id, price=False)
        self.assertEqual(packet['items'], [])
        self.assertTrue(all(packet['room'][field] is None for field in
                            ('length_m', 'width_m', 'height_m', 'floor_area_m2', 'wall_area_m2')))
        self.assertEqual(packet['room']['scale_method'], '')

    def test_duration_spans_the_saved_frames(self):
        """Report the sweep duration as the span of its retained frames."""
        self.seed_one_identified_book()
        self.assertEqual(build_packet(self.database, self.sweep_id, price=False)['sweep']['duration_s'], 42.0)

    def test_packet_is_json_serialisable(self):
        """Keep the packet free of values that cannot be written to the structured file."""
        self.seed_one_identified_book()
        restored = json.loads(json.dumps(build_packet(self.database, self.sweep_id, price=False)))
        self.assertEqual(restored['books'][0]['title'], 'THE GOOD PEOPLE')

    def test_missing_sweep_is_reported(self):
        """Raise rather than invent a packet for an unknown sweep."""
        with self.assertRaises(KeyError):
            build_packet(self.database, 'absent', price=False)


class ReportTests(unittest.TestCase):
    """Check the readable report built from a packet."""

    def setUp(self):
        """Build a packet holding one identified and one unidentified book."""
        self.packet = {
            'sweep': {'id': 'sweep1', 'captured_at': '2026-11-14T09:00:00+00:00', 'device': 'browser camera',
                      'duration_s': 42.0, 'country': 'GB', 'currency': 'GBP', 'frames_saved': 2,
                      'generated_at': '2026-11-14T09:05:00+00:00'},
            'room': {'length_m': None, 'width_m': None, 'height_m': None, 'floor_area_m2': None,
                     'wall_area_m2': None, 'shelved_wall_area_m2': None, 'scale_method': '', 'confidence': 0},
            'books': [
                {'id': 'b1', 'shelf': '', 'position': None, 'frame_ref': 'frames/aa.jpg', 'frame_refs': [],
                 'status': 'identified', 'title': '<script>x</script>', 'author': 'A', 'edition': '', 'isbn': '',
                 'spine_height_cm': 19.5, 'spine_thickness_cm': 3.2, 'scale_method': 'webxr_depth',
                 'id_confidence': None, 'catalogue_url': '', 'observation_count': 2,
                 'replacement_cost': {'amount': 12.5, 'source': 'Example Books', 'url': 'https://example.test/a',
                                      'retrieved_at': '2026-11-14', 'converted': False},
                 'used_value': {'amount': None, 'source': '', 'url': '', 'retrieved_at': '', 'condition_assumed': ''}},
            ],
            'items': [],
            'totals': {'book_count': 1, 'books_identified': 1, 'books_unidentified': 0, 'shelf_run_m': 0.032,
                       'books_replacement_cost': 12.5, 'books_used_value': 0, 'items_replacement_cost_low': 0,
                       'items_replacement_cost_high': 0, 'excluded_from_totals': 0},
            'review_queue': [{'ref_id': 'room', 'reason': 'Room dimensions were not measured'}],
        }

    def test_report_shows_figures_prices_and_their_sources(self):
        """Print the totals, the priced amount and the source backing it."""
        html = render_report(self.packet)
        self.assertIn('GBP 12.50', html)
        self.assertIn('https://example.test/a', html)
        self.assertIn('Example Books', html)
        self.assertIn('frames/aa.jpg', html)
        self.assertIn('Room dimensions were not measured', html)

    def test_report_marks_unknown_values_rather_than_guessing(self):
        """Show an unpriced figure as having no source instead of printing a number."""
        html = render_report(self.packet)
        self.assertIn('no price source', html)

    def test_report_escapes_model_supplied_text(self):
        """Escape readings that came from a model so they cannot inject markup."""
        html = render_report(self.packet)
        self.assertNotIn('<script>x</script>', html)
        self.assertIn('&lt;script&gt;', html)


class ArchiveTests(unittest.TestCase):
    """Check the deliverable archive carries the packet, the report and its frames."""

    def setUp(self):
        """Save one sweep whose book cites a frame held in the evidence store."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = Path(self.directory.name)
        self.database = root / 'claims.sqlite3'
        self.frames = root / 'frames'
        start = datetime(2026, 11, 14, 9, 0, tzinfo=timezone.utc)
        self.key = EvidenceStore(self.frames).save(b'\xff\xd8\xff-not-a-real-jpeg', 'image/jpeg')
        with Repository(self.database) as repo:
            repo.add_many([
                Sweep(id='sweep1', captured_at=start, device='browser camera'),
                Frame(id='f1', sweep_id='sweep1', captured_at=start, evidence_key=self.key,
                      width_px=640, height_px=480),
                Book(id='b1', sweep_id='sweep1'),
                BookObservation(sweep_id='sweep1', book_id='b1', frame_id='f1',
                                bbox=Box(x_min=.1, y_min=.1, x_max=.5, y_max=.9),
                                association_method='new_provisional', title='THE GOOD PEOPLE'),
            ])

    def test_archive_holds_the_packet_report_and_cited_frames(self):
        """Write both deliverable files and every frame the packet references."""
        archive = packet_archive(self.database, self.frames, 'sweep1', price=False)
        with zipfile.ZipFile(io.BytesIO(archive)) as opened:
            names = set(opened.namelist())
            packet = json.loads(opened.read('claim_packet.json'))
            self.assertIn('report.html', names)
            # Every frame_ref in the packet must resolve to a file inside the archive.
            references = {path for line in packet['books'] for path in line['frame_refs']}
            self.assertTrue(references)
            self.assertLessEqual(references, names)
            self.assertEqual(opened.read(f'frames/{self.key}'), b'\xff\xd8\xff-not-a-real-jpeg')

    def test_missing_sweep_is_reported(self):
        """Raise rather than return an archive for an unknown sweep."""
        with self.assertRaises(KeyError):
            packet_archive(self.database, self.frames, 'absent', price=False)


class PricedPacketTests(unittest.TestCase):
    """Check priced lines, their sources and the totals summed from them."""

    def setUp(self):
        """Save a sweep holding one readable book."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Path(self.directory.name) / 'claims.sqlite3'
        start = datetime(2026, 11, 14, 9, 0, tzinfo=timezone.utc)
        with Repository(self.database) as repo:
            repo.add_many([
                Sweep(id='sweep1', captured_at=start, device='browser camera', country='US', currency='USD'),
                Frame(id='f1', sweep_id='sweep1', captured_at=start, evidence_key=f'{"a" * 64}.jpg',
                      width_px=640, height_px=480),
                Book(id='b1', sweep_id='sweep1'),
                BookObservation(sweep_id='sweep1', book_id='b1', frame_id='f1',
                                bbox=Box(x_min=.1, y_min=.1, x_max=.5, y_max=.9),
                                association_method='new_provisional', title='The Good People',
                                author='Hannah Kent'),
            ])

    @staticmethod
    def priced_client(amount=18.99, currency='USD'):
        """Build an HTTP client answering every price lookup with one figure.

        Args:
            amount: Retail amount returned.
            currency: Currency of that amount.

        Returns:
            A client making no network calls.
        """
        item = {'volumeInfo': {'title': 'The Good People', 'infoLink': 'https://books.example/info'},
                'saleInfo': {'buyLink': 'https://books.example/buy',
                             'retailPrice': {'amount': amount, 'currencyCode': currency}}}
        return httpx.Client(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={'items': [item]})))

    def test_priced_line_carries_its_source_and_joins_the_totals(self):
        """Report the figure with its source and sum it into the packet totals."""
        with self.priced_client() as http:
            packet = build_packet(self.database, 'sweep1', client=http)
        line = packet['books'][0]
        self.assertEqual(line['replacement_cost']['amount'], 18.99)
        self.assertEqual(line['replacement_cost']['source'], 'Google Books')
        self.assertEqual(line['replacement_cost']['url'], 'https://books.example/buy')
        self.assertTrue(line['replacement_cost']['retrieved_at'])
        self.assertEqual(packet['totals']['books_replacement_cost'], 18.99)
        self.assertEqual(packet['totals']['excluded_from_totals'], 0)

    def test_unreachable_source_leaves_the_line_empty_and_flagged(self):
        """Keep an unpriced line out of the totals and name the reason."""
        with httpx.Client(transport=httpx.MockTransport(
                lambda request: httpx.Response(429, json={'error': {}}))) as http:
            packet = build_packet(self.database, 'sweep1', client=http)
        self.assertIsNone(packet['books'][0]['replacement_cost']['amount'])
        self.assertEqual(packet['totals']['books_replacement_cost'], 0)
        self.assertEqual(packet['totals']['excluded_from_totals'], 1)
        self.assertTrue(any('quota' in entry['reason'] for entry in packet['review_queue']))

    def test_appraisal_line_is_never_auto_priced(self):
        """Withhold a figure above the threshold and mark the line for appraisal."""
        with self.priced_client(amount=CONFIG.pricing.appraisal_threshold + 50) as http:
            packet = build_packet(self.database, 'sweep1', client=http)
        line = packet['books'][0]
        self.assertEqual(line['status'], 'needs_appraisal')
        self.assertIsNone(line['replacement_cost']['amount'])
        self.assertEqual(packet['totals']['books_replacement_cost'], 0)

    def test_country_override_prices_the_same_books_elsewhere(self):
        """Price the same sweep in a second market to show locale is a setting."""
        with self.priced_client(amount=9.99, currency='GBP') as http:
            packet = build_packet(self.database, 'sweep1', price=True, country='GB', client=http)
        self.assertEqual(packet['books'][0]['replacement_cost']['amount'], 9.99)
        self.assertEqual(packet['books'][0]['replacement_cost']['currency'], 'GBP')
