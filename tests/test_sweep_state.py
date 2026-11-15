"""Verify the sweep's locale, end time and packet time are saved and reported."""
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from library_claim.domain.models import Frame, Item, Sweep
from library_claim.packet import build_packet
from library_claim.storage.repository import Repository
from library_claim.storage.sweep_state import (mark_packet_ready, mark_sweep_ended, set_stated_print,
                                               set_sweep_locale)

START = datetime(2026, 11, 14, 9, tzinfo=timezone.utc)


class SweepStateTests(unittest.TestCase):
    """Small writes the live agent makes to the sweep record."""

    def setUp(self):
        """Open an empty capture database."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.database = Path(directory.name) / 'claims.sqlite3'

    def sweep(self):
        """Read the sweep record.

        Returns:
            The saved sweep.
        """
        with Repository(self.database) as repo:
            return repo.get(Sweep, 's')

    def test_locale_creates_the_sweep_when_no_frame_was_saved_yet(self):
        """The country can be confirmed before the camera starts."""
        set_sweep_locale(self.database, 's', 'GB')
        self.assertEqual((self.sweep().country, self.sweep().currency), ('GB', 'GBP'))

    def test_locale_updates_an_existing_sweep(self):
        """A later answer replaces the earlier market."""
        with Repository(self.database) as repo:
            repo.add(Sweep(id='s', captured_at=START, device='phone'))
        set_sweep_locale(self.database, 's', 'US')
        self.assertEqual((self.sweep().country, self.sweep().device), ('US', 'phone'))

    def test_time_to_packet_is_reported_and_a_slow_packet_is_flagged(self):
        """The packet states the time from sweep end to packet, and flags one over five minutes."""
        with Repository(self.database) as repo:
            repo.add_many([Sweep(id='s', captured_at=START, device='phone', country='US', currency='USD'),
                           Frame(id='f', sweep_id='s', captured_at=START, evidence_key=f'{"a" * 64}.jpg',
                                 width_px=10, height_px=10)])
        mark_sweep_ended(self.database, 's', START + timedelta(minutes=3))
        mark_packet_ready(self.database, 's', START + timedelta(minutes=3, seconds=40))
        packet = build_packet(self.database, 's', price=False)
        self.assertEqual(packet['sweep']['time_to_packet_s'], 40.0)
        self.assertFalse(any('minute' in entry['reason'] for entry in packet['review_queue']))
        mark_packet_ready(self.database, 's', START + timedelta(minutes=9))
        packet = build_packet(self.database, 's', price=False)
        self.assertEqual(packet['sweep']['time_to_packet_s'], 360.0)
        self.assertTrue(any('5-minute' in entry['reason'] for entry in packet['review_queue']))

    def test_unended_sweep_has_no_time_to_packet(self):
        """Without a recorded end the field stays empty."""
        with Repository(self.database) as repo:
            repo.add(Sweep(id='s', captured_at=START, device='phone'))
        self.assertIsNone(build_packet(self.database, 's', price=False)['sweep']['time_to_packet_s'])

    def test_stated_print_is_recorded(self):
        """A 'print' answer releases the piece from appraisal; 'original' keeps it."""
        with Repository(self.database) as repo:
            repo.add_many([Sweep(id='s', captured_at=START), Item(id='i', sweep_id='s', category='art',
                                                                  needs_appraisal=True)])
        set_stated_print(self.database, 'i', True)
        with Repository(self.database) as repo:
            self.assertTrue(repo.get(Item, 'i').stated_print)


if __name__ == '__main__':
    unittest.main()
