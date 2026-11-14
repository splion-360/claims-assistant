"""Verify spoken book announcements: wording, one per title, and skipped books."""
import unittest

from live_demo.announcements import announcement


class AnnouncementTests(unittest.TestCase):
    """Exercise announcement wording and de-duplication."""

    def test_single_new_book(self):
        """Announce one new title in a natural sentence, in readable case."""
        text, keys = announcement([{'title': 'DID I SAY YOU COULD GO'}], set())
        self.assertEqual(text, 'I found a book called Did I Say You Could Go.')
        self.assertEqual(keys, {'did i say you could go'})

    def test_two_and_many_books(self):
        """Group several new titles into one sentence."""
        two, _ = announcement([{'title': 'Dune'}, {'title': 'Emma'}], set())
        self.assertEqual(two, 'I found two books: Dune and Emma.')
        many, _ = announcement([{'title': 'Dune'}, {'title': 'Emma'}, {'title': 'Ulysses'}], set())
        self.assertEqual(many, 'I found 3 books: Dune, Emma and Ulysses.')

    def test_already_announced_untitled_and_excluded_books_stay_quiet(self):
        """Repeat sightings, unreadable spines and excluded books are not announced."""
        books = [{'title': 'Did I Say You Could Go'}, {'title': 'DID I SAY YOU COULD GO'},
                 {'title': None}, {'title': 'Emma', 'excluded': True}]
        self.assertEqual(announcement(books, {'did i say you could go'}), (None, set()))

    def test_duplicate_records_of_one_title_are_announced_once(self):
        """Two unmerged records with the same title produce one mention."""
        text, _ = announcement([{'title': 'Dune'}, {'title': 'DUNE'}], set())
        self.assertEqual(text, 'I found a book called Dune.')
