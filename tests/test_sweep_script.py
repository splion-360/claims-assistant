"""Verify what the agent says at the start and end of a sweep, and how it reads short answers."""
import unittest

from live_demo.sweep_script import (art_question, coverage_prompt, greeting, locale_confirmation, print_answer,
                                    resolve_country, summary)


class LocaleTests(unittest.TestCase):
    """The agent confirms country and currency before pricing."""

    def test_greeting_names_the_default_market_and_asks(self):
        """The opening line proposes the default market and asks for the country."""
        text = greeting('US')
        self.assertIn('the United States', text)
        self.assertIn('US dollars', text)
        self.assertIn('?', text)

    def test_country_names_and_currencies_are_recognised(self):
        """Common ways of naming a configured market resolve to its code."""
        self.assertEqual(resolve_country("I'm in the UK", 'US'), 'GB')
        self.assertEqual(resolve_country('we live in England, pounds please', 'US'), 'GB')
        self.assertEqual(resolve_country('United States', 'GB'), 'US')

    def test_yes_confirms_the_proposed_default(self):
        """Agreeing with the proposal confirms the default market."""
        self.assertEqual(resolve_country('Yes, that is right', 'US'), 'US')
        self.assertEqual(resolve_country('yep', 'US'), 'US')

    def test_unrelated_or_ambiguous_answers_resolve_to_nothing(self):
        """A question, an unknown country or a word inside another word is not an answer."""
        self.assertIsNone(resolve_country('what should I do first?', 'US'))
        self.assertIsNone(resolve_country("I'm in France", 'US'))
        self.assertIsNone(resolve_country('my house is big', 'US'))
        self.assertIsNone(resolve_country('Yes, but I live in France', 'US'))

    def test_confirmation_explains_the_sweep(self):
        """The confirmation names the market and says how to sweep."""
        text = locale_confirmation('GB')
        self.assertIn('British pounds', text)
        self.assertIn('shelf', text)


class ArtTests(unittest.TestCase):
    """The agent asks about framed art and reads the answer."""

    def test_question_describes_the_piece(self):
        """The question names what was seen."""
        self.assertIn('framed portrait', art_question({'description': 'framed portrait'}))

    def test_answers(self):
        """Print wording and original wording are told apart; anything else is no answer."""
        self.assertEqual(print_answer("It's just a print"), 'print')
        self.assertEqual(print_answer('a poster from IKEA'), 'print')
        self.assertEqual(print_answer('No, it is an original oil painting'), 'original')
        self.assertIsNone(print_answer('which shelf next?'))
        self.assertEqual(print_answer('It is not an original'), 'print')


class CoverageTests(unittest.TestCase):
    """The agent asks for the walls the sweep skipped before the user leaves."""

    def test_partial_turn_asks_for_the_rest(self):
        """A sweep that looked across 40% of a turn names the share and asks to continue."""
        text = coverage_prompt({'yaw_coverage': 0.4})
        self.assertIn('40%', text)
        self.assertIn('camera back on', text)

    def test_full_turn_needs_nothing(self):
        """A sweep that covered the room says nothing more."""
        self.assertIsNone(coverage_prompt({'yaw_coverage': 0.95}))

    def test_no_room_asks_for_walls_and_floor(self):
        """Without posed depth the agent still asks for the walls and floor."""
        self.assertIn('walls and the floor', coverage_prompt(None))


class SummaryTests(unittest.TestCase):
    """The spoken summary is built from the packet's code-computed totals."""

    PACKET = {'sweep': {'currency': 'USD'},
              'totals': {'book_count': 12, 'books_identified': 9, 'books_unidentified': 3,
                         'books_replacement_cost': 143.5, 'books_used_value': 61.0,
                         'items_replacement_cost_low': 400.0, 'items_replacement_cost_high': 900.0,
                         'excluded_from_totals': 2},
              'items': [{}] * 6, 'room': {'floor_area_m2': 12.4, 'wall_area_m2': 31.0},
              'review_queue': [{}] * 5}

    def test_summary_reads_the_totals(self):
        """Counts, money and room figures come from the packet, rounded for speech."""
        text = summary(self.PACKET)
        for fragment in ('12 books', '9 identified', '6 other items', '143.50', '61.00', '400', '900',
                         '12.4 square metres', '5 lines need review'):
            self.assertIn(fragment, text)

    def test_summary_without_room_or_prices_says_so(self):
        """Missing figures are stated as missing, never as zero."""
        packet = {**self.PACKET, 'room': {'floor_area_m2': None, 'wall_area_m2': None},
                  'totals': {**self.PACKET['totals'], 'books_replacement_cost': 0, 'books_used_value': 0,
                             'items_replacement_cost_low': 0, 'items_replacement_cost_high': 0}}
        text = summary(packet)
        self.assertIn('room was not measured', text)
        self.assertNotIn('0.00', text)


if __name__ == '__main__':
    unittest.main()
