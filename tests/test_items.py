"""Verify non-book detections are recorded, associated and valued without invention."""
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from library_claim.domain.models import Box, Frame, FrameAssociation, Item, ItemObservation, Sweep
from library_claim.items.models import APPRAISAL_CATEGORIES, ItemDetection, ItemFrameAnalysis
from library_claim.items.pricing import price_item
from library_claim.items.storage import inventory, save_items
from library_claim.storage.evidence import EvidenceStore
from library_claim.storage.repository import Repository
from test_fetch import quiet_sources


def detection(**fields):
    """Build one non-book detection.

    Args:
        **fields: Overrides such as bbox, category or brand_model.

    Returns:
        An item detection.
    """
    return ItemDetection(**{'bbox': Box(x_min=.1, y_min=.1, x_max=.4, y_max=.9),
                            'category': 'lighting', 'description': 'brass floor lamp', **fields})


class ItemStorageTests(unittest.TestCase):
    """Exercise detection storage and cross-frame association on real SQLite."""

    def setUp(self):
        """Save a sweep holding two frames and an encodable image."""
        quiet_sources(self)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Path(self.directory.name) / 'claims.sqlite3'
        self.start = datetime(2026, 11, 14, 9, 0, tzinfo=timezone.utc)
        image = np.random.default_rng(5).integers(40, 210, (480, 640, 3)).astype(np.uint8)
        self.jpeg = cv2.imencode('.jpg', image)[1].tobytes()
        self.compared = []
        # Candidate crops are cut from the stored frames, so the evidence must exist,
        # and the two frames must be different images or they count as one view.
        store = EvidenceStore(self.database.parent / 'frames')
        other = np.random.default_rng(9).integers(40, 210, (480, 640, 3)).astype(np.uint8)
        keys = [store.save(self.jpeg, 'image/jpeg'),
                store.save(cv2.imencode('.jpg', other)[1].tobytes(), 'image/jpeg')]
        with Repository(self.database) as repo:
            repo.add_many([
                Sweep(id='s1', captured_at=self.start),
                Frame(id='f1', sweep_id='s1', captured_at=self.start,
                      evidence_key=keys[0], width_px=640, height_px=480),
                Frame(id='f2', sweep_id='s1', captured_at=self.start,
                      evidence_key=keys[1], width_px=640, height_px=480),
            ])

    def save(self, frame_id, detections, complete=True, same=None):
        """Save one frame's detections.

        Args:
            frame_id: Frame the detections came from.
            detections: Item detections.
            complete: Whether the frame was fully enumerated.
            same: Same-object probabilities per candidate; no match by default.

        Returns:
            The save result including the current inventory.
        """
        def decide(crop, category, description, candidates):
            """Answer the same-object question without a network call.

            Args:
                crop: JPEG of the new detection.
                category: Its category.
                description: Its description.
                candidates: Saved objects being compared.

            Returns:
                One probability per candidate.
            """
            self.compared.append([item.item_id for item in candidates])
            return list(same) if same else [0.0] * len(candidates)

        return save_items(self.database, 's1', frame_id,
                          ItemFrameAnalysis(items=detections, complete=complete), self.jpeg,
                          'test-model', decide_same=decide)

    def link_frames(self):
        """Record a verified identity mapping from the second frame to the first."""
        with Repository(self.database) as repo:
            repo.add(FrameAssociation(sweep_id='s1', frame_id='f2', reference_frame_id='f1',
                                      matrix=((1, 0, 0), (0, 1, 0), (0, 0, 1)),
                                      method='sift_ransac', matched_points=40, reprojection_error_px=0.5))

    def test_each_detection_becomes_one_object(self):
        """Record one object per distinct detection, with its category and description."""
        result = self.save('f1', [detection(),
                                  detection(bbox=Box(x_min=.5, y_min=.1, x_max=.9, y_max=.9),
                                            category='shelving', description='three-shelf bookcase')])
        self.assertEqual(result['detected_count'], 2)
        self.assertEqual({entry['category'] for entry in result['inventory']}, {'lighting', 'shelving'})

    def test_retrying_a_frame_adds_nothing(self):
        """Return the saved result rather than recording a frame's objects twice."""
        self.save('f1', [detection()])
        self.save('f1', [detection()])
        with Repository(self.database) as repo:
            self.assertEqual(len(repo.list_for_sweep(Item, 's1')), 1)

    def test_one_object_seen_twice_stays_one(self):
        """Merge a second sighting of the same object through verified geometry."""
        self.save('f1', [detection()])
        self.link_frames()
        result = self.save('f2', [detection()])
        self.assertEqual(len(result['inventory']), 1)
        self.assertEqual(result['inventory'][0]['observation_count'], 2)

    def test_a_different_category_is_a_different_object(self):
        """Keep two objects apart when the category disagrees, however they overlap."""
        self.save('f1', [detection()])
        self.link_frames()
        result = self.save('f2', [detection(category='decor', description='vase')])
        self.assertEqual(len(result['inventory']), 2)

    def test_objects_elsewhere_in_the_frame_stay_separate(self):
        """Keep two objects apart when their boxes do not overlap."""
        self.save('f1', [detection()])
        self.link_frames()
        result = self.save('f2', [detection(bbox=Box(x_min=.6, y_min=.1, x_max=.9, y_max=.9))])
        self.assertEqual(len(result['inventory']), 2)

    def test_unlinked_frames_do_not_merge_objects(self):
        """Refuse to merge sightings whose frames share no verified basis."""
        self.save('f1', [detection()])
        result = self.save('f2', [detection()])
        self.assertEqual(len(result['inventory']), 2)
        self.assertTrue(any('may be a duplicate' in reason
                            for entry in result['inventory'] for reason in entry['review_reasons']))

    def test_one_object_cannot_absorb_two_detections_in_a_frame(self):
        """Keep co-visible objects distinct rather than collapsing them into one."""
        self.save('f1', [detection()])
        self.link_frames()
        result = self.save('f2', [detection(), detection(bbox=Box(x_min=.11, y_min=.11, x_max=.41, y_max=.91))])
        self.assertEqual(len(result['inventory']), 2)

    def test_framed_art_is_marked_for_appraisal(self):
        """Mark art and mirrors for a person rather than valuing them automatically."""
        result = self.save('f1', [detection(category='art', description='framed portrait'),
                                  detection(bbox=Box(x_min=.5, y_min=.1, x_max=.9, y_max=.5),
                                            category='mirror', description='round wall mirror')])
        self.assertTrue(all(entry['needs_appraisal'] for entry in result['inventory']))
        self.assertEqual(set(APPRAISAL_CATEGORIES), {'art', 'mirror'})

    def test_ordinary_contents_are_not_marked_for_appraisal(self):
        """Leave furniture and lighting open to automatic valuation."""
        result = self.save('f1', [detection()])
        self.assertFalse(result['inventory'][0]['needs_appraisal'])

    def test_readings_are_kept_and_never_invented(self):
        """Carry the material and brand that were read, and leave the rest empty."""
        result = self.save('f1', [detection(material='brass', brand_model='Anglepoise 1227')])
        entry = result['inventory'][0]
        self.assertEqual((entry['material'], entry['brand_model']), ('brass', 'Anglepoise 1227'))
        second = self.save('f2', [detection(bbox=Box(x_min=.5, y_min=.5, x_max=.8, y_max=.9),
                                            category='seating', description='armchair')])
        armchair = next(item for item in second['inventory'] if item['category'] == 'seating')
        self.assertIsNone(armchair['brand_model'])
        self.assertIsNone(armchair['material'])

    def test_an_empty_frame_records_nothing(self):
        """Record no objects for a frame showing none."""
        result = self.save('f1', [], complete=False)
        self.assertEqual(result['detected_count'], 0)
        self.assertEqual(result['inventory'], [])
        self.assertFalse(result['complete'])


class ItemPricingTests(unittest.TestCase):
    """Exercise what a non-book line may claim about its value."""

    def setUp(self):
        """Close source breakers and skip backoff sleeps."""
        quiet_sources(self)

    @staticmethod
    def item(**fields):
        """Build an item inventory entry.

        Args:
            **fields: Overrides such as category or brand_model.

        Returns:
            An inventory entry.
        """
        return {'id': 'i1', 'category': 'lighting', 'description': 'brass floor lamp',
                'brand_model': None, 'stated_print': False, **fields}

    def test_art_is_withheld_from_automatic_valuation(self):
        """Refuse to value framed art, naming the appraisal reason."""
        valuation = price_item(self.item(category='art', description='framed portrait'), 'US')
        self.assertTrue(valuation.needs_appraisal)
        self.assertIsNone(valuation.replacement)

    def test_a_stated_print_is_no_longer_an_appraisal(self):
        """Release a framed piece from appraisal once the policyholder says it is a print."""
        valuation = price_item(self.item(category='art', stated_print=True), 'US')
        self.assertFalse(valuation.needs_appraisal)

    def test_no_listing_source_leaves_the_figure_empty_with_a_reason(self):
        """Say that the listing source is not configured instead of writing a price."""
        valuation = price_item(self.item(), 'US')
        self.assertIsNone(valuation.replacement)
        self.assertTrue(any('not configured' in reason for reason in valuation.reasons))

    def test_an_unidentified_object_says_it_needs_a_range(self):
        """Record that an unbranded object can only be given a sourced range."""
        valuation = price_item(self.item(), 'US')
        self.assertTrue(any('range' in reason for reason in valuation.reasons))

    def test_currency_follows_the_market(self):
        """Report the market's own currency for the line."""
        self.assertEqual(price_item(self.item(), 'GB').currency, 'GBP')
        self.assertEqual(price_item(self.item(), 'US').currency, 'USD')


class ItemAppearanceTests(unittest.TestCase):
    """Check appearance settles what geometry cannot, without guessing."""

    def setUp(self):
        """Close source breakers and skip backoff sleeps."""
        quiet_sources(self)

    setUp = ItemStorageTests.setUp
    save = ItemStorageTests.save
    link_frames = ItemStorageTests.link_frames

    def test_an_object_seen_from_another_angle_stays_one_object(self):
        """Merge a sighting whose box does not overlap when it looks like the same object."""
        self.save('f1', [detection()])
        # No frame association and a box elsewhere: only appearance can decide.
        result = self.save('f2', [detection(bbox=Box(x_min=.6, y_min=.1, x_max=.9, y_max=.9))], same=[0.95])
        self.assertEqual(len(result['inventory']), 1)
        self.assertEqual(result['inventory'][0]['observation_count'], 2)

    def test_a_different_looking_object_stays_separate(self):
        """Keep a sighting separate when appearance says it is a different object."""
        self.save('f1', [detection()])
        result = self.save('f2', [detection(bbox=Box(x_min=.6, y_min=.1, x_max=.9, y_max=.9))], same=[0.05])
        self.assertEqual(len(result['inventory']), 2)

    def test_competing_confident_answers_leave_it_unmatched(self):
        """Refuse to pick when two saved objects both look like the sighting."""
        self.save('f1', [detection(),
                         detection(bbox=Box(x_min=.5, y_min=.1, x_max=.8, y_max=.9))])
        result = self.save('f2', [detection(bbox=Box(x_min=.82, y_min=.1, x_max=.95, y_max=.9))],
                           same=[0.95, 0.95])
        self.assertEqual(len(result['inventory']), 3)

    def test_only_objects_of_the_same_category_are_compared(self):
        """Compare a sighting against its own category, never across categories."""
        self.save('f1', [detection(category='seating', description='armchair')])
        self.compared.clear()
        self.save('f2', [detection(bbox=Box(x_min=.6, y_min=.1, x_max=.9, y_max=.9))], same=[0.95])
        # With nothing of that category saved, no decision is asked for at all.
        self.assertEqual(self.compared, [])

    def test_a_failed_decision_records_a_separate_object(self):
        """Keep the sighting rather than lose it when the decision cannot be made."""
        from library_claim.decisions.drex import DrexError

        def failing(crop, category, description, candidates):
            """Raise as an unreachable decision service would.

            Args:
                crop: JPEG of the new detection.
                category: Its category.
                description: Its description.
                candidates: Saved objects being compared.

            Raises:
                DrexError: Always.
            """
            raise DrexError('unavailable')

        self.save('f1', [detection()])
        result = save_items(self.database, 's1', 'f2',
                            ItemFrameAnalysis(items=[detection(bbox=Box(x_min=.6, y_min=.1, x_max=.9, y_max=.9))],
                                              complete=True),
                            self.jpeg, 'test-model', decide_same=failing)
        self.assertEqual(len(result['inventory']), 2)
