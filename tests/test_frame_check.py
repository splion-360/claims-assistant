"""Verify the Drex frame check mapping, admission threshold and persisted record."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from library_claim.decisions.frame_check import FrameCheck, check_frame, is_book_candidate, save_frame_check
from library_claim.domain.models import ProcessingJob
from library_claim.storage.live_capture import LiveCaptureStore
from library_claim.storage.repository import Repository
from live_demo.camera_tracking import CameraTracking
from test_live_persistence import jpeg
from test_tracking import textured_view

RESPONSE = {
    'model': 'drex-max-v1.1',
    'answers': {
        'book_visible': {'type': 'noul', 'noul': 0.002, 'confidence': 0.998},
        'spine_legible': {'type': 'noul', 'noul': 0.004, 'confidence': 0.996},
        'issue': {'type': 'choice', 'choice': 'no_books', 'confidence': 0.99,
                  'probabilities': {'none': 0.001, 'blur': 0.005, 'no_books': 0.99}},
    },
    'usage': {'input_tokens': 803},
    'request_id': 'req_frame',
}


def check(book_visible: float) -> FrameCheck:
    """Build a frame check with a chosen book probability.

    Args:
        book_visible: Probability that a book is visible.

    Returns:
        A frame check with fixed remaining fields.
    """
    return FrameCheck(book_visible=book_visible, spine_legible=0.5, issue='none', issue_probabilities={},
                      model='drex-max-v1.1', request_id=None, latency_ms=1, input_tokens=1)


class FrameCheckTests(unittest.TestCase):
    """Exercise D1 parsing, admission and storage."""

    def test_check_frame_maps_answers(self):
        """Map the three D1 answers and send exactly the frame image."""
        def handler(request):
            body = json.loads(request.content)
            self.assertEqual(len(body['images']), 1)
            self.assertEqual(set(body['questions']), {'book_visible', 'spine_legible', 'issue'})
            self.assertIn('no_books', body['questions']['issue']['criteria'])
            return httpx.Response(200, json=RESPONSE)

        with patch.dict(os.environ, {'DREX_API_KEY': 'k'}):
            result = check_frame(b'\xff\xd8\xff', client=httpx.Client(transport=httpx.MockTransport(handler)))
        self.assertEqual(result.issue, 'no_books')
        self.assertEqual(result.book_visible, 0.002)
        self.assertEqual(result.spine_legible, 0.004)
        self.assertEqual(result.request_id, 'req_frame')

    def test_candidate_threshold(self):
        """Admit at the configured 0.5 boundary and never without a check."""
        self.assertTrue(is_book_candidate(check(0.5)))
        self.assertFalse(is_book_candidate(check(0.49)))
        self.assertFalse(is_book_candidate(None))

    def test_save_frame_check_is_idempotent(self):
        """Store one frame_check job per saved frame."""
        with tempfile.TemporaryDirectory() as directory:
            store = LiveCaptureStore(Path(directory), 'sweep1')
            CameraTracking(store).assess(jpeg(textured_view()), 'f0', 0)
            save_frame_check(store.database, 'sweep1', 'f0', check(0.9))
            save_frame_check(store.database, 'sweep1', 'f0', check(0.9))
            with Repository(store.database) as repo:
                jobs = [job for job in repo.list_for_sweep(ProcessingJob, 'sweep1') if job.task == 'frame_check']
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0].result['book_visible'], 0.9)

    def test_blurry_unreadable_frame_is_not_admitted(self):
        """The real frame from the sweep: book visible, title unreadable, main issue blur."""
        real = check(0.935).model_copy(update={'spine_legible': 0.022, 'issue': 'blur'})
        self.assertFalse(is_book_candidate(real))
        for defect in ('glare', 'dark'):
            self.assertFalse(is_book_candidate(real.model_copy(update={'issue': defect})))

    def test_clear_but_unreadable_spine_is_still_admitted(self):
        """A sharp frame of a spine with no readable text is logged as an unidentified book."""
        clear = check(0.9).model_copy(update={'spine_legible': 0.1, 'issue': 'none'})
        self.assertTrue(is_book_candidate(clear))

    def test_legible_title_overrides_a_minor_defect(self):
        """A readable title is admitted even if Drex names slight blur as the main issue."""
        legible = check(0.9).model_copy(update={'spine_legible': 0.7, 'issue': 'blur'})
        self.assertTrue(is_book_candidate(legible))
