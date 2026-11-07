"""Verify camera summaries throttle output without losing interval diagnostics."""
import unittest
from unittest.mock import patch

from library_claim.capture.relevance.classifier import decide
from library_claim.capture.tracking.tracker import TrackingResult
from library_claim.config import CONFIG
from live_demo.camera_logging import CameraSummary
from test_relevance import quality


class CameraLoggingTests(unittest.TestCase):
    """Check table readability, retained results, and summary cadence."""

    def test_summary_counts_saves_and_dates_skipped_classification(self):
        """Aggregate all saves while keeping a previous classification visibly dated."""
        summary = CameraSummary()
        tracking = TrackingResult(frame_id='private-frame', status='tracked', elapsed_ms=4)
        relevance = decide({'books': .233}, 35)
        interval = CONFIG.capture.summary_log_interval_s
        with self.assertLogs('uvicorn.error.camera_quality', level='INFO') as logs:
            for time in (0, interval / 4, interval / 2, interval):
                summary.log(tracking, time, quality=quality(brisque=24.2, sharpness=207.5),
                            quality_at=time, relevance=relevance if time == 0 else None, saved=True)
        self.assertEqual(len(logs.records), 2)
        final = logs.output[-1]
        self.assertIn('BRISQUE 24.2', final)
        self.assertIn('Sharpness 207.5', final)
        self.assertIn('uncertain', final)
        self.assertIn(f'{interval:.1f}s ago', final)
        self.assertIn('Books 0.233', final)
        self.assertIn(f'Threshold {CONFIG.relevance.min_similarity:.3f}', final)
        self.assertNotIn('Background', final)
        self.assertRegex(final, r'Evidence saved\s+3')
        self.assertNotIn('private-frame', final)
        self.assertNotIn('Dark pixels', final)
        self.assertEqual(summary.saved_count, 0)

    def test_new_quality_failure_replaces_old_classification(self):
        """Do not retain old similarity scores after a blocked assessment arrives."""
        summary = CameraSummary()
        tracking = TrackingResult(frame_id='frame', status='uncertain', elapsed_ms=4)
        first = decide({'books': .4}, 10)
        blocked = first.model_copy(update={'label': 'uncertain', 'status': 'quality_blocked',
                                          'scores': {}, 'reason': 'extreme_exposure'})
        with self.assertLogs('uvicorn.error.camera_quality', level='INFO') as logs:
            summary.log(tracking, 0, quality=None, quality_at=0, relevance=first, saved=False)
            summary.log(tracking, CONFIG.capture.summary_log_interval_s, quality=None,
                        quality_at=0, relevance=blocked, saved=False)
        self.assertIn('quality blocked', logs.output[-1])
        self.assertNotIn('Books 0.400', logs.output[-1])

    def test_summary_interval_comes_from_config(self):
        """Honor an experimental interval without changing processing settings."""
        summary = CameraSummary()
        tracking = TrackingResult(frame_id='frame', status='reference_created', elapsed_ms=4)
        changed = CONFIG.model_copy(update={'capture': CONFIG.capture.model_copy(update={'summary_log_interval_s': 10})})
        with patch('live_demo.camera_logging.CONFIG', changed), \
             self.assertLogs('uvicorn.error.camera_quality', level='INFO') as logs:
            for time in (0, 5, 9, 10):
                summary.log(tracking, time, quality=None, quality_at=0, relevance=None, saved=False)
        self.assertEqual(len(logs.records), 2)
        self.assertIn('Not assessed', logs.output[-1])
