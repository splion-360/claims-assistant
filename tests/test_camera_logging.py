"""Verify camera summaries throttle output without losing interval diagnostics."""
import unittest
from unittest.mock import patch

import cv2

from library_claim.capture.quality import QualityAssessment
from library_claim.capture.tracking.tracker import TrackingResult
from library_claim.config import CONFIG
from library_claim.domain.models import Quality
from live_demo.camera_logging import CameraSummary
from test_frame_check import check


def quality(**signals):
    """Construct known measured signals without running an image model.

    Args:
        **signals: Overrides for baseline quality measurements.

    Returns:
        A valid quality assessment for testing summaries.
    """
    values = dict(sharpness=100, dark_fraction=.1, bright_fraction=.1)
    values.update(signals)
    return QualityAssessment(signals=Quality(**values), assessed_width_px=480,
                             assessed_height_px=360, elapsed_ms=1, dark_cutoff=16,
                             bright_cutoff=239, opencv_version=cv2.__version__)


class CameraLoggingTests(unittest.TestCase):
    """Check table readability, dated Drex checks, and summary cadence."""

    def test_summary_counts_saves_and_dates_last_check(self):
        """Aggregate all saves while showing how old the last Drex check is."""
        summary = CameraSummary()
        tracking = TrackingResult(frame_id='private-frame', status='tracked', elapsed_ms=4)
        decision = check(0.91)
        interval = CONFIG.capture.summary_log_interval_s
        with self.assertLogs('uvicorn.error.camera_quality', level='INFO') as logs:
            for time in (0, interval / 4, interval / 2, interval):
                summary.log(tracking, time, quality=quality(brisque=24.2, sharpness=207.5),
                            quality_at=time, check=decision, check_at=0, saved=True)
        self.assertEqual(len(logs.records), 2)
        final = logs.output[-1]
        self.assertIn('BRISQUE 24.2', final)
        self.assertIn('Sharpness 207.5', final)
        self.assertIn('Book 0.91', final)
        self.assertIn(f'{interval:.1f}s ago', final)
        self.assertRegex(final, r'Evidence saved\s+3')
        self.assertNotIn('private-frame', final)
        self.assertEqual(summary.saved_count, 0)

    def test_summary_interval_comes_from_config(self):
        """Honor an experimental interval without changing processing settings."""
        summary = CameraSummary()
        tracking = TrackingResult(frame_id='frame', status='reference_created', elapsed_ms=4)
        changed = CONFIG.model_copy(update={'capture': CONFIG.capture.model_copy(update={'summary_log_interval_s': 10})})
        with patch('live_demo.camera_logging.CONFIG', changed), \
             self.assertLogs('uvicorn.error.camera_quality', level='INFO') as logs:
            for time in (0, 5, 9, 10):
                summary.log(tracking, time, quality=None, quality_at=0, saved=False)
        self.assertEqual(len(logs.records), 2)
        self.assertIn('Not checked', logs.output[-1])
