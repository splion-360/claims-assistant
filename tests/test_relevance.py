"""Verify relevance abstention, preprocessing, bounded calls, and evidence provenance."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from library_claim.capture.quality import QualityAssessment
from library_claim.capture.relevance.classifier import decide, prepare_image
from library_claim.domain.models import Frame, Quality
from library_claim.storage.live_capture import LiveCaptureStore
from library_claim.storage.repository import Repository
from live_demo.camera_relevance import CameraRelevance
from live_demo.camera_tracking import CameraTracking
from test_tracking import textured_view


def quality(**signals):
    """Construct known measured signals without running an image model.

    Args:
        **signals: Overrides for baseline quality measurements.

    Returns:
        A valid quality assessment for testing model admission.
    """
    values = dict(sharpness=100, dark_fraction=.1, bright_fraction=.1)
    values.update(signals)
    return QualityAssessment(signals=Quality(**values), assessed_width_px=480,
                             assessed_height_px=360, elapsed_ms=1, dark_cutoff=16,
                             bright_cutoff=239, opencv_version=cv2.__version__)


class RelevanceTests(unittest.TestCase):
    """Exercise classifier decisions without downloads or pretrained weights."""

    def test_strong_matches_and_uncertainty(self):
        """Keep both target classes distinct and abstain on weak or ambiguous matches."""
        for label in ('books', 'empty_background'):
            scores = dict(books=.2, empty_background=.2)
            scores[label] = .35
            self.assertEqual(decide(scores, 10).label, label)
        self.assertEqual(decide(dict(books=.31, empty_background=.30), 1).label, 'uncertain')
        self.assertEqual(decide(dict(books=.18, empty_background=.1), 1).reason, 'weak_match')

    def test_invalid_scores_are_not_claimed_as_classifications(self):
        """Reject missing and nonfinite model scores."""
        for scores in ({}, dict(books=float('nan'), empty_background=.2)):
            with self.assertRaises(ValueError):
                decide(scores, 1)

    def test_preprocessing_keeps_edges_and_converts_bgr(self):
        """Retain colored edge objects instead of cropping the camera field of view."""
        image = np.zeros((224, 448, 3), np.uint8)
        image[:, :40] = (0, 0, 255)
        image[:, -40:] = (255, 0, 0)
        result = prepare_image(image)
        self.assertEqual(result.shape, (1, 3, 224, 224))
        self.assertEqual(result.dtype, np.float32)
        self.assertGreater(result[0, 0, 112, 5], 1)
        self.assertGreater(result[0, 2, 112, 218], 1)
        self.assertTrue(np.isfinite(result).all())
        with self.assertRaises(ValueError):
            prepare_image(np.zeros((0, 0, 3), np.uint8))

    def test_quality_failures_make_no_model_calls(self):
        """Reject extreme exposure and insufficient texture before loading CLIP."""
        service = CameraRelevance()
        with patch('live_demo.camera_relevance.shared_classifier') as loader:
            for assessment in (quality(dark_fraction=1), quality(bright_fraction=1), quality(sharpness=0)):
                result = service.assess(np.zeros((32, 32, 3), np.uint8), assessment)
                self.assertEqual(result.status, 'quality_blocked')
                self.assertEqual(result.label, 'uncertain')
                self.assertFalse(result.scores)
            loader.assert_not_called()

    def test_model_failure_is_explicit_and_not_retried_each_frame(self):
        """Mark absent weights unavailable while avoiding a repeated load storm."""
        with patch('live_demo.camera_relevance.shared_classifier', side_effect=FileNotFoundError) as loader:
            service = CameraRelevance()
            for _ in range(3):
                result = service.assess(np.zeros((32, 32, 3), np.uint8), quality())
                self.assertEqual(result.status, 'unavailable')
                self.assertEqual(result.label, 'uncertain')
            self.assertEqual(loader.call_count, 1)

    def test_saved_candidate_is_scored_once_and_logs_its_own_id(self):
        """Persist exact-frame diagnostics and skip repeated tracked views."""
        result = decide(dict(books=.4, empty_background=.1), 12)
        service = Mock(assess=Mock(return_value=result))
        with tempfile.TemporaryDirectory() as directory:
            store = LiveCaptureStore(Path(directory), 'sweep')
            stream = CameraTracking(store, relevance=service)
            _, encoded = cv2.imencode('.jpg', textured_view())
            first = stream.assess(encoded.tobytes(), 'first', 0)
            with self.assertLogs('uvicorn.error.camera_quality', level='DEBUG') as logs:
                stream.log(first, 0)
            self.assertTrue(any('camera_relevance frame=first' in log and '"label":"books"' in log for log in logs.output))
            for i in range(1, 8):
                stream.assess(encoded.tobytes(), f'f{i}', i * .2)
            self.assertEqual(service.assess.call_count, 1)
            self.assertIsNone(stream.latest_relevance)
            with Repository(store.database) as repo:
                frame = repo.get(Frame, 'first')
                self.assertEqual(frame.relevance_metadata, result.model_dump(mode='json'))

    def test_uncertain_tracking_still_checks_quality_without_saving(self):
        """Expose untrackable flat views rather than inventing a negative book label."""
        service = CameraRelevance()
        stream = CameraTracking(relevance=service)
        _, encoded = cv2.imencode('.jpg', np.full((360, 480, 3), 127, np.uint8))
        with patch('live_demo.camera_relevance.shared_classifier') as loader:
            result = stream.assess(encoded.tobytes(), 'wall', 0)
            self.assertEqual(result.status, 'uncertain')
            self.assertEqual(stream.latest_relevance.status, 'quality_blocked')
            self.assertIsNone(stream.saved_path)
            loader.assert_not_called()

    def test_failed_model_does_not_prevent_reference_storage(self):
        """Continue saving verified tracking evidence when model setup is absent."""
        with tempfile.TemporaryDirectory() as directory, \
             patch('live_demo.camera_relevance.shared_classifier', side_effect=FileNotFoundError):
            store = LiveCaptureStore(Path(directory), 'sweep')
            stream = CameraTracking(store, relevance=CameraRelevance())
            _, encoded = cv2.imencode('.jpg', textured_view())
            stream.assess(encoded.tobytes(), 'first', 0)
            with Repository(store.database) as repo:
                self.assertEqual(repo.get(Frame, 'first').relevance_metadata['status'], 'unavailable')
