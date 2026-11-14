"""Verify centralized experiment settings and their runtime consumers."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from pydantic import ValidationError
import yaml

from library_claim.config import CONFIG, ROOT, load_config
from library_claim.capture.relevance.classifier import RelevanceAssessment, decide


class ConfigTests(unittest.TestCase):
    """Check strict config loading and behavior with alternate experiment values."""

    def test_invalid_and_unknown_settings_fail_early(self):
        """Reject typos, malformed prompts, and unsafe numeric ranges."""
        raw = yaml.safe_load((ROOT / 'config.yml').read_text())
        cases = [
            ('relevance', 'min_similarity', 2),
            ('relevance', 'min_similiarity', .4),
            ('relevance', 'prompts', {'books': []}),
            ('relevance', 'prompts', {'books': ['book'], 'empty_background': ['wall'], 'other_objects': ['mug']}),
            ('quality', 'dark_cutoff', 250),
            ('tracking', 'min_matches', 3),
            ('tracking', 'max_gap_s', float('nan')),
            ('tracking', 'min_projected_area_ratio', 8),
            ('capture', 'quality_interval_s', 0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'config.yml'
            for section, key, value in cases:
                with self.subTest(section=section, key=key):
                    data = copy.deepcopy(raw)
                    data[section][key] = value
                    path.write_text(yaml.safe_dump(data))
                    with self.assertRaises(ValidationError):
                        load_config(path)

    def test_book_only_target_abstains_below_threshold(self):
        """Keep low similarity uncertain without a competing background target."""
        self.assertEqual(set(CONFIG.relevance.prompts.model_dump()), {'books', 'empty_background'})
        self.assertEqual(decide({'books': .1}, 1).label, 'uncertain')
        with self.assertRaises(ValidationError):
            RelevanceAssessment(label='empty_background', reason='retired')

    def test_experiment_values_reach_processing_defaults(self):
        """Load alternate YAML before consumers and verify settings affect behavior."""
        data = CONFIG.model_dump(mode='json')
        data['relevance']['min_similarity'] = .5
        data['relevance']['prompts']['books'] = ['a book cover']
        data['quality']['max_dimension'] = 320
        data['quality']['dark_cutoff'] = 24
        data['tracking']['capacity'] = 5
        data['tracking']['reanchor_every'] = 7
        data['selection']['new_coverage_fraction'] = .4
        data['scheduler']['pending_limit'] = 3
        data['inference']['model'] = 'experiment-text'
        data['speech']['stt_model'] = 'experiment-speech'
        data['capture']['browser']['frame_interval_ms'] = 350
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'experiment.yml'
            path.write_text(yaml.safe_dump(data))
            # A fresh process reproduces startup imports without mutating the live config.
            code = '''
import inspect, json, sys
from pathlib import Path
import library_claim.config as config
config.CONFIG = config.load_config(Path(sys.argv[1]))
from library_claim.capture.relevance.classifier import decide, PROMPTS
from library_claim.capture.quality import QualityAssessor
from library_claim.capture.tracking.tracker import FrameTracker
from library_claim.capture.selection import SETTINGS as selection
from library_claim.orchestration.scheduler import Scheduler
import provider, speech
from live_demo.server import health
assessor = QualityAssessor()
tracker = FrameTracker()
print(json.dumps(dict(label=decide(dict(books=.4),1).label,
 prompts=PROMPTS, dimension=assessor.max_dimension, dark=assessor.dark_cutoff,
 capacity=tracker.keyframes.capacity, reanchor=tracker.reanchor_every,
 coverage=selection.new_coverage_fraction,
 pending=inspect.signature(Scheduler).parameters['pending_limit'].default,
 model=provider.MODEL, stt=speech.CONFIG.speech.stt_model,
 browser=health()['camera'])))
'''
            result = subprocess.run([sys.executable, '-c', code, str(path)], cwd=ROOT,
                                    capture_output=True, text=True, check=True, timeout=20)
        values = json.loads(result.stdout)
        self.assertEqual(values['label'], 'uncertain')
        self.assertEqual(values['prompts']['books'], ['a book cover'])
        self.assertEqual((values['dimension'], values['dark']), (320, 24))
        self.assertEqual((values['capacity'], values['reanchor']), (5, 7))
        self.assertEqual(values['coverage'], .4)
        self.assertEqual(values['pending'], 3)
        self.assertEqual(values['model'], 'experiment-text')
        self.assertEqual(values['stt'], 'experiment-speech')
        self.assertEqual(values['browser']['frame_interval_ms'], 350)
        self.assertEqual(set(values['browser']), set(CONFIG.capture.browser.model_dump()))

    def test_default_config_is_independent_of_working_directory(self):
        """Resolve the project YAML when launched outside the repository directory."""
        code = f'import sys; sys.path.insert(0, {str(ROOT)!r}); from library_claim.config import load_config; print(load_config().relevance.model_id)'
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, '-c', code], cwd=directory,
                                    capture_output=True, text=True, check=True, timeout=20)
        self.assertEqual(result.stdout.strip(), CONFIG.relevance.model_id)

    def test_drex_thresholds_loaded(self):
        """Expose Drex decision thresholds from config.yml."""
        self.assertEqual(CONFIG.drex.same_book_min, 0.8)
        self.assertEqual(CONFIG.drex.different_book_max, 0.2)
        self.assertEqual(CONFIG.measurement.depth_inset, 0.15)
