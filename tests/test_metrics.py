"""Verify per-sweep latency and usage are collected from stage events and costed from unit prices."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from library_claim import telemetry
from library_claim.metrics import SweepMeter, load_metrics


class MeterTests(unittest.TestCase):
    """Stage timings and usage counters roll up into the packet's metrics block."""

    def test_latency_per_stage(self):
        """Finished stages are summarised by count, median, 95th percentile and maximum."""
        meter = SweepMeter('s')
        for ms in (100, 200, 300, 400, 1000):
            meter.record({'stage': 'd1', 'phase': 'done', 'ms': ms})
        meter.record({'stage': 'd1', 'phase': 'start', 'ms': None})
        meter.record({'stage': 'd1', 'phase': 'error', 'ms': 50})
        stats = meter.summary()['latency_ms']['d1']
        self.assertEqual((stats['count'], stats['errors'], stats['p50'], stats['max']), (6, 1, 250, 1000))

    def test_usage_is_counted_from_telemetry(self):
        """Usage events emitted anywhere under the listener are added up."""
        meter = SweepMeter('s')
        with telemetry.listening(meter.record):
            telemetry.usage('drex', calls=1, input_tokens=800)
            telemetry.usage('drex', calls=1, input_tokens=200)
            telemetry.usage('nimble', calls=1)
        usage = meter.summary()['usage']
        self.assertEqual(usage['drex'], {'calls': 2, 'input_tokens': 1000})
        self.assertEqual(usage['nimble'], {'calls': 1})

    def test_cost_uses_only_configured_unit_prices(self):
        """A source with a unit price is costed; one without is listed as not estimated."""
        meter = SweepMeter('s')
        meter.record({'stage': 'usage', 'phase': 'done', 'usage': {'source': 'nimble', 'calls': 4}})
        meter.record({'stage': 'usage', 'phase': 'done', 'usage': {'source': 'drex', 'calls': 2}})
        prices = {'nimble': {'calls': 0.001}}
        with patch('library_claim.metrics.unit_prices', return_value=prices):
            cost = meter.summary()['cost_usd']
        self.assertEqual(cost['by_source'], {'nimble': 0.004})
        self.assertEqual(cost['not_estimated'], ['drex'])
        self.assertAlmostEqual(cost['estimated_total'], 0.004)

    def test_saved_metrics_survive_a_restart(self):
        """The meter writes beside the database and loads back into a fresh meter."""
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'claims.sqlite3'
            meter = SweepMeter('s')
            meter.record({'stage': 'd1', 'phase': 'done', 'ms': 120})
            meter.save(database)
            self.assertEqual(load_metrics(database, 's')['latency_ms']['d1']['count'], 1)
            again = SweepMeter.load(database, 's')
            again.record({'stage': 'd1', 'phase': 'done', 'ms': 80})
            self.assertEqual(again.summary()['latency_ms']['d1']['count'], 2)
            self.assertIsNone(load_metrics(database, 'other'))


if __name__ == '__main__':
    unittest.main()


class PacketMetricsTests(unittest.TestCase):
    """The packet and report carry the sweep's metered cost and latency."""

    def test_packet_reports_saved_metrics(self):
        """Saved metrics appear in the packet and the report."""
        from datetime import datetime, timezone
        from library_claim.domain.models import Sweep
        from library_claim.packet import build_packet, render_report
        from library_claim.storage.repository import Repository
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / 'claims.sqlite3'
            with Repository(database) as repo:
                repo.add(Sweep(id='s', captured_at=datetime(2026, 11, 14, tzinfo=timezone.utc)))
            meter = SweepMeter('s')
            meter.record({'stage': 'spine_reading', 'phase': 'done', 'ms': 2400})
            meter.record({'stage': 'usage', 'phase': 'done', 'usage': {'source': 'nimble', 'calls': 3}})
            meter.save(database)
            packet = build_packet(database, 's', price=False)
        self.assertEqual(packet['metrics']['latency_ms']['spine_reading']['p50'], 2400)
        self.assertAlmostEqual(packet['metrics']['cost_usd']['by_source']['nimble'], 0.003)
        self.assertIn('Estimated cost USD 0.0030', render_report(packet))
