"""Verify pipeline stage events reach the active listener, including from worker threads."""
import asyncio
import unittest

from library_claim import telemetry


class TelemetryTests(unittest.IsolatedAsyncioTestCase):
    """Exercise emit, span timing and context propagation."""

    async def test_span_reports_start_and_done_with_detail_and_time(self):
        """A span emits start, then done with its detail and elapsed milliseconds."""
        events = []
        with telemetry.listening(events.append):
            with telemetry.span('d1') as stage:
                stage.detail = 'glare 0.91'
        self.assertEqual([(e['stage'], e['phase']) for e in events], [('d1', 'start'), ('d1', 'done')])
        self.assertEqual(events[1]['detail'], 'glare 0.91')
        self.assertGreaterEqual(events[1]['ms'], 0)

    async def test_span_reports_error_and_reraises(self):
        """A failing span emits an error with the exception type and still raises."""
        events = []
        with telemetry.listening(events.append):
            with self.assertRaises(ValueError):
                with telemetry.span('qwen_vl'):
                    raise ValueError('bad')
        self.assertEqual(events[-1]['phase'], 'error')
        self.assertEqual(events[-1]['detail'], 'ValueError')

    async def test_no_listener_is_silent(self):
        """Without a listener, emit and span do nothing."""
        telemetry.emit('camera', 'done')
        with telemetry.span('d2'):
            pass

    async def test_worker_threads_report_to_the_listener(self):
        """asyncio.to_thread carries the listener into the worker thread."""
        events = []
        with telemetry.listening(events.append):
            await asyncio.to_thread(telemetry.emit, 'depth', 'done', 'ok')
        self.assertEqual(events[0]['stage'], 'depth')
