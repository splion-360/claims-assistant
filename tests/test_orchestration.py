"""Verify sweep transitions and bounded scheduling with simulated processing."""
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

from library_claim.domain.models import Frame, ProcessingJob, Sweep
from library_claim.orchestration.scheduler import Scheduler
from library_claim.orchestration.state_machine import SweepController
from library_claim.storage.repository import Repository

NOW = datetime.now(timezone.utc)


class StateTests(unittest.TestCase):
    """Verify persisted capture state and context invalidation."""

    def test_locked_cycle_and_recovery(self):
        """Reject skipped states and preserve the target through the locked cycle."""
        with tempfile.TemporaryDirectory() as directory, Repository(Path(directory) / 'db') as repo:
            sweep = repo.add(Sweep(captured_at=NOW))
            controller = SweepController(repo, sweep.id)
            with self.assertRaises(ValueError):
                controller.transition('Capturing')
            with self.assertRaises(ValueError):
                controller.transition('Approaching')
            for state in ('Approaching', 'Capturing', 'Resuming', 'Sweeping'):
                result = controller.transition(state, **({'target_id': 'shelf1'} if state == 'Approaching' else {}))
                self.assertEqual(result.state, state)
                self.assertEqual(result.active_target, None if state == 'Sweeping' else 'shelf1')
            self.assertEqual(result.context_revision, 4)
            controller.transition('Approaching', target_id='shelf2')
            recovered = controller.recover()
            self.assertEqual(recovered.state, 'Sweeping')
            self.assertIsNone(recovered.active_target)
            self.assertEqual(recovered.context_revision, 6)


class SchedulerTests(unittest.IsolatedAsyncioTestCase):
    """Exercise admission, supersession, recovery, and late-result rejection."""

    async def asyncSetUp(self):
        """Create a persisted sweep and a controlled fake processor."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.database = Path(self.directory.name) / 'db'
        self.repo = Repository(self.database)
        self.addCleanup(self.repo.close)
        self.sweep = self.repo.add(Sweep(captured_at=NOW))
        self.started = asyncio.Queue()
        self.release = asyncio.Event()
        self.active = 0
        self.peak = 0
        self.scheduler = Scheduler(self.database, self.sweep.id, self.run_job,
                                   concurrency=2, pending_limit=2, timeout_s=2)
        self.addAsyncCleanup(self.scheduler.close)
        await self.scheduler.start()

    async def run_job(self, job):
        """Block simulated processing until the test releases it.

        Args:
            job: Scheduled request.

        Returns:
            A result naming the processed frame.
        """
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.started.put_nowait(job.id)
        try:
            await self.release.wait()
            return {'frame': job.frame_id}
        finally:
            self.active -= 1

    def make_job(self, target='shelf1', score=.5):
        """Persist a frame and construct a request using the current context.

        Args:
            target: Shelf region used for coalescing requests.
            score: Comparative evidence quality supplied by the caller.

        Returns:
            A new pending request.
        """
        frame = self.repo.add(Frame(sweep_id=self.sweep.id, captured_at=NOW,
                                    evidence_key='a' * 64 + '.jpg', width_px=640, height_px=480))
        sweep = self.repo.get(Sweep, self.sweep.id)
        return ProcessingJob(sweep_id=sweep.id, frame_id=frame.id, target_id=target,
                             task='identify', created_at=NOW, evidence_score=score,
                             context_revision=sweep.context_revision)

    async def next_started(self):
        """Wait briefly for a simulated processing call to begin.

        Returns:
            The started job identifier.
        """
        return await asyncio.wait_for(self.started.get(), 2)

    async def idle(self):
        """Bound the test's wait for all scheduled work to finish."""
        await asyncio.wait_for(self.scheduler.wait_idle(), 3)

    async def test_concurrency_and_pending_capacity(self):
        """Keep active calls and queued targets within their configured limits."""
        jobs = [self.make_job(f'shelf{i}') for i in range(5)]
        for job in jobs[:2]:
            self.assertTrue(await self.scheduler.submit(job))
            await self.next_started()
        for job in jobs[2:4]:
            self.assertTrue(await self.scheduler.submit(job))
        self.assertFalse(await self.scheduler.submit(jobs[4]))
        self.release.set()
        await self.idle()
        self.assertEqual(self.peak, 2)
        self.assertEqual(len(self.repo.list_for_sweep(ProcessingJob, self.sweep.id)), 4)
        self.assertTrue(all(self.repo.get(ProcessingJob, job.id).status == 'succeeded' for job in jobs[:4]))

    async def test_pending_slot_retains_best_evidence(self):
        """Replace inferior pending evidence while preserving a bounded queue."""
        for target in ('active1', 'active2'):
            await self.scheduler.submit(self.make_job(target))
            await self.next_started()
        first = self.make_job('waiting', .5)
        better = self.make_job('waiting', .9)
        await self.scheduler.submit(first)
        self.assertFalse(await self.scheduler.submit(self.make_job('waiting', .4)))
        self.assertTrue(await self.scheduler.submit(better))
        self.assertEqual(self.repo.get(ProcessingJob, first.id).status, 'cancelled')
        self.release.set()
        await self.idle()
        self.assertEqual(self.repo.get(ProcessingJob, better.id).status, 'succeeded')

    async def test_superseded_runner_cannot_publish_late_result(self):
        """Discard results even when a superseded processor swallows cancellation."""
        async def stubborn(job):
            """Simulate a provider returning a response after cancellation.

            Args:
                job: Request being processed.

            Returns:
                A late provider result.
            """
            self.started.put_nowait(job.id)
            try:
                await self.release.wait()
            except asyncio.CancelledError:
                return {'late': True}
            return {'late': False}

        self.scheduler.runner = stubborn
        old = self.make_job(score=.4)
        await self.scheduler.submit(old)
        await self.next_started()
        new = self.make_job(score=.8)
        await self.scheduler.submit(new)
        await self.next_started()
        self.release.set()
        await self.idle()
        self.assertEqual(self.repo.get(ProcessingJob, old.id).status, 'cancelled')
        self.assertIsNone(self.repo.get(ProcessingJob, old.id).result)
        self.assertEqual(self.repo.get(ProcessingJob, new.id).status, 'succeeded')

    async def test_context_revision_rejects_stale_results_and_admission(self):
        """Prevent an old view from publishing after a user correction."""
        job = self.make_job()
        await self.scheduler.submit(job)
        await self.next_started()
        SweepController(self.repo, self.sweep.id).invalidate_context()
        self.assertFalse(await self.scheduler.submit(job.model_copy(update={'id': 'stale'})))
        self.release.set()
        await self.idle()
        stored = self.repo.get(ProcessingJob, job.id)
        self.assertEqual(stored.status, 'cancelled')
        self.assertIsNone(stored.result)

    async def test_failure_and_timeout_do_not_stop_workers(self):
        """Persist safe errors and let the worker process subsequent requests."""
        async def fail(job):
            """Raise an exception containing details that must not be persisted.

            Args:
                job: Request being processed.
            """
            raise RuntimeError('secret provider response')

        self.scheduler.runner = fail
        failed = self.make_job()
        await self.scheduler.submit(failed)
        await self.idle()
        self.assertEqual(self.repo.get(ProcessingJob, failed.id).error, 'Processing failed')
        self.scheduler.runner = self.run_job
        self.scheduler.timeout_s = .02
        timed_out = self.make_job()
        await self.scheduler.submit(timed_out)
        await self.idle()
        self.assertEqual(self.repo.get(ProcessingJob, timed_out.id).error, 'Processing timed out')
        self.release.set()
        good = self.make_job()
        await self.scheduler.submit(good)
        await self.idle()
        self.assertEqual(self.repo.get(ProcessingJob, good.id).status, 'succeeded')

    async def test_shutdown_and_restart_recovery(self):
        """Cancel active work and preserve completed results across restart."""
        job = self.make_job()
        await self.scheduler.submit(job)
        await self.next_started()
        await self.scheduler.close()
        self.assertEqual(self.repo.get(ProcessingJob, job.id).status, 'cancelled')
        abandoned = self.repo.add(self.make_job().model_copy(update={'status': 'running'}))
        pending = self.repo.add(self.make_job())
        finished = self.repo.add(self.make_job().model_copy(update={'status': 'succeeded', 'result': {'saved': True}}))
        restarted = Scheduler(self.database, self.sweep.id, self.run_job)
        await restarted.start()
        try:
            for old in (abandoned, pending):
                self.assertEqual(self.repo.get(ProcessingJob, old.id).status, 'cancelled')
            self.assertEqual(self.repo.get(ProcessingJob, finished.id).result, {'saved': True})
            self.assertEqual(self.repo.get(Sweep, self.sweep.id).state, 'Sweeping')
        finally:
            await restarted.close()

    async def test_stale_pending_job_never_reaches_processor(self):
        """Reject queued evidence after a context change before dispatch."""
        for target in ('active1', 'active2'):
            await self.scheduler.submit(self.make_job(target))
            await self.next_started()
        pending = self.make_job('waiting')
        await self.scheduler.submit(pending)
        SweepController(self.repo, self.sweep.id).invalidate_context()
        self.release.set()
        await self.idle()
        self.assertTrue(self.started.empty())
        self.assertEqual(self.repo.get(ProcessingJob, pending.id).status, 'cancelled')

    async def test_timeout_remains_failure_when_runner_swallows_cancellation(self):
        """Reject late results returned in response to a timeout cancellation."""
        async def late(job):
            """Return only after the scheduler's timeout cancels the call.

            Args:
                job: Request being processed.

            Returns:
                A result that arrived after its deadline.
            """
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                return {'late': True}

        self.scheduler.runner = late
        self.scheduler.timeout_s = .01
        job = self.make_job()
        await self.scheduler.submit(job)
        await self.idle()
        stored = self.repo.get(ProcessingJob, job.id)
        self.assertEqual(stored.status, 'failed')
        self.assertIsNone(stored.result)

    async def test_close_cancels_waiting_jobs_and_rejects_new_work(self):
        """Persist cancellation for queued work and stop admission on close."""
        for target in ('active1', 'active2'):
            await self.scheduler.submit(self.make_job(target))
            await self.next_started()
        pending = self.make_job('waiting')
        await self.scheduler.submit(pending)
        await self.scheduler.close()
        self.assertEqual(self.repo.get(ProcessingJob, pending.id).status, 'cancelled')
        with self.assertRaises(RuntimeError):
            await self.scheduler.submit(self.make_job('new'))
        await self.idle()
