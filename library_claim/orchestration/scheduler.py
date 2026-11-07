"""Schedule bounded processing jobs and persist only current results."""
import asyncio
import math
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from library_claim.domain.models import ProcessingJob, Sweep
from library_claim.config import CONFIG
from library_claim.storage.repository import Repository

Runner = Callable[[ProcessingJob], Awaitable[dict[str, Any]]]
Key = tuple[str, str]


class Scheduler:
    """Own a fixed worker pool for one sweep in a single application process."""

    def __init__(self, database: str | Path, sweep_id: str, runner: Runner,
                 *, concurrency: int = CONFIG.scheduler.concurrency, pending_limit: int = CONFIG.scheduler.pending_limit,
                 timeout_s: float = CONFIG.scheduler.timeout_s):
        """Configure bounded execution without starting background tasks.

        Args:
            database: Path to an initialized SQLite database.
            sweep_id: Sweep exclusively owned by this scheduler.
            runner: Async, cancellation-cooperative processor returning JSON data.
            concurrency: Maximum simultaneous processing calls.
            pending_limit: Maximum waiting target/task pairs.
            timeout_s: Processing timeout for each call.

        Raises:
            ValueError: A configured limit is nonpositive or nonfinite.
        """
        if concurrency < 1 or pending_limit < 1 or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('Scheduler limits must be positive and finite')
        self.database = database
        self.sweep_id = sweep_id
        self.runner = runner
        self.concurrency = concurrency
        self.pending_limit = pending_limit
        self.timeout_s = timeout_s
        self._condition = asyncio.Condition()
        self._pending: OrderedDict[Key, ProcessingJob] = OrderedDict()
        self._active: dict[Key, tuple[ProcessingJob, asyncio.Task]] = {}
        self._workers: list[asyncio.Task] = []
        self._failure: Exception | None = None
        self._started = False
        self._closed = False

    async def _store(self, operation: Callable[[Repository], Any]) -> Any:
        """Run SQLite work off the event loop and finish it before cancellation.

        Args:
            operation: Synchronous repository operation.

        Returns:
            The operation's result.
        """
        def execute():
            """Execute one operation with a connection owned by this thread."""
            with Repository(self.database) as repository:
                return operation(repository)

        # A cancelled await must not leave an unobserved database write running.
        task = asyncio.create_task(asyncio.to_thread(execute))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            await task
            raise

    async def start(self) -> None:
        """Cancel abandoned jobs, reset capture context, and start workers.

        Raises:
            RuntimeError: This scheduler has already started or closed.
        """
        from .state_machine import SweepController

        async with self._condition:
            if self._started or self._closed:
                raise RuntimeError('Scheduler cannot be started again')

            def recover(repository):
                """Mark abandoned requests terminal before accepting fresh frames."""
                for job in repository.list_for_sweep(ProcessingJob, self.sweep_id):
                    if job.status in ('pending', 'running'):
                        repository.update(job.model_copy(update={
                            'status': 'cancelled', 'result': None,
                            'error': 'Interrupted before scheduler startup; capture fresh evidence',
                        }))
                SweepController(repository, self.sweep_id).recover()

            await self._store(recover)
            self._started = True
            self._workers = [asyncio.create_task(self._worker()) for _ in range(self.concurrency)]

    @staticmethod
    def _key(job: ProcessingJob) -> Key:
        """Identify the pending slot for a target and processing task.

        Args:
            job: Request carrying a target or physical-book identifier.

        Returns:
            Target identifier and task name.

        Raises:
            ValueError: Neither a region target nor a book is supplied.
        """
        target = job.target_id or job.book_id
        if target is None:
            raise ValueError('A scheduling target is required')
        return target, job.task

    async def submit(self, job: ProcessingJob) -> bool:
        """Accept fresh evidence or replace inferior work for the same target.

        Args:
            job: New pending request with a caller-supplied comparative score.

        Returns:
            Whether the request was accepted and persisted. Equal-score views
            do not replace work; a new context revision permits a fresh request.

        Raises:
            ValueError: The request belongs to another sweep or is not pending.
            RuntimeError: The scheduler is not accepting requests.
        """
        job = ProcessingJob.model_validate(job.model_dump())
        if job.sweep_id != self.sweep_id or job.status != 'pending' or job.result is not None:
            raise ValueError('Submit a new pending job for this sweep')
        key = self._key(job)
        async with self._condition:
            if not self._started or self._closed:
                raise RuntimeError('Scheduler is not accepting requests')
            sweep = await self._store(lambda repo: repo.get(Sweep, self.sweep_id))
            if job.context_revision != sweep.context_revision:
                return False
            pending = self._pending.get(key)
            active = self._active.get(key)
            existing = pending or (active[0] if active else None)
            if existing and existing.context_revision == job.context_revision:
                if existing.frame_id == job.frame_id or existing.evidence_score >= job.evidence_score:
                    return False
            if pending is None and len(self._pending) >= self.pending_limit:
                return False
            # Persist admission before retiring a previous request.
            await self._store(lambda repo: repo.add(job))
            if pending:
                await self._terminal(pending, 'cancelled', error='Superseded by better evidence')
            if active:
                await self._terminal(active[0], 'cancelled', error='Superseded by better evidence')
                active[1].cancel()
            self._pending[key] = job
            self._condition.notify_all()
            return True

    async def _terminal(self, job: ProcessingJob, status: str,
                        *, result: dict | None = None, error: str | None = None) -> None:
        """Persist a terminal job outcome.

        Args:
            job: Original processing request.
            status: Terminal execution status.
            result: Validated JSON-compatible result when successful.
            error: Safe diagnostic message for an unsuccessful request.
        """
        await self._store(lambda repo: repo.update(job.model_copy(update={
            'status': status, 'result': result, 'error': error,
        })))

    async def _worker(self) -> None:
        """Surface infrastructure failures instead of silently losing a worker."""
        try:
            await self._work_loop()
        except Exception as error:
            async with self._condition:
                self._failure = error
                self._closed = True
                self._condition.notify_all()
            raise

    async def _work_loop(self) -> None:
        """Process eligible slots while preserving per-target serialization."""
        while True:
            async with self._condition:
                await self._condition.wait_for(lambda: self._closed or any(
                    key not in self._active for key in self._pending))
                if self._closed:
                    return
                key = next(key for key in self._pending if key not in self._active)
                job = self._pending.pop(key)
                sweep = await self._store(lambda repo: repo.get(Sweep, self.sweep_id))
                if job.context_revision != sweep.context_revision:
                    await self._terminal(job, 'cancelled', error='Capture context changed')
                    self._condition.notify_all()
                    continue
                await self._store(lambda repo: repo.update(job.model_copy(update={'status': 'running'})))
                task = asyncio.create_task(self.runner(job))
                self._active[key] = (job, task)
            result, error, status = None, None, 'succeeded'
            try:
                deadline = asyncio.get_running_loop().time() + self.timeout_s
                result = await asyncio.wait_for(task, timeout=self.timeout_s)
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError('Processor returned after its deadline')
                # Validate serialization before a successful result reaches storage.
                validated = ProcessingJob.model_validate(job.model_dump() | {'result': result})
                validated.model_dump_json()
                if not isinstance(result, dict):
                    raise ValueError('Runner must return an object')
            except asyncio.CancelledError:
                status, error = 'cancelled', 'Processing cancelled'
            except TimeoutError:
                status, error = 'failed', 'Processing timed out'
            except Exception:
                status, error = 'failed', 'Processing failed'
            async with self._condition:
                # Cancellation is advisory: even a runner that returns late cannot win.
                def finish(repository):
                    """Accept a result only while its persisted context is current."""
                    # Serialize revision checks and acceptance against state changes.
                    repository.connection.execute('BEGIN IMMEDIATE')
                    current = repository.get(ProcessingJob, job.id)
                    sweep = repository.get(Sweep, self.sweep_id)
                    if current.status == 'cancelled':
                        return
                    stale = self._closed or sweep.context_revision != job.context_revision
                    repository.update(job.model_copy(update={
                        'status': 'cancelled' if stale else status,
                        'result': result if not stale and status == 'succeeded' else None,
                        'error': 'Capture context changed' if stale else error,
                    }))
                try:
                    await self._store(finish)
                finally:
                    del self._active[key]
                    self._condition.notify_all()

    async def wait_idle(self) -> None:
        """Wait until all admitted work has reached a terminal outcome."""
        async with self._condition:
            await self._condition.wait_for(lambda: self._failure is not None or (
                not self._pending and not self._active))
            if self._failure is not None:
                raise RuntimeError('Scheduler storage or worker failed') from self._failure

    async def close(self) -> None:
        """Stop admission, cancel unfinished work, and await worker cleanup."""
        async with self._condition:
            self._closed = True
            for job in self._pending.values():
                await self._terminal(job, 'cancelled', error='Scheduler closed')
            self._pending.clear()
            for job, task in self._active.values():
                await self._terminal(job, 'cancelled', error='Scheduler closed')
                task.cancel()
            self._condition.notify_all()
        await asyncio.gather(*self._workers)
