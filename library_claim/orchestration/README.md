# Sweep orchestration

The controller preserves the four states:

`Sweeping → Approaching → Capturing → Resuming → Sweeping`

Only Approaching accepts a new target; Capturing and Resuming retain it, and
Sweeping clears it. Staying in a state needs no transition. Restart recovery is
an explicit reset to Sweeping, not another state. Every transition, correction
invalidation, and recovery increments `context_revision`. A job carrying an old
revision cannot publish a result. Capture can retain unknowns and advance to
Resuming when attempts are exhausted; the controller does not invent evidence.

`SweepController` is synchronous and uses a caller-owned repository. Invoke it on
that repository's owning thread. The application must serialize controller
mutations per sweep. SQLite job acceptance obtains a write transaction before
checking the persisted revision, so completed results are accepted against a
consistent snapshot.

## Scheduler lifecycle

1. Create one Scheduler per sweep and call `await start()` once.
2. Read the recovered sweep's current revision before creating requests.
3. Submit a fresh ProcessingJob with a saved frame, target ID (or book ID), task,
   revision, and evidence score. The score comes from the future evidence selector.
4. Await `wait_idle()` to drain work, or `close()` to cancel and release workers.

This is single-process ownership, not distributed claiming. Do not start a second
scheduler on an actively owned sweep. Startup marks persisted pending/running
jobs cancelled, retains terminal results and book evidence, and resets capture
context. It does not replay provider calls that may already have incurred costs.

Each target/task pair has at most one running call and one waiting observation.
A higher score replaces waiting evidence and cancels an inferior active call.
Equal scores and the same frame in the same context do not replace current work.
A newer context allows fresh evidence regardless of the previous score. Idle
slots do not retain a history-based duplicate cache; the capture selector is
responsible for suppressing repeated completed observations.

The global pending limit bounds memory; admission returns False when no slot is
available. Rejected requests are not persisted. Fixed worker count bounds active
calls. Cancelling a coroutine does not guarantee that a provider stops billing.
Late responses are discarded even if the processor suppresses cancellation.
Processors must cooperate with cancellation and use transport-level timeouts:
Python cannot forcibly terminate a coroutine that never yields or never exits.

SQLite operations use short-lived connections in background threads. Provider
errors persist safe messages, not exception bodies. Infrastructure errors stop
admission and surface to the caller; they are not silently reported as successful
jobs. The caller must await scheduler lifecycle methods rather than fire and
forget them. Result persistence is the publication boundary: runners return data
and must not write inventory or speak independently. Downstream consumers must
also check revision before emitting delayed guidance.

Existing provider and browser integrations are unchanged. The runner is injected,
so the scheduler is verified with fake asynchronous jobs before live integration.
Job metadata uses the existing validated JSON payload; old payloads receive field
defaults, so this increment requires no SQL migration.
