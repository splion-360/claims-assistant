"""Report pipeline stage starts, results and failures to whoever is listening, if anyone."""
from contextlib import contextmanager
from contextvars import ContextVar
import time
from typing import Callable

# Context-local so each live session's tasks and worker threads report to that session's listener.
_listener: ContextVar[Callable[[dict], None] | None] = ContextVar('pipeline_listener', default=None)
# The innermost running span, so code inside it (e.g. the provider) can record which model answered.
_current: ContextVar['_Span | None'] = ContextVar('pipeline_span', default=None)


def emit(stage: str, phase: str, detail: str | None = None, ms: float | None = None,
         model: str | None = None) -> None:
    """Send one stage event to the current listener.

    Args:
        stage: Pipeline block identifier, e.g. 'd1' or 'spine_reading'.
        phase: 'start', 'done', 'error' or 'skipped'.
        detail: Short human-readable result.
        ms: Elapsed milliseconds for a finished stage.
        model: Model that did the work, when known.
    """
    listener = _listener.get()
    if listener is not None:
        listener({'stage': stage, 'phase': phase, 'detail': detail, 'model': model,
                  'ms': None if ms is None else round(ms), 't': time.time()})


def record_model(model: str) -> None:
    """Note which model answered inside the current span, if any.

    Args:
        model: Model identifier as named by the provider.
    """
    current = _current.get()
    if current is not None:
        current.model = model


@contextmanager
def listening(listener: Callable[[dict], None]):
    """Route events emitted in this context (and tasks/threads started from it) to a listener.

    Args:
        listener: Callable receiving event dicts; must be thread-safe.

    Yields:
        None.
    """
    token = _listener.set(listener)
    try:
        yield
    finally:
        _listener.reset(token)


def attach(listener: Callable[[dict], None]) -> None:
    """Set the listener for the rest of the current task and everything it starts.

    Args:
        listener: Callable receiving event dicts; must be thread-safe.
    """
    _listener.set(listener)


class _Span:
    """Mutable holder for the result a span reports when it finishes."""

    def __init__(self):
        """Start with no detail or model."""
        self.detail: str | None = None
        self.model: str | None = None


@contextmanager
def span(stage: str, detail: str | None = None):
    """Emit start, then done or error with elapsed time, around a block of work.

    Args:
        stage: Pipeline block identifier.
        detail: Optional detail shown while the stage runs.

    Yields:
        A holder whose `detail` is reported when the stage finishes.
    """
    holder = _Span()
    token = _current.set(holder)
    emit(stage, 'start', detail)
    started = time.monotonic()
    try:
        yield holder
    except BaseException as error:
        emit(stage, 'error', type(error).__name__, (time.monotonic() - started) * 1000, holder.model)
        raise
    finally:
        _current.reset(token)
    emit(stage, 'done', holder.detail, (time.monotonic() - started) * 1000, holder.model)
