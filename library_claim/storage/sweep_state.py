"""Record the facts the live agent establishes about a sweep: market, end time, packet time."""
from datetime import datetime, timezone
from pathlib import Path

from library_claim.config import CONFIG
from library_claim.domain.models import Item, Sweep
from .repository import Repository


def _update_sweep(database: Path, sweep_id: str, **fields) -> None:
    """Set fields on a sweep, creating the record when no frame has been saved yet.

    Args:
        database: Capture SQLite path.
        sweep_id: Sweep to update.
        **fields: Sweep fields to set.
    """
    with Repository(database) as repo:
        try:
            sweep = repo.get(Sweep, sweep_id)
        except KeyError:
            repo.add(Sweep(id=sweep_id, captured_at=datetime.now(timezone.utc), device='browser camera', **fields))
            return
        repo.update(sweep.model_copy(update=fields))


def set_sweep_locale(database: Path, sweep_id: str, country: str) -> None:
    """Save the market the policyholder confirmed.

    Args:
        database: Capture SQLite path.
        sweep_id: Sweep to update.
        country: Configured market code.
    """
    code = country.upper()
    _update_sweep(database, sweep_id, country=code, currency=CONFIG.pricing.markets[code].currency)


def mark_sweep_ended(database: Path, sweep_id: str, when: datetime | None = None) -> None:
    """Record when the camera stopped, the start of the time-to-packet clock.

    Args:
        database: Capture SQLite path.
        sweep_id: Sweep to update.
        when: End time; now when omitted.
    """
    _update_sweep(database, sweep_id, ended_at=when or datetime.now(timezone.utc), packet_ready_at=None)


def mark_packet_ready(database: Path, sweep_id: str, when: datetime | None = None) -> None:
    """Record when background work finished and the packet could be produced.

    Args:
        database: Capture SQLite path.
        sweep_id: Sweep to update.
        when: Ready time; now when omitted.
    """
    _update_sweep(database, sweep_id, packet_ready_at=when or datetime.now(timezone.utc))


def set_stated_print(database: Path, item_id: str, stated_print: bool) -> None:
    """Record whether the policyholder said a framed piece is a print.

    Args:
        database: Capture SQLite path.
        item_id: Object the answer is about.
        stated_print: True for a print, which releases it from appraisal.
    """
    with Repository(database) as repo:
        item = repo.get(Item, item_id)
        repo.update(item.model_copy(update={'stated_print': stated_print}))
