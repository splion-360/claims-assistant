"""Measure one sweep's room once and keep the figure for the panel and the packet."""
from datetime import datetime, timezone
import logging
from pathlib import Path

from library_claim.config import CONFIG, CONFIG_DIGEST
from library_claim.domain.models import Frame, ItemObservation, ProcessingJob
from library_claim.storage.depth import load_depth
from library_claim.storage.repository import Repository
from .room import RoomEstimate, estimate_room, surface_area_m2

logger = logging.getLogger(__name__)
# Categories whose detections count towards the wall area the shelving covers.
SHELVED_CATEGORIES = ('shelving',)


def shelved_wall_area_m2(repo: Repository, data_dir: Path, sweep_id: str) -> float:
    """Measure the wall area the sweep's shelving covers.

    One unit seen in several frames is measured once, at its largest reading,
    rather than counted again for every sighting.

    Args:
        repo: Open repository on the caller's thread.
        data_dir: Capture data directory holding the depth sidecars.
        sweep_id: Owning sweep.

    Returns:
        Square metres of wall covered by shelving; zero when none was measurable.
    """
    largest: dict[str, float] = {}
    for row in repo.list_for_sweep(ItemObservation, sweep_id):
        if row.category not in SHELVED_CATEGORIES:
            continue
        depth = load_depth(data_dir, row.frame_id)
        if depth is None:
            continue
        area = surface_area_m2(depth, row.bbox)
        if area is not None:
            largest[row.item_id] = max(largest.get(row.item_id, 0.0), area)
    return float(sum(largest.values()))


def stored_room(repo: Repository, sweep_id: str) -> dict | None:
    """Read the saved room estimate for a sweep.

    Args:
        repo: Open repository on the caller's thread.
        sweep_id: Owning sweep.

    Returns:
        The saved room block, or None when the sweep has no estimate.
    """
    for job in repo.list_for_sweep(ProcessingJob, sweep_id):
        if job.task == 'room' and job.status == 'succeeded' and job.result:
            return job.result['room']
    return None


def measure_room(database: Path, data_dir: Path, sweep_id: str, *, refresh: bool = True) -> dict | None:
    """Measure the room from the sweep's posed depth frames and save the figure.

    Args:
        database: Capture SQLite path.
        data_dir: Capture data directory holding the depth sidecars.
        sweep_id: Owning sweep.
        refresh: Measure again even when an estimate is already saved.

    Returns:
        The room block, or None when too little posed depth was captured.
    """
    job_id = f'room_{sweep_id}'
    with Repository(database) as repo:
        try:
            existing = repo.get(ProcessingJob, job_id)
        except KeyError:
            existing = None
        if existing is not None and not refresh and existing.result:
            return existing.result['room']
        frames = repo.list_for_sweep(Frame, sweep_id)
        shelved = shelved_wall_area_m2(repo, data_dir, sweep_id)
    depths = [depth for depth in (load_depth(data_dir, frame.id) for frame in frames) if depth is not None]
    estimate = estimate_room(depths, shelved_area_m2=shelved)
    if estimate is None:
        return None
    room = estimate.as_packet_room()
    record = ProcessingJob(id=job_id, sweep_id=sweep_id, frame_id=frames[0].id, task='room',
                           status='succeeded', created_at=datetime.now(timezone.utc),
                           model=estimate.scale_method, configuration_version=CONFIG_DIGEST,
                           result={'room': room})
    with Repository(database) as repo:
        if existing is None:
            repo.add(record)
        else:
            repo.update(record)
    return room
