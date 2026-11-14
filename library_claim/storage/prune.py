"""Remove saved frames that no book evidence depends on once a sweep has ended."""
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path

from library_claim.domain.models import BookObservation, Frame, FrameAssociation
from library_claim.storage.evidence import EvidenceStore
from library_claim.storage.repository import Repository

logger = logging.getLogger(__name__)


def prune_sweep(database: Path, sweep_id: str) -> int:
    """Delete a finished sweep's frames that no observation cites, directly or as a tracking reference.

    Kept: every frame a book observation points to, plus the chain of reference frames those
    frames were tracked against, so every frame_ref still opens and its geometry still resolves.
    Removed: all other frames, their tracking links, their jobs (frame checks, empty reads,
    measurements), their depth sidecars, and any JPEG no remaining frame in any sweep uses.

    Args:
        database: Capture SQLite path; evidence files live beside it.
        sweep_id: Sweep to prune. It must no longer be live.

    Returns:
        Number of frames removed.
    """
    root = Path(database).parent
    with Repository(database) as repo:
        frames = {frame.id: frame for frame in repo.list_for_sweep(Frame, sweep_id)}
        links = {row.frame_id: row.reference_frame_id for row in repo.list_for_sweep(FrameAssociation, sweep_id)}
        keep = {row.frame_id for row in repo.list_for_sweep(BookObservation, sweep_id)}
        # Follow each kept frame's tracking reference until the chain ends.
        pending = list(keep)
        while pending:
            reference = links.get(pending.pop())
            if reference and reference not in keep:
                keep.add(reference)
                pending.append(reference)
        removed = [frame_id for frame_id in frames if frame_id not in keep]
        if not removed:
            return 0
        marks = ','.join('?' * len(removed))
        connection = repo.connection
        # Children first: foreign keys forbid deleting a frame that is still referenced.
        with connection:
            connection.execute(f'DELETE FROM frame_associations WHERE sweep_id = ? AND '
                               f'(frame_id IN ({marks}) OR reference_frame_id IN ({marks}))',
                               (sweep_id, *removed, *removed))
            connection.execute(f'DELETE FROM jobs WHERE sweep_id = ? AND frame_id IN ({marks})', (sweep_id, *removed))
            connection.execute(f'DELETE FROM frames WHERE sweep_id = ? AND id IN ({marks})', (sweep_id, *removed))
        # Images are stored once per content hash, so another frame may still use the same file.
        still_used = {row[0] for row in connection.execute("SELECT json_extract(data, '$.evidence_key') FROM frames")}
    evidence = EvidenceStore(root / 'frames')
    for frame_id in removed:
        key = frames[frame_id].evidence_key
        if key not in still_used:
            evidence.resolve(key).unlink(missing_ok=True)
            still_used.add(key)
        (root / 'depth' / f'{frame_id}.npz').unlink(missing_ok=True)
    logger.info('pruned sweep=%s removed_frames=%d kept_frames=%d', sweep_id, len(removed), len(keep))
    return len(removed)


def prune_finished(database: Path, live: set[str], idle_s: float) -> int:
    """Prune every sweep that is not live and has been idle long enough, e.g. at server start.

    The idle check protects sweeps another server process may still be using.

    Args:
        database: Capture SQLite path.
        live: Sweep IDs known to be in use, which are left untouched.
        idle_s: Minimum seconds since a sweep's newest frame before it may be pruned.

    Returns:
        Total number of frames removed.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=idle_s)
    with Repository(database) as repo:
        sweeps = [row[0] for row in repo.connection.execute('SELECT id FROM sweeps')]
        finished = []
        for sweep_id in sweeps:
            frames = repo.list_for_sweep(Frame, sweep_id)
            if sweep_id not in live and all(frame.captured_at < cutoff for frame in frames):
                finished.append(sweep_id)
    return sum(prune_sweep(database, sweep_id) for sweep_id in finished)
