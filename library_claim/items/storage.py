"""Associate non-book detections across frames and build the room's item inventory."""
from datetime import datetime, timezone
import logging
from pathlib import Path

import cv2
import numpy as np

from library_claim.capture.quality import QualityAssessor
from library_claim.config import CONFIG, CONFIG_DIGEST
from library_claim.domain.models import (Frame, FrameAssociation, Item, ItemObservation,
                                         ProcessingJob, Quality)
from library_claim.capture.tracking.geometry import transform
from library_claim.identification.storage import overlap, polygon, reference_geometry
from library_claim.storage.repository import Repository
from .models import APPRAISAL_CATEGORIES, ItemFrameAnalysis

logger = logging.getLogger(__name__)


def _match(detection, frame: Frame, previous: list[ItemObservation], frames: dict,
           associations: dict, taken: set[str]) -> str | None:
    """Find the saved object a detection shows, using verified geometry only.

    Two views of one object must agree on category and overlap in a shared
    coordinate basis. Appearance alone never merges two records, and one saved
    object cannot absorb two detections from the same frame.

    Args:
        detection: One detection from the current frame.
        frame: The current saved frame.
        previous: Observations already saved for this sweep.
        frames: Saved frames indexed by ID.
        associations: Verified frame mappings indexed by source frame.
        taken: Item IDs already claimed by earlier detections in this frame.

    Returns:
        The matching item ID, or None when the evidence does not establish one.
    """
    best, best_score = None, CONFIG.items.association_min_iou
    for observation in previous:
        if observation.item_id in taken or observation.category != detection.category:
            continue
        try:
            # The same saved JPEG shares a basis outright; otherwise both views must
            # reach the same reference frame through verified mappings.
            if frames[observation.frame_id].evidence_key == frame.evidence_key:
                current_poly, old_poly = polygon(detection.bbox), polygon(observation.bbox)
            else:
                root, matrix = reference_geometry(frame.id, associations)
                old_root, old_matrix = reference_geometry(observation.frame_id, associations)
                if root != old_root:
                    continue
                current_poly = transform(polygon(detection.bbox), matrix)
                old_poly = transform(polygon(observation.bbox), old_matrix)
            score = overlap(current_poly, old_poly)
        except (KeyError, ValueError, cv2.error):
            continue
        if score > best_score:
            best, best_score = observation.item_id, score
    return best


def save_items(database: Path, sweep_id: str, frame_id: str, analysis: ItemFrameAnalysis,
               jpeg: bytes, model: str) -> dict:
    """Append one frame's non-book detections and return the sweep's item inventory.

    Args:
        database: Capture SQLite path.
        sweep_id: Owning sweep.
        frame_id: Frame the detections came from.
        analysis: The model's detections for that frame.
        jpeg: The frame's encoded bytes, used for region quality only.
        model: Vision model that produced the detections.

    Returns:
        The detected count, completeness flag and the current item inventory.

    Raises:
        ValueError: The frame bytes cannot be decoded.
    """
    job_id = f'items_{frame_id}'
    with Repository(database) as repo:
        try:
            existing = repo.get(ProcessingJob, job_id)
            return {**existing.result, 'inventory': inventory(repo, sweep_id)}
        except KeyError:
            pass
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Cannot decode source frame')
        frame = repo.get(Frame, frame_id)
        frames = {row.id: row for row in repo.list_for_sweep(Frame, sweep_id)}
        associations = {row.frame_id: row for row in repo.list_for_sweep(FrameAssociation, sweep_id)}
        previous = repo.list_for_sweep(ItemObservation, sweep_id)
        records, taken = [], set()
        assessor = QualityAssessor() if analysis.items else None
        for detection in analysis.items:
            item_id = _match(detection, frame, previous, frames, associations, taken)
            method = 'geometry' if item_id else 'new_provisional'
            reasons = list(detection.issues)
            if item_id is None:
                # Art and mirrors are never valued automatically until the owner says otherwise.
                item = Item(sweep_id=sweep_id, category=detection.category, description=detection.description,
                            material=detection.material, brand_model=detection.brand_model,
                            needs_appraisal=detection.category in APPRAISAL_CATEGORIES)
                item_id = item.id
                records.append(item)
                if previous:
                    reasons.append('Object association unverified; possible duplicate')
            taken.add(item_id)
            signals = Quality()
            try:
                signals = assessor.assess(image, region=detection.bbox).signals
            except (ValueError, cv2.error):
                reasons.append('Object region too small or invalid for quality assessment')
            records.append(ItemObservation(
                sweep_id=sweep_id, item_id=item_id, frame_id=frame_id, bbox=detection.bbox,
                category=detection.category, description=detection.description,
                material=detection.material, brand_model=detection.brand_model,
                visual_issues=detection.issues, review_reasons=reasons, quality=signals,
                association_method=method, source_model=model))
        result = {'frame_id': frame_id, 'detected_count': len(analysis.items), 'complete': analysis.complete}
        records.append(ProcessingJob(id=job_id, sweep_id=sweep_id, frame_id=frame_id, task='items',
                                     status='succeeded', created_at=datetime.now(timezone.utc), model=model,
                                     configuration_version=CONFIG_DIGEST, result=result))
        repo.add_many(records)
        return {**result, 'inventory': inventory(repo, sweep_id)}


def inventory(repo: Repository, sweep_id: str) -> list[dict]:
    """Build the room's non-book inventory from saved observations.

    Args:
        repo: Open repository on the caller's thread.
        sweep_id: Owning sweep.

    Returns:
        One record per physical object, with its readings, frames and review reasons.
    """
    observations = repo.list_for_sweep(ItemObservation, sweep_id)
    result = []
    for item in repo.list_for_sweep(Item, sweep_id):
        rows = [row for row in observations if row.item_id == item.id]
        # The fullest reading wins; a later sighting never erases an earlier one.
        description = max((row.description for row in rows), key=len, default=item.description)
        brand = next((row.brand_model for row in reversed(rows) if row.brand_model), item.brand_model)
        material = next((row.material for row in reversed(rows) if row.material), item.material)
        reasons = [reason for row in rows for reason in row.review_reasons]
        result.append({
            # A settled object leaves the review list; its recorded reasons stay.
            'needs_review': not item.reviewed and (bool(reasons) or not description),
            'reviewed': item.reviewed,
            'id': item.id, 'category': item.category, 'description': description,
            'material': material, 'brand_model': brand,
            'needs_appraisal': item.needs_appraisal and not item.stated_print,
            'stated_print': item.stated_print, 'excluded': item.excluded,
            'observation_count': len(rows),
            'frame_ids': list(dict.fromkeys(row.frame_id for row in rows)),
            # Sighting IDs so a reviewer can see the crop each reading came from.
            'observation_ids': [row.id for row in rows],
            'review_reasons': list(dict.fromkeys(reasons)),
        })
    return result


def current_items(database: Path, sweep_id: str) -> list[dict]:
    """Read the item inventory in its own connection.

    Args:
        database: Capture SQLite path.
        sweep_id: Owning sweep.

    Returns:
        Inventory entries as produced by `inventory`.
    """
    with Repository(database) as repo:
        return inventory(repo, sweep_id)
