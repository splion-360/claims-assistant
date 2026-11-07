"""Persist book readings and associate physical copies through verified frame geometry."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import cv2
import numpy as np

from library_claim.capture.quality import QualityAssessor
from library_claim.capture.tracking.geometry import transform
from library_claim.config import CONFIG, CONFIG_DIGEST
from library_claim.domain.models import Book, BookObservation, Frame, FrameAssociation, ProcessingJob, Quality, ReviewIssue
from library_claim.storage.repository import Repository
from .models import BookFrameAnalysis


def polygon(box) -> np.ndarray:
    """Convert a normalized box to clockwise corner points.

    Args:
        box: Validated image bounding box.

    Returns:
        Four normalized corners.
    """
    return np.float32([(box.x_min, box.y_min), (box.x_max, box.y_min),
                       (box.x_max, box.y_max), (box.x_min, box.y_max)])


def reference_geometry(frame_id: str, associations: dict) -> tuple[str, np.ndarray]:
    """Follow saved mappings to a shared reference without inventing missing links.

    Args:
        frame_id: Frame whose coordinate basis is requested.
        associations: Validated frame-ID-to-association mapping for this sweep.

    Returns:
        Root frame ID and normalized current-to-root homography.

    Raises:
        ValueError: The chain cycles or exceeds its configured bound.
    """
    matrix, visited = np.eye(3), set()
    for _ in range(CONFIG.identification.association_max_hops):
        if frame_id not in associations:
            return frame_id, matrix
        if frame_id in visited:
            raise ValueError('Cyclic frame association')
        visited.add(frame_id)
        edge = associations[frame_id]
        matrix = np.asarray(edge.matrix) @ matrix
        frame_id = edge.reference_frame_id
    raise ValueError('Frame association chain exceeds limit')


def overlap(first: np.ndarray, second: np.ndarray) -> float:
    """Measure polygon intersection over union in a verified common coordinate basis.

    Args:
        first: Four projected corners.
        second: Four projected corners in the same basis.

    Returns:
        Intersection over union, or zero for invalid/nonconvex polygons.
    """
    a, b = np.float32(first), np.float32(second)
    if not cv2.isContourConvex(a) or not cv2.isContourConvex(b):
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - intersection
    return max(0.0, min(1.0, float(intersection / union))) if union > 0 else 0.0


def association_choices(frame: Frame, analysis: BookFrameAnalysis, frames: dict,
                        observations: list[BookObservation], associations: dict) -> list[str | None]:
    """Choose only unique geometric matches, preserving separate copies of one title.

    Args:
        frame: Current saved source frame.
        analysis: Model's per-book detections.
        frames: Saved frames in this sweep indexed by ID.
        observations: Previously stored book observations.
        associations: Saved frame mappings indexed by source frame.

    Returns:
        Existing physical-book ID per detection, or None for unverified identity.
    """
    choices = []
    for detection in analysis.books:
        scores = {}
        for old in observations:
            try:
                # Identical JPEG evidence is the only fallback when geometry is absent.
                if frames[old.frame_id].evidence_key == frame.evidence_key:
                    current_poly, old_poly = polygon(detection.bbox), polygon(old.bbox)
                else:
                    root, matrix = reference_geometry(frame.id, associations)
                    old_root, old_matrix = reference_geometry(old.frame_id, associations)
                    if root != old_root:
                        continue
                    current_poly = transform(polygon(detection.bbox), matrix)
                    old_poly = transform(polygon(old.bbox), old_matrix)
                score = overlap(current_poly, old_poly)
                scores[old.book_id] = max(scores.get(old.book_id, 0), score)
            except (ValueError, cv2.error):
                continue
        ranked = sorted(scores, key=scores.get, reverse=True)
        selected = None
        if ranked:
            best = ranked[0]
            runner_up = scores[ranked[1]] if len(ranked) > 1 else 0
            conflicting = any(old.book_id == best and old.title and detection.title
                              and old.title.casefold() != detection.title.casefold() for old in observations)
            if (scores[best] >= CONFIG.identification.association_min_iou
                    and scores[best] - runner_up >= CONFIG.identification.association_min_margin
                    and not conflicting and 'uncertain_book' not in detection.issues):
                selected = best
        choices.append(selected)
    # Competing detections cannot both consume the same physical copy.
    return [choice if choice is not None and choices.count(choice) == 1 else None for choice in choices]


def inventory(repo: Repository, sweep_id: str) -> list[dict]:
    """Build a provisional inventory from nonconflicting visual readings.

    Args:
        repo: Open repository on the caller's thread.
        sweep_id: Owning sweep.

    Returns:
        Physical-copy records with observations, unknowns, and review reasons.
    """
    observations = repo.list_for_sweep(BookObservation, sweep_id)
    issues = repo.list_for_sweep(ReviewIssue, sweep_id)
    result = []
    for book in repo.list_for_sweep(Book, sweep_id):
        evidence = [row for row in observations if row.book_id == book.id]
        readings = {}
        for field in ('title', 'author', 'publisher'):
            values = {getattr(row, field).casefold(): getattr(row, field) for row in evidence if getattr(row, field)}
            readings[field] = next(iter(values.values())) if len(values) == 1 else None
        result.append({'id': book.id, **readings, 'status': 'provisional',
                       'observation_count': len(evidence),
                       'frame_ids': list(dict.fromkeys(row.frame_id for row in evidence)),
                       'review_reasons': list(dict.fromkeys(issue.reason for issue in issues
                                                           if issue.book_id == book.id and issue.status == 'open'))})
    return result


def save_analysis(database: Path, sweep_id: str, frame_id: str, analysis: BookFrameAnalysis,
                  jpeg: bytes, model: str) -> dict:
    """Append a frame's readings atomically and return its provisional inventory.

    Args:
        database: SQLite database containing the source frame.
        sweep_id: Owning capture session.
        frame_id: Exact saved frame inspected by the model.
        analysis: Validated structured extraction.
        jpeg: Original frame for local per-book quality measurements.
        model: Provider model used for extraction.

    Returns:
        Frame detection count, completion flag, and provisional inventory.

    Raises:
        ValueError: Source frame ownership or image decoding is invalid.
        KeyError: The source frame was not saved.
    """
    with Repository(database) as repo:
        frame = repo.get(Frame, frame_id)
        if frame.sweep_id != sweep_id:
            raise ValueError('Book evidence belongs to another sweep')
        if hashlib.sha256(jpeg).hexdigest() != frame.evidence_key.split('.')[0]:
            raise ValueError('Image bytes do not match the saved source frame')
        job_id = f'books_{frame_id}'
        try:
            existing = repo.get(ProcessingJob, job_id)
        except KeyError:
            existing = None
        if existing is not None:
            return {**existing.result, 'inventory': inventory(repo, sweep_id)}
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError('Cannot decode source frame')
        frames = {row.id: row for row in repo.list_for_sweep(Frame, sweep_id)}
        observations = repo.list_for_sweep(BookObservation, sweep_id)
        associations = {row.frame_id: row for row in repo.list_for_sweep(FrameAssociation, sweep_id)}
        choices = association_choices(frame, analysis, frames, observations, associations)
        records = []
        assessor = QualityAssessor() if analysis.books else None
        for detection, matched in zip(analysis.books, choices):
            book_id = matched
            reasons = list(detection.issues)
            if book_id is None:
                book = Book(sweep_id=sweep_id)
                book_id = book.id
                records.append(book)
                if observations:
                    reasons.append('Physical-copy association unverified; possible duplicate')
            if not analysis.complete:
                reasons.append('Frame enumeration incomplete')
            # Region scores describe this book, never the sharp background around it.
            signals, metadata = Quality(), None
            try:
                measured = assessor.assess(image, region=detection.bbox)
                signals = measured.signals
                metadata = measured.model_dump(mode='json', exclude={'signals'})
            except (ValueError, cv2.error):
                reasons.append('Book region too small or invalid for quality assessment')
            unresolved = [field for field in ('title', 'author', 'publisher') if getattr(detection, field) is None]
            observation = BookObservation(sweep_id=sweep_id, book_id=book_id, frame_id=frame_id,
                                          bbox=detection.bbox, view=detection.view, visible_text=detection.visible_text,
                                          title=detection.title, author=detection.author, publisher=detection.publisher,
                                          quality=signals, quality_metadata=metadata, visual_issues=detection.issues,
                                          association_method='verified_geometry_or_identical_image' if matched else 'new_provisional',
                                          source_model=model, unresolved_fields=unresolved)
            records.append(observation)
            for reason in reasons:
                records.append(ReviewIssue(sweep_id=sweep_id, book_id=book_id, field='visual_evidence', reason=reason))
        result = {'frame_id': frame_id, 'detected_count': len(analysis.books), 'complete': analysis.complete,
                  'analysis': analysis.model_dump(mode='json')}
        records.append(ProcessingJob(id=job_id, sweep_id=sweep_id, frame_id=frame_id, task='identify',
                                     status='succeeded', created_at=datetime.now(timezone.utc), model=model,
                                     configuration_version=CONFIG_DIGEST, result=result))
        repo.add_many(records)
        return {**result, 'inventory': inventory(repo, sweep_id)}
