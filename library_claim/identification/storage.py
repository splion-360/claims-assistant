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
from library_claim.storage.evidence import EvidenceStore
from .matching import crop_features, visual_match, text_relation, preferred_reading
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
                        observations: list[BookObservation], associations: dict,
                        image: np.ndarray, evidence: EvidenceStore) -> tuple[list[str | None], list[str]]:
    """Apply geometry/text gates and use crop matching only without shared geometry.

    Args:
        frame: Current saved source frame.
        analysis: Model's per-book detections.
        frames: Saved frames indexed by ID.
        observations: Previous observations in the sweep.
        associations: Saved mappings indexed by source frame.
        image: Current decoded source image.
        evidence: Original JPEG storage.

    Returns:
        Existing book IDs and association methods, including unresolved decisions.
    """
    choices, methods, selected_scores = [], [], []
    features = {}
    established = [old for old in observations if old.association_method != 'unresolved']
    for index, detection in enumerate(analysis.books):
        scores, routes = {}, {}
        possible = False
        for old in established:
            title = text_relation(old.title, detection.title)
            author = text_relation(old.author, detection.author)
            if title == 'conflicting' or author == 'conflicting':
                continue
            compatible_title = title in ('exact', 'partial', 'fuzzy')
            if 'uncertain_book' in detection.issues or 'uncertain_book' in old.visual_issues:
                continue
            try:
                shared = False
                if frames[old.frame_id].evidence_key == frame.evidence_key:
                    current_poly, old_poly = polygon(detection.bbox), polygon(old.bbox)
                    shared = True
                else:
                    root, matrix = reference_geometry(frame.id, associations)
                    old_root, old_matrix = reference_geometry(old.frame_id, associations)
                    if root == old_root:
                        current_poly = transform(polygon(detection.bbox), matrix)
                        old_poly = transform(polygon(old.bbox), old_matrix)
                        shared = True
                if shared:
                    score = overlap(current_poly, old_poly)
                    possible |= score > 0
                    if score < CONFIG.identification.association_min_iou:
                        continue
                    possible = True
                    method = 'geometry_and_compatible_readings'
                else:
                    possible |= compatible_title
                    # Compare like views; a spine/cover change is not a visual contradiction.
                    if not compatible_title or detection.view != old.view:
                        continue
                    current_key, old_key = ('current', index), ('old', old.id)
                    if current_key not in features:
                        features[current_key] = crop_features(image, detection.bbox)
                    if old_key not in features:
                        data = evidence.read(frames[old.frame_id].evidence_key)
                        previous = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                        if previous is None:
                            continue
                        features[old_key] = crop_features(previous, old.bbox)
                    score = visual_match(features[current_key], features[old_key])
                    if score is None:
                        continue
                    method = 'appearance_and_compatible_readings'
                # A contradictory reading elsewhere in the track blocks automatic reuse.
                if any(other.book_id == old.book_id and
                       (text_relation(other.title, detection.title) == 'conflicting' or
                        text_relation(other.author, detection.author) == 'conflicting') for other in established):
                    continue
                if score > scores.get(old.book_id, -1):
                    scores[old.book_id], routes[old.book_id] = score, method
            except (ValueError, OSError, cv2.error):
                continue
        ranked = sorted(scores, key=scores.get, reverse=True)
        selected = None
        method = 'unresolved' if possible else 'new_provisional'
        if ranked:
            best = ranked[0]
            runner_up = scores[ranked[1]] if len(ranked) > 1 else 0
            if scores[best] - runner_up >= CONFIG.identification.association_min_margin:
                selected, method = best, routes[best]
        choices.append(selected)
        methods.append(method)
        selected_scores.append(scores.get(selected, 0))
    # Co-visible exact-title copies establish multiplicity even with similar designs.
    for choice in set(choices) - {None}:
        competitors = [index for index, value in enumerate(choices) if value == choice]
        if len(competitors) < 2:
            continue
        same_title = all(text_relation(analysis.books[competitors[0]].title, analysis.books[index].title) == 'exact'
                         for index in competitors)
        winner = max(competitors, key=lambda index: selected_scores[index]) if same_title else None
        for index in competitors:
            if index != winner:
                choices[index] = None
                methods[index] = 'new_provisional' if same_title else 'unresolved'
    return choices, methods


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
            readings[field] = preferred_reading([getattr(row, field) for row in evidence if getattr(row, field)])
        status = 'unresolved' if evidence and all(row.association_method == 'unresolved' for row in evidence) else 'provisional'
        result.append({'id': book.id, **readings, 'status': status,
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
        choices, methods = association_choices(frame, analysis, frames, observations, associations,
                                                image, EvidenceStore(database.parent / 'frames'))
        records = []
        assessor = QualityAssessor() if analysis.books else None
        for detection, matched, method in zip(analysis.books, choices, methods):
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
                                          association_method=method,
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
