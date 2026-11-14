"""Persist book readings and associate physical copies through verified frame geometry."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import cv2
import numpy as np

from library_claim.capture.quality import QualityAssessor
from library_claim.capture.tracking.geometry import intersection_area, transform
from library_claim.config import CONFIG, CONFIG_DIGEST
from library_claim.domain.models import Book, BookObservation, Frame, FrameAssociation, ProcessingJob, Quality, ReviewIssue
from library_claim.measurement.spine import (DepthMap, book_dimensions, measure_spine,
                                             shelf_run_m, size_refusal)
from library_claim.pricing.storage import stored_valuations
from library_claim.decisions.drex import DrexError
from library_claim.decisions.same_book import SameBookCandidate, choose_same_book, crop_jpeg, same_book_probabilities
from library_claim.storage.repository import Repository
from library_claim.storage.evidence import EvidenceStore
from .matching import crop_features, visual_match, text_relation, preferred_reading, normalize
from .models import BookFrameAnalysis
from .review import identities, reconcile


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
    intersection = intersection_area(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - intersection
    return max(0.0, min(1.0, float(intersection / union))) if union > 0 else 0.0


def association_choices(frame: Frame, analysis: BookFrameAnalysis, frames: dict,
                        observations: list[BookObservation], associations: dict,
                        image: np.ndarray, evidence: EvidenceStore,
                        decide_same=same_book_probabilities) -> tuple[list[str | None], list[str]]:
    """Apply geometry/text vetoes in code, then let Drex decide among surviving candidates.

    Args:
        frame: Current saved source frame.
        analysis: Model's per-book detections.
        frames: Saved frames indexed by ID.
        observations: Previous observations in the sweep.
        associations: Saved mappings indexed by source frame.
        image: Current decoded source image.
        evidence: Original JPEG storage.
        decide_same: D2 decision `(detection_crop, detection, candidates) -> probabilities`.

    Returns:
        Existing book IDs and association methods, including unresolved decisions.
    """
    choices, methods, selected_scores = [], [], []
    features, images = {}, {}

    def source_image(frame_id):
        """Decode a saved frame once per call.

        Args:
            frame_id: Saved frame ID.

        Returns:
            The decoded image, or None if it cannot be decoded.
        """
        if frame_id not in images:
            data = evidence.read(frames[frame_id].evidence_key)
            images[frame_id] = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return images[frame_id]

    established = [old for old in observations if old.association_method != 'unresolved']
    for index, detection in enumerate(analysis.books):
        scores, routes, sources = {}, {}, {}
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
                        previous = source_image(old.frame_id)
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
                    scores[old.book_id], routes[old.book_id], sources[old.book_id] = score, method, old
            except (ValueError, OSError, cv2.error):
                continue
        # Drex compares crops of the strongest surviving candidates; no score margin decides identity.
        ranked = sorted(scores, key=scores.get, reverse=True)[:3]
        selected = None
        method = 'unresolved' if possible else 'new_provisional'
        if ranked:
            try:
                candidates = []
                for book_id in ranked:
                    old = sources[book_id]
                    previous = source_image(old.frame_id)
                    if previous is None:
                        raise ValueError('Candidate evidence cannot be decoded')
                    candidates.append(SameBookCandidate(book_id, crop_jpeg(previous, old.bbox), old.title, old.author,
                                                        old.view, scores[book_id], routes[book_id]))
                # Candidates already passed geometry or spine-image verification; an exact title makes them strong.
                strong = [text_relation(detection.title, item.title) == 'exact' for item in candidates]
                probabilities = decide_same(crop_jpeg(image, detection.bbox), detection, candidates)
                decision = choose_same_book(probabilities, strong)
            except (DrexError, ValueError, OSError, cv2.error):
                decision = 'unresolved'
            if decision == 'unresolved':
                method = 'unresolved'
            elif decision is None:
                method = 'new_provisional'
            else:
                selected = ranked[decision]
                # Record whether Drex confirmed alone or agreed with strong code evidence.
                method = 'drex' if probabilities[decision] >= CONFIG.drex.same_book_min else 'evidence_with_drex'
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


def _size_reason(evidence: list[BookObservation]) -> str:
    """Say why a book has no measured size, from its own sightings.

    Args:
        evidence: The book's visual observations.

    Returns:
        The newest recorded refusal, or a general reason when none was recorded.
    """
    for row in reversed(evidence):
        if row.size_reason:
            return row.size_reason
    return 'No depth was captured for this book'


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
    # Prices are fetched by a background job; the inventory reports what was saved.
    valuations = stored_valuations(repo, sweep_id)
    # Background measure jobs add sizes to observations saved before depth was available.
    measured = {}
    for job in repo.list_for_sweep(ProcessingJob, sweep_id):
        if job.task == 'measure' and job.result:
            measured.update(job.result['measurements'])
    observations = [row.model_copy(update=measured[row.id]) if row.id in measured and row.spine_height_cm is None else row
                    for row in observations]
    result = []
    roots, separate = identities(repo, sweep_id)
    for book in repo.list_for_sweep(Book, sweep_id):
        if roots[book.id] != book.id:
            continue
        rows = [row for row in observations if roots[row.book_id] == book.id]
        # Readings, counts and sizes come from camera evidence; catalogue and user rows are shown separately.
        evidence = [row for row in rows if row.provenance == 'visual']
        catalogue = next((row for row in reversed(rows) if row.provenance == 'catalogue'), None)
        readings = {}
        for field in ('title', 'author', 'publisher'):
            readings[field] = preferred_reading([getattr(row, field) for row in evidence if getattr(row, field)])
        dimensions = book_dimensions(evidence)
        status = 'unresolved' if evidence and all(row.association_method == 'unresolved' for row in evidence) else 'provisional'
        if any(book.id in pair for pair in separate) or len({row.book_id for row in evidence}) > 1:
            status = 'provisional'
        # Visual issues matter only while no sighting of the book is clean; one clear
        # reading settles it, and so does a catalogue match, which corroborates the
        # reading independently of how good the crop was.
        unclear = bool(evidence and all(row.visual_issues for row in evidence)) and catalogue is None
        needs_review = status == 'unresolved' or not readings['title'] or unclear
        # Those same defects stop being review reasons once a catalogue has confirmed the reading.
        settled = {issue for row in evidence for issue in row.visual_issues} if catalogue else set()
        result.append({'needs_review': needs_review, 'id': book.id, **readings, 'status': status,
                       'observation_count': len(evidence),
                       'spine_height_cm': dimensions[0] if dimensions else None,
                       'spine_thickness_cm': dimensions[1] if dimensions else None,
                       'scale_method': dimensions[2] if dimensions else None,
                       'excluded': book.excluded, 'needs_appraisal': book.needs_appraisal,
                       'catalogue': None if catalogue is None else {
                           'title': catalogue.title, 'author': catalogue.author, 'edition': catalogue.edition,
                           'isbn': catalogue.isbn, 'url': catalogue.catalogue_url,
                           'retrieved_at': catalogue.retrieved_at.isoformat() if catalogue.retrieved_at else None},
                       'valuation': valuations.get(book.id),
                       'member_ids': [key for key, root in roots.items() if root == book.id],
                       'frame_ids': list(dict.fromkeys(row.frame_id for row in evidence)),
                       'observations': [{'id': row.id, 'frame_id': row.frame_id, 'bbox': row.bbox.model_dump(),
                                         'title': row.title, 'issues': row.visual_issues,
                                         'sharpness': row.quality.sharpness} for row in evidence],
                       'review_reasons': list(dict.fromkeys(
                           [issue.reason for issue in issues if issue.book_id == book.id
                            and issue.status == 'open' and issue.reason not in settled]
                           + ([] if dimensions else [_size_reason(evidence)])))})
    for book in result:
        book['candidates'] = [other['id'] for other in result if other['id'] != book['id']
                              and not set(book['frame_ids']) & set(other['frame_ids'])
                              and frozenset((book['id'], other['id'])) not in separate
                              and text_relation(book['author'], other['author']) != 'conflicting'
                              and (text_relation(book['title'], other['title']) in ('exact', 'partial', 'fuzzy')
                                   or (min(len(normalize(book['title'] or '')), len(normalize(other['title'] or ''))) >= CONFIG.identification.text_min_characters
                                       and (normalize(book['title']) in normalize(other['title']) or normalize(other['title']) in normalize(book['title']))))]
        if book['candidates']:
            book['needs_review'] = True
    return result


def inventory_totals(books: list[dict]) -> dict:
    """Compute inventory totals in code from the inventory lines.

    Args:
        books: Inventory entries from `inventory`.

    Returns:
        Book counts and shelf run over books the user has not excluded.
    """
    kept = [book for book in books if not book.get('excluded')]
    identified = sum(1 for book in kept if book.get('title'))
    return {'book_count': len(kept), 'books_identified': identified, 'books_unidentified': len(kept) - identified,
            'shelf_run_m': round(shelf_run_m([book['spine_thickness_cm'] for book in kept
                                              if book.get('spine_thickness_cm')]), 3)}


def current_inventory(database: Path, sweep_id: str) -> list[dict]:
    """Read the provisional inventory in its own connection.

    Args:
        database: Capture SQLite path.
        sweep_id: Owning sweep.

    Returns:
        Inventory entries as produced by `inventory`.
    """
    with Repository(database) as repo:
        return inventory(repo, sweep_id)


def save_measurements(database: Path, sweep_id: str, frame_id: str, depth: DepthMap) -> None:
    """Measure a frame's saved spine observations from depth that arrived after extraction.

    Args:
        database: SQLite database holding the observations.
        sweep_id: Owning sweep.
        frame_id: Frame whose observations to measure.
        depth: Metric depth for that exact frame.
    """
    job_id = f'measure_{frame_id}'
    with Repository(database) as repo:
        try:
            repo.get(ProcessingJob, job_id)
            return
        except KeyError:
            pass
        measurements = {}
        for row in repo.list_for_sweep(BookObservation, sweep_id):
            if row.frame_id != frame_id or row.provenance != 'visual':
                continue
            size = measure_spine(row.bbox, row.view, row.visual_issues, depth)
            if size:
                measurements[row.id] = {'spine_height_cm': size.height_cm, 'spine_thickness_cm': size.thickness_cm,
                                        'scale_method': size.method}
        repo.add(ProcessingJob(id=job_id, sweep_id=sweep_id, frame_id=frame_id, task='measure', status='succeeded',
                               created_at=datetime.now(timezone.utc), model=depth.method,
                               configuration_version=CONFIG_DIGEST, result={'measurements': measurements}))


def save_analysis(database: Path, sweep_id: str, frame_id: str, analysis: BookFrameAnalysis,
                  jpeg: bytes, model: str, depth: DepthMap | None = None,
                  decide_same=same_book_probabilities) -> dict:
    """Append a frame's readings atomically and return its provisional inventory.

    Args:
        database: SQLite database containing the source frame.
        sweep_id: Owning capture session.
        frame_id: Exact saved frame inspected by the model.
        analysis: Validated structured extraction.
        jpeg: Original frame for local per-book quality measurements.
        model: Provider model used for extraction.
        depth: Metric depth for this frame; without it books are saved unmeasured.
        decide_same: D2 decision function, injectable for tests.

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
        # Only camera evidence takes part in physical-copy association.
        observations = [row for row in repo.list_for_sweep(BookObservation, sweep_id) if row.provenance == 'visual']
        associations = {row.frame_id: row for row in repo.list_for_sweep(FrameAssociation, sweep_id)}
        roots, _ = identities(repo, sweep_id)
        associated = [row.model_copy(update={'book_id': roots[row.book_id]}) for row in observations]
        # Drex calls happen here, before any write, so no SQLite write lock is held while waiting.
        choices, methods = association_choices(frame, analysis, frames, associated, associations,
                                                image, EvidenceStore(database.parent / 'frames'), decide_same)
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
            # Dimensions come only from measured depth, never from a model's guess.
            size = measure_spine(detection.bbox, detection.view, detection.issues, depth) if depth else None
            # A book left unmeasured records which evidence was missing, not a blanket reason.
            size_reason = size_refusal(detection.bbox, detection.view, detection.issues, depth)
            observation = BookObservation(sweep_id=sweep_id, book_id=book_id, frame_id=frame_id,
                                          bbox=detection.bbox, view=detection.view, visible_text=detection.visible_text,
                                          title=detection.title, author=detection.author, publisher=detection.publisher,
                                          quality=signals, quality_metadata=metadata, visual_issues=detection.issues,
                                          association_method=method,
                                          source_model=model, unresolved_fields=unresolved,
                                          spine_height_cm=size.height_cm if size else None,
                                          spine_thickness_cm=size.thickness_cm if size else None,
                                          scale_method=size.method if size else None,
                                          size_reason=size_reason)
            records.append(observation)
            for reason in reasons:
                records.append(ReviewIssue(sweep_id=sweep_id, book_id=book_id, field='visual_evidence', reason=reason))
        result = {'frame_id': frame_id, 'detected_count': len(analysis.books), 'complete': analysis.complete,
                  'analysis': analysis.model_dump(mode='json')}
        records.append(ProcessingJob(id=job_id, sweep_id=sweep_id, frame_id=frame_id, task='identify',
                                     status='succeeded', created_at=datetime.now(timezone.utc), model=model,
                                     configuration_version=CONFIG_DIGEST, result=result))
        repo.add_many(records)
        reconcile(repo, sweep_id, frame, image, database, decide_same)
        return {**result, 'inventory': inventory(repo, sweep_id)}
