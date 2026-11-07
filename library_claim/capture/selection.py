"""Compare associated views to identify candidate evidence improvements."""
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

from .quality import QualityAssessment
from library_claim.config import CONFIG

from .tracking.tracker import TrackingResult

SETTINGS = CONFIG.selection


class EvidenceDecision(BaseModel):
    """Explain a heuristic evidence decision independently of semantic relevance."""

    model_config = ConfigDict(extra='forbid', frozen=True)
    action: Literal['analyze_candidate', 'skip', 'review_association']
    reason: str


def compare_evidence(current: TrackingResult, quality: QualityAssessment,
                     previous: TrackingResult | None = None,
                     previous_quality: QualityAssessment | None = None,
                     *, unresolved: bool = True) -> EvidenceDecision:
    """Compare a view against the last retained evidence for its reference.

    Args:
        current: Tracking output for the currently assessed frame.
        quality: Quality measurement of the entire current frame.
        previous: Last retained tracking result for this reference.
        previous_quality: Quality of the retained frame under the same settings.
        unresolved: Whether additional detail could resolve a known or unknown gap.

    Returns:
        A candidate-analysis, skip, or association-review decision. This is not
        a book relevance classifier or a guarantee of useful information.
    """
    if current.status == 'uncertain':
        return EvidenceDecision(action='review_association', reason='unverified_view')
    if previous is None or previous.reference_frame_id != current.reference_frame_id:
        return EvidenceDecision(action='analyze_candidate', reason='unmatched_reference')
    if quality.region is not None or (previous_quality and previous_quality.region is not None):
        return EvidenceDecision(action='review_association', reason='region_comparison_requires_target_mapping')
    unit = [(0, 0), (1, 0), (1, 1), (0, 1)]
    try:
        now = np.float32(current.map_polygon(unit))
        before = np.float32(previous.map_polygon(unit))
        area = cv2.contourArea(now)
        intersection, _ = cv2.intersectConvexConvex(now, before)
        new_fraction = 1 - min(1, intersection / area)
    except (ValueError, cv2.error, ZeroDivisionError):
        return EvidenceDecision(action='review_association', reason='invalid_footprint')
    if new_fraction > SETTINGS.new_coverage_fraction:
        return EvidenceDecision(action='analyze_candidate', reason='additional_coverage_candidate')
    if unresolved and previous_quality is not None:
        # A closer view covers less reference area with comparable analysis pixels.
        before_pixels = previous_quality.assessed_width_px * previous_quality.assessed_height_px
        now_pixels = quality.assessed_width_px * quality.assessed_height_px
        detail_gain = (now_pixels / area) / (before_pixels / cv2.contourArea(before))
        same_settings = (quality.method, quality.assessed_width_px, quality.assessed_height_px,
                         quality.dark_cutoff, quality.bright_cutoff) == (
            previous_quality.method, previous_quality.assessed_width_px,
            previous_quality.assessed_height_px, previous_quality.dark_cutoff, previous_quality.bright_cutoff)
        a, b = quality.signals, previous_quality.signals
        if detail_gain > SETTINGS.detail_gain:
            return EvidenceDecision(action='analyze_candidate', reason='more_pixel_detail_candidate')
        if same_settings and a.brisque is not None and b.brisque is not None and b.brisque - a.brisque > SETTINGS.brisque_improvement:
            return EvidenceDecision(action='analyze_candidate', reason='better_perceptual_quality_candidate')
        if same_settings and a.sharpness is not None and b.sharpness is not None and a.sharpness > max(SETTINGS.sharpness_floor, b.sharpness * SETTINGS.sharpness_gain):
            return EvidenceDecision(action='analyze_candidate', reason='sharper_view_candidate')
    return EvidenceDecision(action='skip', reason='no_measured_evidence_gain')
