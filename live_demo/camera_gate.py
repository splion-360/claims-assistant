"""Reject frames whose exposure or sharpness already rules out reading spines."""
from library_claim.capture.quality import QualityAssessment
from library_claim.config import CONFIG


def local_gate(quality: QualityAssessment) -> str | None:
    """Apply instant whole-frame checks before spending a Drex request.

    Args:
        quality: Measurements from this exact frame.

    Returns:
        A failure reason, or None when the frame may be checked by Drex.
    """
    signals = quality.signals
    if (signals.dark_fraction or 0) > CONFIG.quality.max_extreme_fraction or (signals.bright_fraction or 0) > CONFIG.quality.max_extreme_fraction:
        return 'extreme_exposure'
    if signals.sharpness is not None and signals.sharpness < CONFIG.quality.min_sharpness:
        return 'insufficient_texture_or_sharpness'
    return None
