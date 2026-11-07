"""Summarize camera diagnostics in a compact, rate-limited Rich table."""
from datetime import datetime
import logging

from rich import box
from rich.console import Console
from rich.table import Table

from library_claim.capture.quality import QualityAssessment
from library_claim.capture.relevance.classifier import RelevanceAssessment
from library_claim.capture.tracking.tracker import TrackingResult
from library_claim.config import CONFIG
from live_demo.camera_quality import logger


class CameraSummary:
    """Accumulate evidence counts and retain explicitly dated relevance results."""

    def __init__(self):
        """Initialize an independent summary for one camera activation."""
        self.last_print_at = float('-inf')
        self.saved_count = 0
        self.relevance = None
        self.relevance_at = None
        self.console = Console(width=96, color_system=None, highlight=False)

    def log(self, tracking: TrackingResult, timestamp: float, *,
            quality: QualityAssessment | None, quality_at: float,
            relevance: RelevanceAssessment | None, saved: bool) -> None:
        """Print a periodic summary while counting saves between updates.

        Args:
            tracking: Association result for the latest completed frame.
            timestamp: Monotonic capture time in seconds.
            quality: Most recent measured image quality, if available.
            quality_at: Monotonic time of that quality measurement.
            relevance: Classification of this frame, or None if skipped.
            saved: Whether this frame was persisted.
        """
        self.saved_count += int(saved)
        if relevance is not None:
            self.relevance, self.relevance_at = relevance, timestamp
        if timestamp - self.last_print_at < CONFIG.capture.summary_log_interval_s or not logger.isEnabledFor(logging.INFO):
            return
        table = Table(title=f'Camera · {datetime.now():%H:%M:%S}', box=box.SIMPLE,
                      padding=(0, 1), show_edge=False)
        table.add_column('Signal', no_wrap=True)
        table.add_column('Result', no_wrap=True)
        table.add_column('Details')
        table.add_row('Tracking', tracking.status.replace('_', ' '), '')
        if quality is not None:
            signals = quality.signals
            brisque = 'N/A' if signals.brisque is None else f'{signals.brisque:.1f}'
            sharpness = 'N/A' if signals.sharpness is None else f'{signals.sharpness:.1f}'
            table.add_row('Quality', f'BRISQUE {brisque}',
                          f'Sharpness {sharpness} · {timestamp - quality_at:.1f}s ago')
        else:
            table.add_row('Quality', 'Not measured', '')
        # A skipped frame must not make an older classification appear current.
        if self.relevance is not None:
            assessment = self.relevance
            label = assessment.label if assessment.status == 'assessed' else assessment.status
            table.add_row('Last relevance', label.replace('_', ' '),
                          f'{timestamp - self.relevance_at:.1f}s ago · {assessment.reason.replace("_", " ")}')
            if assessment.scores:
                table.add_row('Similarity', f'Books {assessment.scores["books"]:.3f}',
                              f'Background {assessment.scores["empty_background"]:.3f} (not probabilities)')
        else:
            table.add_row('Last relevance', 'Not assessed', '')
        table.add_row('Evidence saved', str(self.saved_count), 'Since previous summary')
        # Render through the existing logger so Uvicorn levels and handlers still apply.
        with self.console.capture() as output:
            self.console.print(table)
        logger.info('\n%s', output.get().rstrip())
        self.saved_count = 0
        self.last_print_at = timestamp
