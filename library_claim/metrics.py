"""Collect one sweep's latency per stage and paid usage, and estimate its cost.

Timings come from the same stage events the flow view shows; usage comes from
`telemetry.usage` events that the Drex client, the model provider and the price
sources emit. Cost is estimated only from unit prices in configuration; a source
without a configured price is listed as not estimated rather than guessed.
"""
import json
from pathlib import Path
import statistics
import threading

from library_claim.config import CONFIG


def unit_prices() -> dict[str, dict[str, float]]:
    """Return configured USD unit prices per source and counter.

    Returns:
        e.g. {'nimble': {'calls': 0.001}}; sources or counters left empty are omitted.
    """
    return {source: {counter: price for counter, price in counters.items() if price is not None}
            for source, counters in CONFIG.costs.items()}


def _path(database: Path, sweep_id: str) -> Path:
    """Locate a sweep's saved metrics.

    Args:
        database: Capture SQLite path.
        sweep_id: Sweep.

    Returns:
        The JSON file beside the database.
    """
    return Path(database).parent / 'metrics' / f'{sweep_id}.json'


def load_metrics(database: Path, sweep_id: str) -> dict | None:
    """Read a sweep's saved metrics summary.

    Args:
        database: Capture SQLite path.
        sweep_id: Sweep.

    Returns:
        The summary, or None when nothing was saved.
    """
    path = _path(database, sweep_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())['summary']
    except (ValueError, KeyError):
        return None


def _percentile(values: list[float], share: float) -> float:
    """Return a nearest-rank percentile.

    Args:
        values: Samples.
        share: 0..1.

    Returns:
        The percentile value.
    """
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(share * len(ordered)) - 1))]


class SweepMeter:
    """Thread-safe accumulator fed by telemetry events."""

    def __init__(self, sweep_id: str):
        """Start empty.

        Args:
            sweep_id: Sweep being measured.
        """
        self.sweep_id = sweep_id
        self._lock = threading.Lock()
        self._timings: dict[str, list[float]] = {}
        self._errors: dict[str, int] = {}
        self._usage: dict[str, dict[str, float]] = {}

    def record(self, event: dict) -> None:
        """Add one telemetry event.

        Args:
            event: A stage event from `telemetry.emit`, or a usage event.
        """
        with self._lock:
            usage = event.get('usage')
            if usage:
                counters = self._usage.setdefault(usage['source'], {})
                for key, value in usage.items():
                    if key != 'source' and isinstance(value, (int, float)):
                        counters[key] = counters.get(key, 0) + value
                return
            if event.get('phase') in ('done', 'error') and event.get('ms') is not None:
                self._timings.setdefault(event['stage'], []).append(float(event['ms']))
                if event['phase'] == 'error':
                    self._errors[event['stage']] = self._errors.get(event['stage'], 0) + 1

    def summary(self) -> dict:
        """Summarise latency, usage and estimated cost.

        Returns:
            {'latency_ms': {stage: stats}, 'usage': {source: counters}, 'cost_usd': {...}}.
        """
        with self._lock:
            timings = {stage: list(values) for stage, values in self._timings.items()}
            usage = {source: dict(counters) for source, counters in self._usage.items()}
            errors = dict(self._errors)
        latency = {stage: {'count': len(values), 'errors': errors.get(stage, 0),
                           'p50': round(statistics.median(values)), 'p95': round(_percentile(values, 0.95)),
                           'max': round(max(values)), 'total': round(sum(values))}
                   for stage, values in sorted(timings.items())}
        prices, by_source, missing = unit_prices(), {}, []
        for source, counters in sorted(usage.items()):
            rates = prices.get(source)
            if not rates:
                missing.append(source)
                continue
            by_source[source] = round(sum(counters.get(counter, 0) * rate for counter, rate in rates.items()), 6)
        return {'latency_ms': latency, 'usage': usage,
                'cost_usd': {'estimated_total': round(sum(by_source.values()), 6), 'by_source': by_source,
                             'not_estimated': missing,
                             'basis': 'usage counted during the sweep times unit prices in config.yml costs'}}

    def save(self, database: Path) -> None:
        """Write the raw counters and summary beside the database.

        Args:
            database: Capture SQLite path.
        """
        path = _path(database, self.sweep_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            raw = {'timings': self._timings, 'errors': self._errors, 'usage': self._usage}
        path.write_text(json.dumps({'raw': raw, 'summary': self.summary()}))

    @classmethod
    def load(cls, database: Path, sweep_id: str) -> 'SweepMeter':
        """Resume a meter from its saved counters, or start a new one.

        Args:
            database: Capture SQLite path.
            sweep_id: Sweep.

        Returns:
            The meter.
        """
        meter = cls(sweep_id)
        path = _path(database, sweep_id)
        try:
            raw = json.loads(path.read_text())['raw']
            meter._timings, meter._errors, meter._usage = raw['timings'], raw['errors'], raw['usage']
        except (OSError, ValueError, KeyError):
            pass
        return meter
