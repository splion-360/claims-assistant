"""Remember today's price answers on disk so repeat lookups cost nothing.

Entries are keyed by source, query and market and only reused on the day they
were fetched, so a cached figure keeps its true retrieval time and is never stale
by more than a day. Transient failures are never cached; a definite "no listing"
answer is, so a missing price is not re-requested all day.
"""
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading

from .models import Price


class PriceCache:
    """A small SQLite store of the day's price answers."""

    MISS = object()

    def __init__(self, path: Path):
        """Open or create the cache file.

        Args:
            path: SQLite file path; its folder must exist.
        """
        self.path = Path(path)
        self._lock = threading.Lock()
        with self._connect() as connection:
            connection.execute('CREATE TABLE IF NOT EXISTS answers '
                               '(key TEXT PRIMARY KEY, day TEXT NOT NULL, price TEXT, reason TEXT)')

    def _connect(self) -> sqlite3.Connection:
        """Open a connection that waits briefly for other writers.

        Returns:
            A connection usable as a transaction context manager.
        """
        return sqlite3.connect(self.path, timeout=5)

    @staticmethod
    def _today() -> str:
        """Return today's UTC date as YYYY-MM-DD."""
        return datetime.now(timezone.utc).date().isoformat()

    @staticmethod
    def _key(parts: tuple) -> str:
        """Serialise a key tuple, ignoring case and surrounding spaces.

        Args:
            parts: Source, query fields and market.

        Returns:
            The stored key.
        """
        return json.dumps([str(part or '').strip().casefold() for part in parts])

    def _row(self, parts: tuple):
        """Fetch today's row for a key.

        Args:
            parts: Key tuple.

        Returns:
            (price JSON, reason) or None.
        """
        with self._lock, self._connect() as connection:
            return connection.execute('SELECT price, reason FROM answers WHERE key = ? AND day = ?',
                                      (self._key(parts), self._today())).fetchone()

    def get(self, parts: tuple):
        """Return today's answer for a key.

        Args:
            parts: Key tuple.

        Returns:
            A Price, None for a remembered "no price", or `PriceCache.MISS`.
        """
        row = self._row(parts)
        if row is None:
            return self.MISS
        return Price.model_validate_json(row[0]) if row[0] else None

    def reason(self, parts: tuple) -> str:
        """Return why a remembered "no price" answer had no price.

        Args:
            parts: Key tuple.

        Returns:
            The stored reason, or ''.
        """
        row = self._row(parts)
        return (row[1] or '') if row else ''

    def put(self, parts: tuple, price: Price | None, *, reason: str = '') -> None:
        """Store today's answer for a key.

        Args:
            parts: Key tuple.
            price: The price found, or None when the source definitely had none.
            reason: Why there was no price, shown again on a cached miss.
        """
        with self._lock, self._connect() as connection:
            connection.execute('INSERT OR REPLACE INTO answers VALUES (?, ?, ?, ?)',
                               (self._key(parts), self._today(),
                                price.model_dump_json() if price else None, reason))
