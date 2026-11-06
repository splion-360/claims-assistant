"""Persist and retrieve typed library capture records in SQLite."""
from pathlib import Path
from typing import TypeVar

from library_claim.domain.models import (
    Book, BookObservation, Frame, ProcessingJob, Record, ReviewIssue, Sweep,
)
from .database import connect

T = TypeVar('T', bound=Record)
TABLES = {Sweep: 'sweeps', Frame: 'frames', Book: 'books',
          BookObservation: 'observations', ReviewIssue: 'review_issues', ProcessingJob: 'jobs'}


class Repository:
    """Own one connection per thread. Call close(), or use as a context manager."""

    def __init__(self, path: str | Path):
        """Open the database connection owned by this repository.

        Args:
            path: SQLite database path with an existing parent directory.
        """
        self.connection = connect(path)

    def __enter__(self):
        """Enter the repository context.

        Returns:
            This repository instance.
        """
        return self

    def __exit__(self, *args):
        """Close the connection when leaving the repository context.

        Args:
            *args: Exception details supplied by the context manager protocol.
        """
        self.close()

    def close(self):
        """Close the owned SQLite connection."""
        self.connection.close()

    def add(self, record: T) -> T:
        """Validate and insert a new record atomically.

        Args:
            record: A supported domain record to persist.

        Returns:
            The validated record.

        Raises:
            pydantic.ValidationError: Record fields violate the domain contract.
            KeyError: The record type has no registered table.
            sqlite3.IntegrityError: An ID is duplicated or an association is invalid.
        """
        record = type(record).model_validate(record.model_dump())
        table = TABLES[type(record)]
        values = {'id': record.id}
        for field in ('sweep_id', 'book_id', 'frame_id'):
            if hasattr(record, field):
                values[field] = getattr(record, field)
        if isinstance(record, ProcessingJob):
            values['status'] = record.status
        values['data'] = record.model_dump_json()
        with self.connection:
            self.connection.execute(
                f"INSERT INTO {table} ({','.join(values)}) VALUES ({','.join('?' for _ in values)})",
                tuple(values.values()),
            )
        return record

    def get(self, kind: type[T], identifier: str) -> T:
        """Retrieve a typed record by its identifier.

        Args:
            kind: Registered domain model class.
            identifier: Identifier of the record to retrieve.

        Returns:
            The validated stored record.

        Raises:
            KeyError: The model type is unsupported or the record does not exist.
            pydantic.ValidationError: Stored data violates the model contract.
        """
        row = self.connection.execute(
            f'SELECT data FROM {TABLES[kind]} WHERE id = ?', (identifier,)
        ).fetchone()
        if row is None:
            raise KeyError(identifier)
        return kind.model_validate_json(row[0])

    def update(self, record: T) -> T:
        """Update a mutable record while preserving its identity links.

        Args:
            record: Replacement contents for an existing mutable record.

        Returns:
            The validated replacement record.

        Raises:
            ValueError: Evidence is being modified or an identity link is changed.
            KeyError: The record type is unsupported or the record does not exist.
            pydantic.ValidationError: Replacement fields violate the model contract.
        """
        record = type(record).model_validate(record.model_dump())
        if isinstance(record, (BookObservation, Frame)):
            raise ValueError('Evidence records are immutable; append a new observation')
        with self.connection:
            old = self.get(type(record), record.id)
            for field in ('sweep_id', 'book_id', 'frame_id'):
                if getattr(old, field, None) != getattr(record, field, None):
                    raise ValueError(f'{field} cannot be reassigned')
            if isinstance(record, ProcessingJob):
                self.connection.execute('UPDATE jobs SET data = ?, status = ? WHERE id = ?',
                                        (record.model_dump_json(), record.status, record.id))
            else:
                self.connection.execute(f'UPDATE {TABLES[type(record)]} SET data = ? WHERE id = ?',
                                        (record.model_dump_json(), record.id))
        return record

    def observations_for(self, book_id: str) -> list[BookObservation]:
        """Retrieve a book's observations in insertion order.

        Args:
            book_id: Persistent physical-book identifier.

        Returns:
            Stored observations, or an empty list when none match.
        """
        rows = self.connection.execute(
            'SELECT data FROM observations WHERE book_id = ? ORDER BY rowid', (book_id,)
        )
        return [BookObservation.model_validate_json(row[0]) for row in rows]

    def list_for_sweep(self, kind: type[T], sweep_id: str) -> list[T]:
        """Retrieve records of one type for a sweep in insertion order.

        Args:
            kind: Registered domain model class associated with a sweep.
            sweep_id: Identifier of the owning sweep.

        Returns:
            Matching validated records.

        Raises:
            ValueError: The requested type is Sweep.
            KeyError: The requested model type has no registered table.
        """
        if kind is Sweep:
            raise ValueError('Use get() for a sweep')
        rows = self.connection.execute(
            f'SELECT data FROM {TABLES[kind]} WHERE sweep_id = ? ORDER BY rowid', (sweep_id,)
        )
        return [kind.model_validate_json(row[0]) for row in rows]
