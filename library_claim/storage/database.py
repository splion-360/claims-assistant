"""Explicit, transactional schema migration; no database is opened at import."""
from pathlib import Path
import sqlite3


def connect(path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    try:
        connection.execute('PRAGMA foreign_keys = ON')
        connection.execute('PRAGMA busy_timeout = 5000')
        connection.execute('PRAGMA journal_mode = WAL')
        migrate(connection)
        return connection
    except Exception:
        connection.close()
        raise


def migrate(connection: sqlite3.Connection) -> None:
    migrations = sorted(Path(__file__).with_name('migrations').glob('*.sql'))
    latest = int(migrations[-1].name.split('_')[0])
    if connection.execute('PRAGMA user_version').fetchone()[0] > latest:
        raise ValueError('Database schema is newer than this application')
    for file in migrations:
        version = int(file.name.split('_')[0])
        connection.execute('BEGIN IMMEDIATE')
        try:
            current = connection.execute('PRAGMA user_version').fetchone()[0]
            if current < version:
                # Migration files contain simple DDL statements, not triggers.
                for statement in file.read_text().split(';'):
                    if statement.strip():
                        connection.execute(statement)
                connection.execute(f'PRAGMA user_version = {version}')
            connection.commit()
        except Exception:
            connection.rollback()
            raise
