"""Open SQLite databases and apply versioned capture-schema migrations."""
from pathlib import Path
import sqlite3


def connect(path: str | Path) -> sqlite3.Connection:
    """Open and initialize a capture database.

    Args:
        path: SQLite database file path; its parent directory must exist.

    Returns:
        An open connection owned by the caller.

    Raises:
        sqlite3.Error: Opening, configuring, or migrating the database fails.
        ValueError: The database schema is newer than the application.
    """
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
    """Apply pending schema migrations transactionally.

    Args:
        connection: An idle SQLite connection with no active transaction.

    Raises:
        ValueError: The database schema is newer than the application.
        sqlite3.Error: A migration fails; its transaction is rolled back.
    """
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
