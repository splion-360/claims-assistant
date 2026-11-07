CREATE TABLE book_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sweep_id TEXT NOT NULL REFERENCES sweeps(id),
    source_id TEXT NOT NULL REFERENCES books(id),
    target_id TEXT NOT NULL REFERENCES books(id),
    action TEXT NOT NULL CHECK(action IN ('same', 'separate')),
    origin TEXT NOT NULL CHECK(origin IN ('user', 'automatic')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
PRAGMA user_version = 3;
