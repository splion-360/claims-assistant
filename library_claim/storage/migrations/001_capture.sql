CREATE TABLE sweeps (
    id TEXT PRIMARY KEY,
    data TEXT NOT NULL
);
CREATE TABLE frames (
    id TEXT PRIMARY KEY,
    sweep_id TEXT NOT NULL REFERENCES sweeps(id),
    data TEXT NOT NULL,
    UNIQUE(id, sweep_id)
);
CREATE TABLE books (
    id TEXT PRIMARY KEY,
    sweep_id TEXT NOT NULL REFERENCES sweeps(id),
    data TEXT NOT NULL,
    UNIQUE(id, sweep_id)
);
CREATE TABLE observations (
    id TEXT PRIMARY KEY,
    sweep_id TEXT NOT NULL REFERENCES sweeps(id),
    book_id TEXT NOT NULL,
    frame_id TEXT NOT NULL,
    data TEXT NOT NULL,
    FOREIGN KEY(book_id, sweep_id) REFERENCES books(id, sweep_id),
    FOREIGN KEY(frame_id, sweep_id) REFERENCES frames(id, sweep_id)
);
CREATE INDEX observations_book ON observations(book_id);
CREATE TABLE review_issues (
    id TEXT PRIMARY KEY,
    sweep_id TEXT NOT NULL REFERENCES sweeps(id),
    book_id TEXT NOT NULL,
    data TEXT NOT NULL,
    FOREIGN KEY(book_id, sweep_id) REFERENCES books(id, sweep_id)
);
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    sweep_id TEXT NOT NULL REFERENCES sweeps(id),
    book_id TEXT,
    frame_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('pending','running','succeeded','failed','cancelled')),
    data TEXT NOT NULL,
    FOREIGN KEY(book_id, sweep_id) REFERENCES books(id, sweep_id),
    FOREIGN KEY(frame_id, sweep_id) REFERENCES frames(id, sweep_id)
);
CREATE INDEX jobs_status ON jobs(sweep_id, status);
