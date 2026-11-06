CREATE TABLE frame_associations (
    id TEXT PRIMARY KEY,
    sweep_id TEXT NOT NULL REFERENCES sweeps(id),
    frame_id TEXT NOT NULL,
    reference_frame_id TEXT NOT NULL,
    data TEXT NOT NULL,
    FOREIGN KEY(frame_id, sweep_id) REFERENCES frames(id, sweep_id),
    FOREIGN KEY(reference_frame_id, sweep_id) REFERENCES frames(id, sweep_id)
);
CREATE INDEX frame_associations_frame ON frame_associations(frame_id);
