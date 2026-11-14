CREATE TABLE items (
    id TEXT PRIMARY KEY,
    sweep_id TEXT NOT NULL REFERENCES sweeps(id),
    data TEXT NOT NULL
);
CREATE TABLE item_observations (
    id TEXT PRIMARY KEY,
    sweep_id TEXT NOT NULL REFERENCES sweeps(id),
    item_id TEXT NOT NULL REFERENCES items(id),
    frame_id TEXT NOT NULL REFERENCES frames(id),
    data TEXT NOT NULL
);
CREATE INDEX items_sweep ON items(sweep_id);
CREATE INDEX item_observations_sweep ON item_observations(sweep_id);
CREATE INDEX item_observations_item ON item_observations(item_id);
PRAGMA user_version = 4;
