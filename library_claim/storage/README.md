# Capture persistence

This is an internal storage contract, not `claim_packet.json`. No database or
provider is started at import. The live camera uses `LiveCaptureStore` to persist
selected reference and evidence frames.

Use `Repository(path)` as a context manager. Create the parent directory first;
`.claim-data/` is the suggested ignored runtime location. SQLite migrations run
transactionally on open and use `PRAGMA user_version`. Foreign keys prevent
missing references and associations across sweeps. One connection belongs to one
thread; the eventual async scheduler must keep this synchronous work off its
latency-sensitive path. Job records persist state; they do not execute jobs or
claim work. The scheduler in `../orchestration/` owns restart recovery and execution.

A book ID identifies a physical copy. A BookObservation links that ID to a frame
and a normalized image box. Observations and frames are append-only. A clearer
reading is another observation, not an overwrite. `Book` holds the accepted
identity snapshot; promoting an observation to that snapshot is a later
association/identification decision. Tracking IDs are intentionally not book IDs.
Quality fields are optional raw signals, not calibrated probabilities. Unknowns
are `None`, never fabricated zero measurements or prices.

Save image bytes with `EvidenceStore.save`, then create the Frame record using
its returned content-hash key. Identical bytes share a file, while distinct Frame
records preserve capture times. Reads verify hashes. The store checks file
signatures, not full decoding or dimensions; the capture boundary must validate
those before saving. Use a private application-owned evidence directory.

Files are written atomically before database insertion. Filesystem and SQLite
writes are not one transaction: a crash or failed insert can leave an unreferenced
file, but must not trigger deletion of a file another frame references. Orphan
cleanup and backup tooling are deferred. Repository.add(Frame) is a low-level
operation and does not itself check that the evidence file exists; the caller
must save evidence first. Back up the database and evidence directory together.

Frame payloads also retain selection reasons, optional measured quality, and
quality-method metadata. These optional fields keep older payloads readable.
Only IDs, foreign keys and job status are relational columns initially. Validated
Pydantic payloads retain evolving fields as JSON. Foreign-key columns cannot be
reassigned through update(). There is no automatic merge or cascade deletion.

The final packet will derive `books[].id`, shelf/position, identity and confidence
from Book records and `frame_ref` from a supporting saved frame. Appraisal is
separate internally from identification. Measurement, valuation, scale evidence,
accepted-fact provenance and final packet export are future increments; none is
inferred by this persistence layer.

Validation:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_library_storage.py'
```

## Live capture files

The default location is `.claim-data/` at the project root:

```text
.claim-data/
├── claims.sqlite3          # sweeps, frames, associations and other domain records
└── frames/
    └── <sha256>.jpg        # original browser JPEG bytes
```

The live sweep ID is the browser session ID. Each retained frame logs its absolute
file path, frame ID, and selection reason. Frame and association records commit in
one transaction. Referenced keyframes must already exist in the same sweep.
Repeated JPEG bytes share a file even if distinct captures have different IDs.
The directory is ignored by Git and is not exposed as a public static route.

New tracking reference views are retained. At the existing one-second quality
cadence, associated views can be retained for additional coverage, closer detail,
or measured quality improvements. If a reference was captured between quality
samples, a later scored view can establish its quality baseline. Intermediate,
unchanged and unverified views are not saved by this selector. References can
contain non-book objects until the relevance classifier is integrated.

Camera toggles and backend restarts do not delete saved evidence. Tracking still
reacquires its in-memory context; this does not restore the browser's session or
reload old descriptors. The stored frames remain inspectable through the
repository or a SQLite viewer. Automatic retention cleanup is not implemented.
Captured frames selected by a worker already in progress may finish saving after
camera shutdown; they remain historical evidence, not current guidance.

When local relevance is enabled, retained frames also include optional
`relevance_metadata` from that exact capture. It records model revision, method,
cosine scores, label, status, and elapsed time. These are diagnostic classifications,
not accepted book identities or calibrated confidence values.
