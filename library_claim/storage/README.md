# Capture persistence

This is an internal storage contract, not `claim_packet.json`. No database or
provider is started at import, and the existing application is not wired to it yet.

Use `Repository(path)` as a context manager. Create the parent directory first;
`.claim-data/` is the suggested ignored runtime location. SQLite migrations run
transactionally on open and use `PRAGMA user_version`. Foreign keys prevent
missing references and associations across sweeps. One connection belongs to one
thread; the eventual async scheduler must keep this synchronous work off its
latency-sensitive path. Job records persist state; they do not execute jobs or
claim work. Resume policy belongs to step 2.

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
