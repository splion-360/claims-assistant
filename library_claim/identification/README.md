# Visual book observations

Approved camera frames now request individual book detections from Crusoe rather
than only a narrative caption. `BookFrameAnalysis` carries one normalized bounding
box per visible physical book, the visible view (spine/cover/pages/partial), literal
visible text, nullable title/author/publisher readings, readability issues, and an
explicit frame-enumeration completion flag. Frame count is derived from the list;
it is not a separate model-supplied number. Duplicate identical boxes and invalid
coordinates are rejected. An unreadable book stays in the list with null fields.

The model is instructed to enumerate individual spines, including flat or stacked
books, not count a whole unresolved stack as one item, and never infer hidden
books, titles, editions, prices, or metric measurements. Model output is still
fallible: validation establishes schema correctness, not ground-truth detection.
The configured model and its normal structured-output/token limits are reused.

## Persistence and association

Every accepted automatic inference frame has a saved original JPEG and a `Frame`
record, including book candidates whose scene tracking is uncertain. Results
verify that the JPEG hash matches that frame. Books, immutable `BookObservation`
rows, review issues, and a succeeded `identify` job are inserted in one transaction.
The job records the full extraction, model and configuration digest. Retrying a
saved frame returns the previous result, including legitimate zero-book results.
No SQL migration is necessary: new observation fields are optional JSON payload
fields, so previous evidence remains readable.

Association uses normalized boxes projected through saved frame-to-reference
homographies. Only a common verified coordinate basis, or identical JPEG evidence,
permits matching. The best match must have IoU at least 0.5 and exceed the next
physical-book candidate by at least 0.15. Two new detections competing for one old
book are not merged. Conflicting non-null title readings also prevent merging.
Same titles alone never merge physical copies. These bounds and the maximum
mapping-chain length are in the `identification` section of `config.yml`.

When identity cannot be established, a new provisional book record is created;
if earlier observations exist, it receives a possible-duplicate review issue.
The UI explicitly labels copy counts as provisional, because unresolved matches
can overcount. Geometry assumes a stable scene/approximately planar support;
moving a book, changing depth, or replacing a book at the same location can defeat
that assumption. These heuristics have not met the assignment's count pass bar.

Observed text remains visual evidence, not an accepted catalogue identity.
`Book.identification_status` stays unidentified. The inventory displays a reading
only when its non-null observations agree (ignoring case). It does not fabricate
missing values or report calibrated identification confidence.

## Book-region quality

Each detected box is assessed locally using the existing quality assessor.
`BookObservation.quality` and `quality_metadata` contain the crop's scores and
region coordinates. Too-small regions retain the observation with unmeasured
quality and a review reason. These measurements happen after the initial remote
call supplies the box; they do not retroactively filter that call or yet drive
follow-up capture requests. The pre-inference whole-frame gates remain active.

## Live output and verification

The browser shows a provisional book inventory plus per-capture readings. Counts
and boxes are available in `evidence_photos`; the existing download's evidence
manifest also includes them. The assignment's final itemized claim packet,
measurement, pricing, catalogue resolution, and targeted voice recapture remain
future increments. The locked sweep states are unchanged.

The tests cover real SQLite persistence, source-byte
identity, duplicate retries, zero detections, unknowns, shifted-frame mappings,
separate copies of one title, conflicting/ambiguous associations, tiny crops, and
the capture-to-UI path with a mocked provider. Browser tests verify escaping of
model-provided text. No paid provider call or ground-truth book-count evaluation
was performed in this implementation pass.

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_book_observations.py'
node tests/client-regressions.cjs
```

Restart the backend and refresh the browser. Show a stationary shelf with readable
spines, then an overlapping view; inspect the inventory and observation counts.
Check `.claim-data/claims.sqlite3` for `books`, `observations`, `review_issues`, and
`jobs`, with original images in `.claim-data/frames/`.
