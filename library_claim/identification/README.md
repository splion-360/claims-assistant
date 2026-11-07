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

## Agreed assumption and next association design

For this prototype, assume a user owns one physical copy of each particular book.
Books and stacks may occupy different shelves or room locations. Different books
with similar titles remain distinct. If multiple copies are visible together,
preserve that direct evidence rather than collapsing them under this assumption.
Separately located identical copies may be undercounted under this simplification.

Association combines verified geometry, compatible title/author
readings, and visual appearance of book crops. Text similarity alone must not
merge records. Partial readings can support a clearer later reading; ambiguous
associations remain unresolved. The matching rules are described below.

## Persistence and association

Every accepted automatic inference frame has a saved original JPEG and a `Frame`
record, including book candidates whose scene tracking is uncertain. Results
verify that the JPEG hash matches that frame. Books, immutable `BookObservation`
rows, review issues, and a succeeded `identify` job are inserted in one transaction.
The job records the full extraction, model and configuration digest. Retrying a
saved frame returns the previous result, including legitimate zero-book results.
No SQL migration is necessary: new observation fields are optional JSON payload
fields, so previous evidence remains readable.

Association first checks normalized boxes in a verified shared coordinate basis
(or identical JPEG evidence). IoU must reach 0.5. Case/punctuation normalization,
ordered contiguous partial-title matching, and normalized Levenshtein similarity
replace exact-string disagreement. Missing readings are unknown; contradictory
titles or authors block matching. Short partial strings do not establish textual
compatibility. These are heuristic gates, not calibrated probabilities.

Without shared geometry, compatible title readings plus a same-view crop match
are required. Bounded SIFT features, a descriptor ratio test, unique destination
matches, and RANSAC verify correspondences. At least 10 inliers, 65% inlier fraction,
and 12% crop-area support on both sides are required. Missing/failed crop matching
leaves identity unsupported; it does not prove two books are different. SIFT
features are cached within a single analysis call. No new model or training is used.

The best eligible match must exceed the runner-up by 0.15. Two detections cannot
consume the same existing ID. Ambiguous competing matches remain unresolved. For co-visible exact-title copies,
the strongest match reuses the existing ID and the additional detections create
separate provisional copies; ties use detection order. This preserves observed
multiplicity without asserting which identical copy was previously seen. Separate
detections on a first sighting also create separate provisional copies.
An unresolved observation is persisted under a provisional Book record with
`association_method=unresolved`, shown separately in the UI and excluded from its
copy count. Raw database Book row count is therefore not a physical inventory count.
Unresolved records are not used as established matching targets. Automatic later
reconciliation of these records is deferred; their source evidence is retained.

Compatible fuller title/author/publisher readings can update display text without
erasing original readings. Fuzzy disagreements prefer the most frequently observed
normalized reading (earliest on a tie), not automatically the longest spelling.
All thresholds live in `identification` in `config.yml`.

Geometry assumes stationary books and approximately planar support. Replacing a
book at the same location, repeated designs, and tiny/blurred spines can defeat
these heuristics. Counts remain provisional. Existing saved records are not
rewritten or deduplicated by this change; validate using a new intake. No live
accuracy or assignment count-pass-bar claim is made.

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
