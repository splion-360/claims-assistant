# Three-crop classification on recent captures — 2026-11-06

The current classifier detected the presented notebook in 43 of 75 visually
labelled book-positive frames (57.3%). The remaining 32 were `uncertain`.
No `books` predictions occurred in 16 frames labelled “notebook not presented.”
Those 16 are not yet confirmed book-free: small stacked background objects may
be books. Four additional frames have ambiguous foreground slivers. Frame-level
precision and false-positive rate cannot be established without resolving these
labels. No thresholds, prompts, routing, or live configuration were changed.

## Dataset and labelling

The cohort contains all 95 unique saved JPEGs absent from the prior 278-image
index, frozen before scoring in `.claim-data/recent-crop-audit/index.json`.
Assistant visual review of two contact sheets, enlarged ambiguous examples, and
an original-resolution frame preceded model scoring. Labels and groups are in
`labels.json`. A `true` label means a clearly visible presented notebook; `null`
means frame-level book absence is unresolved. Null labels are excluded from
recall and are not treated as true negatives.

The user was asked whether the stacked background objects are books; that
clarification is pending. The 75 positives are supported by the foreground
notebook regardless of the answer. The captures are correlated views of one
notebook in one room, with changes in lighting, blur, and position. Many partial
views are cut off by the image boundary; this does not independently validate
occlusion by other objects. Saved frames are selected by the capture pipeline,
so this sample cannot measure performance across all incoming camera frames.

## Fixed configuration and replay

Replay used the actual `BookRelevanceClassifier.assess` implementation, with
three overlapping crops, INT8 vision, FP32 text, similarity threshold 0.22, and
margin threshold 0.025. Both reported class scores come from the selected crop.
Model hashes, complete configuration, config digest, source image hashes, crop
coordinates, per-crop scores, decisions, and timing are retained in
`.claim-data/recent-crop-audit/results.json`.

All 95 frames passed the current whole-frame quality admission rules when
reassessed. This does not imply the notebook regions were sharp or readable.
Tracking, scheduling, and remote inference were not replayed. No images were
sent to a remote model.

## Final classifications

| Visual group | Frames | Books | Uncertain | Empty background |
| --- | ---: | ---: | ---: | ---: |
| Other visible notebook views | 28 | 24 | 4 | 0 |
| Bottom-partial notebook | 8 | 6 | 2 | 0 |
| Left-edge partial notebook | 10 | 5 | 5 | 0 |
| Right-edge partial notebook | 23 | 7 | 16 | 0 |
| Dark, partial or blurred notebook | 6 | 1 | 5 | 0 |
| Notebook not presented; background unresolved | 16 | 0 | 15 | 1 |
| Ambiguous foreground slivers | 4 | 0 | 4 | 0 |
| **Total** | **95** | **43** | **51** | **1** |

On the confirmed positive subset, 43/75 are positive predictions and 32/75
(42.7%) are abstentions. If only `books` is forwarded, these abstentions become
operational false negatives for that routing policy. None of these 75 was
labelled `empty_background`. Of the 32 abstentions, 24 have `ambiguous_match`
and 8 have `weak_match`; similarity alone is not the final decision.

For example, frame 80 contains a partial notebook and scores books 0.2422 versus
background 0.2291. The 0.0131 margin falls below 0.025, so it is `uncertain`.
Frame 86, with the notebook lowered, scores books 0.2126 versus background
0.2104 and also abstains. A nonzero book score is not a book classification.

Median three-crop classification time was 102.3 ms; p95 was 105.1 ms. This covers
preprocessing, inference, and crop selection, excluding model initialization,
JPEG decoding, quality checks, tracking, persistence, and remote inference.

## Comparison with saved live observations

Of the 95 saved frames, 39 have live relevance metadata from the three-crop
version. The other 56 have no relevance assessment stored; saving a tracking
reference does not mean relevance was assessed. The 39 stored observations have
13 `books` and 26 `uncertain` labels. All 39 agree with replay exactly, including
scores and configuration digest. Details are in `live-comparison.json`.

A hypothetical `books`-only gate would admit 43/95 replayed frames (45.3%), or
13/39 of the already-assessed live observations (33.3%). These are sample counts,
not measured remote-call reductions: the current classifier remains diagnostic,
and scheduling, quality, novelty, and sampling also affect calls.

## Implication

This sample shows substantial abstention on visible books, especially narrow
edge views, rather than evidence of excessive positive classifications. Raising
the similarity threshold would not recover these missed books. First resolve
the background labels and test genuinely book-free scenes and other objects
before drawing precision conclusions or tuning thresholds. Retain uncertainty
as a distinct outcome when deciding how to route difficult views.

## Reproduce locally

The fixed manifest, images, script, and detailed outputs remain in ignored local
storage. From the repository root:

```sh
PYTHONPATH=. .venv/bin/python .claim-data/recent-crop-audit/run.py
```

This replays the saved cohort using the current configuration. To reproduce
these particular results, use configuration from commit `20f5093`; the output
also contains its full snapshot. The script overwrites `results.json`, so copy
that artifact first if comparing a later configuration. No claims database rows
or original JPEGs are changed.
