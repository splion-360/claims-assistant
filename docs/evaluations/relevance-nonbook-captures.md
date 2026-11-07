# Three-crop false-positive check — 2026-11-06

The current classifier produced **0 book predictions in 63 book-free frames**:
62 were `uncertain` and 1 was `empty_background`. The observed false-positive
rate is 0/63 (0%). This is a correlated sample from one room, not evidence of a
zero false-positive rate in general. No configuration or routing changes were made.

## Cohort and labels

There were 166 newly saved JPEGs beyond the previous 278-image index and the
95-frame recent-capture cohort. All were inspected in chronological contact
sheets before inference. The earlier frames still contained foreground or
background books. Enlarged views of the removal transition established the
negative cohort as indices 103–165 inclusive, after the background stack was
removed. The preceding 103 frames were excluded from this negative-only check.

The user's statement that non-book frames had been gathered supports the visual
labels; the selected frames show no visible books. They include the person,
chair, bed, window, wall, and incidental room objects. They do not represent a
broad sample of other rooms or visually book-like objects. Repeated views are
correlated, and only frames retained by the existing capture pipeline are
available; these are not all camera frames.

## Method and results

The actual live `BookRelevanceClassifier.assess` implementation was replayed on
the fixed labelled cohort: three overlapping crops, INT8 image encoder, FP32
text encoder, similarity threshold 0.22, margin threshold 0.025. The configuration
digest matches the previous 95-frame experiment. No threshold fitting or remote
inference was performed.

| Final result | Count |
| --- | ---: |
| Books — false positives | 0 |
| Uncertain — abstentions | 62 |
| Empty background | 1 |
| Total confirmed negative frames | 63 |

All 63 passed the existing whole-frame quality admission checks. Book cosine
scores ranged from 0.1853 to 0.2541. The maximum was frame 103: books 0.2541,
background 0.2786, resulting in `uncertain` because the background lead of 0.0245
was below 0.025. Similarity above 0.22 alone is therefore not a positive prediction.

Median classification time was 101.3 ms, p95 104.8 ms, excluding model loading,
JPEG decoding, quality checks, tracking, storage, and remote inference.

Forty of these frames had persisted live relevance metadata: all 40 were
`uncertain`, with labels, scores, and configuration matching replay exactly.
The remaining 23 had no stored relevance assessment. Replay classifies every
selected image, whereas live sampling and tracking can skip assessment.

## Routing interpretation

A `books`-only gate would admit none of these 63 negative examples. A policy that
also forwards `uncertain` would admit 62/63, despite there being no positive book
classifications. This is a policy distinction, not a measured remote-call saving;
relevance still does not gate Crusoe in the current application.

The previous positive cohort had 43 detections among 89 confirmed positive
frames (48.3% recall). Combining that cohort with these 63 negatives gives 43
true positives and 0 observed false positives, hence 100% empirical precision
for this narrow combined sample under a books-only decision. The 46 missed
positives (45 abstentions and 1 background prediction) remain the limiting issue.
Six unresolved frames from that earlier experiment are excluded. These figures
should not be generalized beyond the correlated captures or used to justify
raising thresholds without further validation.

## Artifacts and reproduction

The frozen index, labels, full per-crop results, image and model hashes,
configuration snapshot, and live comparison remain in ignored local storage:
`.claim-data/nonbook-crop-audit/`. Original images and database rows were not
modified. Reproduce with configuration from commit `20f5093`:

```sh
PYTHONPATH=. .venv/bin/python .claim-data/nonbook-crop-audit/run.py
```

The command overwrites `results.json`; preserve it before comparing a different
configuration. It performs local inference only. See the
[positive-cohort report](relevance-recent-captures.md) for the corrected background
book labels and recall findings.
