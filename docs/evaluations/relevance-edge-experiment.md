# Overlapping-crop experiment — 2026-11-06

The offline comparison is implemented. Fresh edge-positioned captures are pending;
the first run below reuses the original 40-frame diagnostic sample. It does not
validate real edge cases or expand the dataset.

## First comparison on the original sample

All variants keep the INT8 image encoder, original prompt wording, and thresholds
(similarity 0.22 and margin 0.025). FP32 text embeddings are computed once.

| Variant | Books detected | False book predictions | Median processing | 95th percentile |
| --- | --- | --- | --- | --- |
| Live INT8 text + full-frame padding | 11/20 | 0/20 | 35.7 ms | 39.8 ms |
| FP32 text + full-frame padding | 12/20 | 0/20 | 34.2 ms | 35.5 ms |
| FP32 text + center crop | 17/20 | 0/20 | 37.2 ms | 39.3 ms |
| FP32 text + three overlapping crops | 19/20 | 0/20 | 101.2 ms | 103.9 ms |

Overlapping crops recover two additional positives on this sample, at roughly
2.7 times the center-crop processing cost. These correlated captures show the
same person, room, notebook, and three other books. Zero false positives on
20 negatives is not a guarantee; inspecting more crops creates more chances for
a false positive. No live model, preprocessing, or routing change has been made.

The measured interval covers tensor preparation, image inference, similarity
scoring, and crop selection. It excludes JPEG decoding, loading models, initial
text encoding, tracking, quality assessment, persistence, and network work.

## Crop policy being tested

Three square crops cover the full long axis: left/center/right for a landscape
image, or top/center/bottom for a portrait image. Squares use the reference CLIP
resize and normalization. Square inputs are deduplicated to one crop. Aspect
ratios that cannot be covered by the chosen count are rejected rather than
silently leaving gaps. `experiments.edge_crop_count` in root `config.yml` controls
this offline experiment only.

Each crop is inferred separately, avoiding changes in INT8 scale caused by batch
composition. Scores stay paired within a crop. If any crop meets the existing
book decision thresholds, choose the qualifying crop with the largest
book-minus-background margin. Otherwise choose the largest margin for diagnostic
reporting. The report includes every crop box/score and the selected crop.
This policy tests book presence; a selected crop's background label does not
establish that the whole frame is empty.

## Pending fresh-capture comparison

The user opted to record actual images rather than run synthetic translations.
There were no additional saved JPEGs when this first comparison ran. The planned
new sample includes whole books near both frame edges, partial books, smaller or
distant books, and empty walls/shelves. Label the captures by visual inspection
before model scoring; record `group` values such as `left_edge`, `right_edge`,
`partial`, `small`, and `negative`. Keep fresh-sample results separate from the
original diagnostic cohort. Do not tune the threshold on these comparisons.

Only retained `.claim-data/frames/` images are available to this tool. A perfectly
featureless wall may not establish a tracking reference and therefore may not be
saved; an empty shelf or a wall corner supplies a more practical capture.

## Reproduce

```sh
.venv/bin/python -m library_claim.capture.relevance.edge_audit \
  .claim-data/relevance-audit/labels.json \
  --full-precision-text .cache/book-relevance-audit/text_model.onnx \
  --output .claim-data/relevance-edge-experiment/original-sample.json
```

The manifest contract matches the earlier audit: an `images` array containing
`path` and boolean `book_present`; `group` is optional. Paths are relative to the
working directory. The output records the exact manifest, settings, model hashes,
per-image scores/decisions/timings, per-crop boxes/scores, and group counts. Images
and manifests remain in ignored local storage.

See [the preceding precision/preprocessing audit](relevance-audit.md) for the
pinned model sources, weight hashes, and initial annotation limitations.
