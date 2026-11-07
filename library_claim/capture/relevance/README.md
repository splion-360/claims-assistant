# Local book relevance

The live backend uses pretrained CLIP ViT-B/32 with an INT8 image encoder and FP32 text encoder from
`Xenova/clip-vit-base-patch32`, pinned to revision
`6b7c523703ae26e39be4b03774966f8571c7f523`. No training or fine-tuning is performed.
Local relevance now gates automatic Crusoe camera inference. Only assessed book
candidates proceed. Capture states and local evidence retention remain unchanged.

## Setup and use

From the project root:

```sh
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m library_claim.capture.relevance setup
```

Setup downloads about 344 MB into the ignored `.cache/book-relevance/` directory
and verifies SHA-256 checksums. Downloads use the pinned revision. The live path
never downloads weights or uploads frames to a classification service. Restart
the backend after installation; missing weights produce `status=unavailable`,
not a fabricated negative result. After a failure, retry by restarting the camera.

The backend includes the latest relevance result in a Rich camera summary every
five seconds by default. Its age is displayed because repeated views can skip
classification. Full per-frame relevance JSON is available at debug level.

All model choices, prompts, thresholds, and preprocessing values come from
[`config.yml`](../../../config.yml). Restart the backend after edits.

Scores are cosine similarities, **not confidence percentages**. The classifier
uses 16 positive descriptions covering covers, spines, open pages, notebooks,
stacks, shelves, upright/flat books, angles, partial views, and occlusion. No
prompt set guarantees coverage of every configuration.

The frame score is the maximum similarity over all configured book prompts and
all crops. A score at or above `relevance.min_similarity` (currently 0.26) produces
`books`; a lower score produces `uncertain`. Ten empty-background descriptions score bare walls, shelves, tables, ceilings,
floors, corners, and windowsills for diagnostics. They do not veto books or
participate in the threshold decision; there is no margin requirement. Uncertain means insufficient book evidence, not proof that
no books exist. The threshold is a provisional choice on a small development
sample; see the [book-only experiment](../../../docs/evaluations/relevance-book-only.md).

## Execution and persistence

Automatic inference starts after local tracking, sampled quality, and relevance
complete. A candidate must have same-frame `status=assessed` and `label=books`.
Below-threshold, quality-blocked, unavailable, and unassessed frames never authorize
remote inference. The full JPEG that passed is frozen with its frame ID, capture
time, and camera activation; later arrivals cannot substitute different bytes.

One pending approved frame is retained while the remote slot is busy or rate
limited; newer eligible evidence replaces it. Subsequent tracking completions
check that pending frame against the configured age limit (12 seconds), existing
remote interval (8 seconds), photo capacity, and camera activation. There is no
unbounded queue. A failed request may retry within freshness/rate limits; successful
identical bytes are not resent. Stopping/restarting the camera invalidates old
candidates, and results from a previous activation are not pinned.

The Rich summary shows the latest filter reason and activation totals for
submitted requests and rejected assessed frames. Unassessed/duplicate frames are
not included in the rejected count. Background similarity is explicitly labelled
diagnostic. A pending approved frame may still be sent while a newer below-threshold
frame is skipped; this always sends the original approved image.

- Tracking and quality assessment run first on the existing background worker.
- Only a frame with a fresh quality assessment is considered, at most once per
  second. New references and heuristic evidence gains are candidates.
- Unverified views are also considered: a featureless scene must not require a
  successful geometric association to reach the relevance stage.
- More than 95% near-black/near-white pixels, or Laplacian variance below 10,
  produces `quality_blocked` without loading or running the model. Low variance
  can mean a flat wall rather than blur, so no movement instruction is inferred.
- Verified duplicate views skip classification. Unverified repeats may still be
  sampled because tracking cannot establish their identity.
- The text encoder runs once at lazy process initialization to encode the configured
  prompts. Its session is then released; image inference reuses a shared CPU
  session with two intra-op threads. Live elapsed time includes lazy load on the
  first successful call; later calls measure preprocessing and inference.
- Model work shares the bounded per-camera tracking worker. Frames arriving
  while it is busy are dropped rather than accumulated. First-load delay can
  cause a tracking gap and subsequent reacquisition.
- `Frame.relevance_metadata` stores exact-frame scores, label, method, model
  revision, thresholds, configuration digest, status, and elapsed time when that frame is saved.
  Unverified unsaved views are logged only. Old rows remain readable with a
  null relevance field. Labels never delete reference evidence.

Three overlapping square crops cover the long image axis, controlled by
`relevance.crop_count` (default 3). Landscape crops are left/center/right and
retain the full height; portrait crops are top/center/bottom. Each crop is resized
to 224 pixels with CLIP normalization and inferred separately. Square inputs use
one unique crop; unsupported extreme aspect ratios fail explicitly.

The crop with the maximum book score supplies the reported frame score.
Metadata stores all crop boxes in original-frame pixels, their book scores, and
the selected zero-based crop index. The Rich table shows the book score, threshold,
one-based crop number, and total classification time. Old stored metadata remains
historical and is not rewritten. Small or occluded books may still be missed.
Crops never replace saved full-frame images or the images sent to Crusoe.

## Evaluation checkpoint

Run an offline evaluation without modifying saved frames or database records:

```sh
.venv/bin/python -m library_claim.capture.relevance evaluate .claim-data/frames
```

The JSON report defaults to `.claim-data/relevance-evaluation.json`. It contains
per-image scores and latency; images remain private and ignored by Git. This
command evaluates all supplied images directly, without the live quality or
novelty gates, to make model behavior inspectable.

Historical result, superseded by the book-only experiment: after removing the catch-all prompts on 2026-11-06, the same 63 existing JPEGs
produced the following results. The prior three-target evaluation is superseded.

| Measurement | Result |
| --- | --- |
| Model load and prompt encoding | 633 ms |
| Per-image latency, median | 34.5 ms |
| Per-image latency, 95th percentile | 39.7 ms |
| Uncertain | 63 |
| Books / empty background | 0 / 0 |

These are measurements on this development machine, not latency guarantees.
A visual spot-check of 21 evenly sampled images showed people, mugs, and
obstructions, with no clear book-positive examples. The frames are unlabelled;
accuracy and book recall are **not established**. The high abstention rate means
this version is not ready to drive automatic routing.

Next validation needs labelled book close-ups, shelves, books near frame edges,
empty walls/shelves, non-book objects, and mixed scenes. Measure book recall,
false negatives, abstentions, and latency before further changing prompts/thresholds. Retain uncertain outcomes for review rather than treating them
as no-books. No training is required for that evaluation.

## Sources

- [Original CLIP model card and limitations](https://huggingface.co/openai/clip-vit-base-patch32)
- [Pinned ONNX export](https://huggingface.co/Xenova/clip-vit-base-patch32/tree/6b7c523703ae26e39be4b03774966f8571c7f523)
- [Original preprocessing configuration](https://huggingface.co/Xenova/clip-vit-base-patch32/blob/6b7c523703ae26e39be4b03774966f8571c7f523/preprocessor_config.json)
- [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)

## Labelled diagnostic follow-up

A controlled book/no-book comparison now exists: see the
[2026-11-06 audit](../../../docs/evaluations/relevance-audit.md). It isolates
preprocessing, text precision, and image precision without changing live settings.

The [overlapping-crop follow-up](../../../docs/evaluations/relevance-edge-experiment.md)
records the comparison that preceded enabling the three-crop version for live testing.
