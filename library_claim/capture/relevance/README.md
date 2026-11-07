# Local book relevance

The live backend uses pretrained CLIP ViT-B/32 via the quantized ONNX export
`Xenova/clip-vit-base-patch32`, pinned to revision
`6b7c523703ae26e39be4b03774966f8571c7f523`. No training or fine-tuning is performed.
This is a diagnostic pilot: labels do not gate Crusoe, trigger speech, change
capture states, identify physical books, or alter evidence retention.

## Setup and use

From the project root:

```sh
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m library_claim.capture.relevance setup
```

Setup downloads about 156 MB into the ignored `.cache/book-relevance/` directory
and verifies SHA-256 checksums. Downloads use the pinned revision. The live path
never downloads weights or uploads frames to a classification service. Restart
the backend after installation; missing weights produce `status=unavailable`,
not a fabricated negative result. After a failure, retry by restarting the camera.

The backend prints lines of this form (illustrative values):

```text
Camera relevance: books | status=assessed | books=0.340 other_objects=0.230 empty_background=0.190 | 35.0 ms | reason=separated_match | frame=...
```

Scores are cosine similarities, **not confidence percentages**. A label is
`uncertain` when the best score is below 0.22 or its lead is below 0.025. These
thresholds and the fixed prompts are provisional, versioned heuristics; they
have not been calibrated on a representative book dataset.

The three scored classes are `books`, `other_objects`, and `empty_background`.
Other objects includes people in the camera view. Mixed scenes can still receive
an uncertain or incorrect label; a single scene label cannot prove book absence.

## Execution and persistence

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
- The text encoder runs once at process initialization to encode the fixed
  prompts. Its session is then released; image inference reuses a shared CPU
  session with two intra-op threads. Live elapsed time includes lazy load on the
  first successful call; later calls measure preprocessing and inference.
- Model work shares the bounded per-camera tracking worker. Frames arriving
  while it is busy are dropped rather than accumulated. First-load delay can
  cause a tracking gap and subsequent reacquisition.
- `Frame.relevance_metadata` stores exact-frame scores, label, method, model
  revision, thresholds, status, and elapsed time when that frame is saved.
  Unverified unsaved views are logged only. Old rows remain readable with a
  null relevance field. Labels never delete reference evidence.

Full-frame square padding preserves edge objects that CLIP's default center crop
could remove. RGB conversion, bicubic resize, and CLIP channel normalization are
applied before 224-pixel image inference. This padding differs from the original
preprocessor and requires evaluation. Small or occluded books can still be lost
at that resolution. Class scores are maxima over each class's four text prompts.

## Evaluation checkpoint

Run an offline evaluation without modifying saved frames or database records:

```sh
.venv/bin/python -m library_claim.capture.relevance evaluate .claim-data/frames
```

The JSON report defaults to `.claim-data/relevance-evaluation.json`. It contains
per-image scores and latency; images remain private and ignored by Git. This
command evaluates all supplied images directly, without the live quality or
novelty gates, to make model behavior inspectable.

On 2026-11-06, the 63 existing JPEGs produced:

| Measurement | Result |
| --- | --- |
| Model load and prompt encoding | 733 ms |
| Per-image latency, median | 34.0 ms |
| Per-image latency, 95th percentile | 34.7 ms |
| Other objects | 6 |
| Uncertain | 57 |
| Books / empty background | 0 / 0 |

These are measurements on this development machine, not latency guarantees.
A visual spot-check of 21 evenly sampled images showed people, mugs, and
obstructions, with no clear book-positive examples. The frames are unlabelled;
accuracy and book recall are **not established**. The high abstention rate means
this version is not ready to drive automatic routing.

Next validation needs labelled book close-ups, shelves, books near frame edges,
empty walls/shelves, non-book objects, and mixed scenes. Measure book recall,
false negatives, abstentions, and latency before changing prompts/thresholds or
enabling routing. Retain uncertain outcomes for review rather than treating them
as no-books. No training is required for that evaluation.

## Sources

- [Original CLIP model card and limitations](https://huggingface.co/openai/clip-vit-base-patch32)
- [Pinned ONNX export](https://huggingface.co/Xenova/clip-vit-base-patch32/tree/6b7c523703ae26e39be4b03774966f8571c7f523)
- [Original preprocessing configuration](https://huggingface.co/Xenova/clip-vit-base-patch32/blob/6b7c523703ae26e39be4b03774966f8571c7f523/preprocessor_config.json)
- [ONNX Runtime Python API](https://onnxruntime.ai/docs/api/python/api_summary.html)
