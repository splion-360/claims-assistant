# Book relevance diagnostic — 2026-11-06

The reported overlap is reproducible. On a manually reviewed sample of 20 book
views and 20 no-book views, the live classifier's mean book similarity was 0.258
versus 0.242. Its book-score ranges overlapped. This is insufficient evidence for
reliable automatic routing, even though some book views were detected correctly.

## Sample and limits

The audit used original JPEG evidence, not screenshots of the app's captions.
Forty frames were selected by visual review before running comparisons. Positive
examples show a composition notebook and three other books at different angles;
negatives show the same person/room without a visibly presented book. The target
annotation does not claim that every tiny background shape is book-free.

The data is correlated: one room, one person, and repeated views of a few books.
It contains no representative empty-wall set or deliberately positioned edge-book
set. The labels were assigned by the assistant, not independently adjudicated.
These are exploratory comparisons, not held-out accuracy estimates or calibrated
probabilities. No thresholds were fitted and no model was trained.

Private labels, source-image paths, and per-frame/prompt scores remain in the
ignored `.claim-data/relevance-audit/` directory. No images were uploaded.

## Controlled comparisons

The original prompts and decision thresholds (similarity 0.22, margin 0.025) were
unchanged. The comparisons use the same pinned CLIP model/revision. Book detections
below mean the existing decision function emitted `books`, not merely that its
book score was larger than a negative example's score.

| Image encoder | Text encoder | Image input | Mean book score: book / no-book | Book detections | False book predictions |
| --- | --- | --- | --- | --- | --- |
| INT8 | INT8 | Full-frame padding (live) | 0.258 / 0.242 | 11/20 | 0/20 |
| INT8 | INT8 | Reference center crop | 0.253 / 0.224 | 12/20 | 0/20 |
| INT8 | FP32 | Full-frame padding | 0.261 / 0.222 | 12/20 | 0/20 |
| INT8 | FP32 | Reference center crop | 0.266 / 0.210 | 17/20 | 0/20 |
| FP32 | INT8 | Full-frame padding | 0.252 / 0.239 | 10/20 | 0/20 |
| FP32 | FP32 | Reference center crop | 0.262 / 0.202 | 18/20 | 0/20 |

INT8 image inference plus preprocessing took about 34–37 ms median per frame in
these runs; FP32 took about 53–55 ms. Text encoding happens at initialization,
so switching only text precision need not add recurring image-inference cost.
These timings exclude model loading, text initialization, tracking, and quality
assessment; they are not end-to-end latency guarantees.

The INT8-image/FP32-text/center-crop variant separated the book-score ranges on
this sample: book frames 0.230–0.294, negatives 0.196–0.226. Three positive views
still failed the existing within-frame book-versus-background margin. Perfect
ranking of this small sample does not imply perfect classification or
performance on a new room.

## Findings

1. **Image preprocessing contributes.** The live implementation pads the whole
   frame into a square. The model's reference preprocessing resizes the shortest
   edge and center-crops it. Padding preserves the edges but makes the presented
   book smaller and retains more background. Changing only this step improved
   the mean score gap from 0.016 to 0.029 on this sample.
2. **Text quantization contributes.** Changing only the text encoder to FP32,
   keeping INT8 vision and full-frame padding, increased the gap to 0.039.
   Replacing only the image encoder did not help. A larger image model is not
   justified by this diagnostic alone.
3. **Prompt wording has an effect, but is not a complete explanation.** On the
   original INT8/padded setup, `a photo of a person holding a book` had a mean
   gap of 0.031 versus 0.017 for `a photo of a book`. `a photo of a notebook`
   had only a 0.005 gap across this mixed-book sample. These exploratory prompts
   fit the capture setup and need broader validation before adoption.
4. **No basic channel/normalization/tokenization mismatch was found.** The live
   code converts BGR to RGB, rescales by 255, and uses the reference channel mean
   and standard deviation. BOS/EOS IDs and configured padding agree with the
   downloaded tokenizer configuration. FP32 padded-batch and individual-unpadded
   text embeddings were identical in the check. INT8 showed small differences
   (minimum cosine agreement 0.9986), so extra diagnostic prompts were encoded
   separately to avoid changing the baseline batch's quantization behavior.

A score around 0.9 is not the target. The useful outcome is separation of book
and negative examples at an acceptable miss rate, with uncertainty retained.

## Recommended next experiment

Keep INT8 image inference, use FP32 text embeddings computed once, and validate
reference-style cropping against a new set that deliberately includes edge books,
small/distant books, partial occlusion, shelves, empty backgrounds, and unrelated
objects. For edge coverage, compare overlapping crops rather than silently
switching to a center-only live view. Keep thresholds unchanged initially.

The live model, prompts, preprocessing, and thresholds have **not** been changed
by this audit. No alternate model architecture is needed to test this narrower
explanation first. If separation remains poor on the expanded set, compare a
second pretrained model before considering training or fine-tuning.

## Reproduce locally

The diagnostic harness requires local weights and an explicit labelled manifest:

```sh
.venv/bin/python -m library_claim.capture.relevance.audit \
  .claim-data/relevance-audit/labels.json \
  --full-precision-dir .cache/book-relevance-audit \
  --output .claim-data/relevance-audit/final.json
```

The manifest contains an `images` array with `path` and boolean `book_present`,
plus optional `candidate_book_prompts`. Paths are read relative to the command's
working directory. A clean checkout needs its own locally labelled captures.
The report embeds the config, digest, manifest, per-image/prompt scores, model
hashes, current-threshold outcomes, and latency statistics. The corresponding
images must be retained separately.

The baseline can be checked against the live model directly with the same
manifest. `book_score_ranges_overlap` reports this symptom without asserting an
arbitrary universal accuracy pass bar. The saved baseline is `baseline.json`;
the final precision/preprocessing comparison is `final.json`.

The full-precision comparison files were downloaded from the same model revision
`6b7c523703ae26e39be4b03774966f8571c7f523` into a separate cache:

| File | SHA-256 |
| --- | --- |
| `vision_model.onnx` | `fd6e1402a588279d1723c7534d4bcba5bc0b14b47dfab0e46f8c47b8270d7d40` |
| `text_model.onnx` | `3f6571f5bad13a97c469c1622e1cfc4d9aef78b79fdbfcff804ca357bfada8cc` |

Sources: [original model card](https://huggingface.co/openai/clip-vit-base-patch32),
[pinned preprocessing configuration](https://huggingface.co/Xenova/clip-vit-base-patch32/blob/6b7c523703ae26e39be4b03774966f8571c7f523/preprocessor_config.json),
[pinned tokenizer configuration](https://huggingface.co/Xenova/clip-vit-base-patch32/blob/6b7c523703ae26e39be4b03774966f8571c7f523/tokenizer_config.json),
[pinned ONNX exports](https://huggingface.co/Xenova/clip-vit-base-patch32/tree/6b7c523703ae26e39be4b03774966f8571c7f523/onnx).
