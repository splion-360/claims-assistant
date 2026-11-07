# Experiment configuration

[`config.yml`](../config.yml) is the source of tuning defaults. Python modules
import the validated settings from `library_claim.config`. The browser receives
only `capture.browser` through `/api/health`; it has no duplicate camera defaults.

| Section | Controls |
| --- | --- |
| `relevance` | Local model/revision, artifact hashes, prompts, similarity threshold, preprocessing, live crop count, CPU threads |
| `identification` | Maximum detections per frame, geometric book-association IoU/margin, mapping-chain bound |
| `quality` | Assessment resolution, brightness cutoffs, BRISQUE assets/ranking scale, relevance quality gates |
| `tracking` | SIFT, FLANN, keyframe budget, Lucas–Kanade, RANSAC and geometric verification |
| `selection` | New coverage, detail gain, BRISQUE improvement, sharpness improvement |
| `capture` | Quality/logging/remote cadences, evidence freshness, browser resolution and JPEG quality |
| `scheduler` | Default worker count, pending budget and timeout |
| `inference` | Crusoe text/vision models, token budgets, temperature, Qwen thinking and request timeouts |
| `speech` | Deepgram STT/TTS models, speech request chunk length and timeouts |
| `experiments` | Offline comparison settings, including overlapping crop count |

For quieter console output, change `capture.summary_log_interval_s` (default 5
seconds). It controls the combined Rich table only; scoring, tracking, and image
saving retain their independent cadences. Debug logs retain per-frame details.

Change a value, restart the backend, and refresh the browser. Configuration is
loaded once per process; existing workers and cached prompt embeddings are not
mutated mid-sweep. CLI evaluation loads the current file each time it starts.
Missing/unknown keys, empty prompt groups, invalid ranges, and nonfinite numeric
values raise startup errors instead of silently using fallback settings.

Model IDs now come from YAML. The previous `CRUSOE_MODEL`, `CRUSOE_VISION_MODEL`,
`DEEPGRAM_STT_MODEL`, and `DEEPGRAM_TTS_MODEL` environment variables are no longer
read. `.env` retains API credentials and the optional `CRUSOE_BASE_URL` endpoint.
No credentials are exposed through the health response or config file.

The relevance classifier scores positive book prompts plus descriptive empty-background
prompts for diagnostics. It uses the maximum
cosine similarity across prompts and crops, with one `relevance.min_similarity`
threshold. Background scores never veto a passing book score; there is no margin setting. Below the
threshold the result is `uncertain`; quality-blocked/unavailable results also
remain uncertain with an explicit status. Scores are not probabilities. Editing
prompts or the threshold does not require retraining, but can change false-positive
rates. Only assessed `books` candidates authorize automatic Crusoe image reasoning.
`capture.remote_interval_s` and `capture.frame_max_age_s` bound dispatch and pending
frame lifetime; inference retains one pending candidate and one in-flight job.

For a different local model, use a compatible CLIP ONNX export. Update its ID,
revision, artifact paths/hashes, cache directory, and matching preprocessing and
tokenizer settings together, then run:

```sh
.venv/bin/python -m library_claim.capture.relevance setup
.venv/bin/python -m library_claim.capture.relevance evaluate .claim-data/frames
```

The model adapter still requires `pixel_values` → `image_embeds` and `input_ids`
→ `text_embeds`. Changing an ID alone cannot adapt a different architecture.
Settings such as image size and normalization describe the model contract and
should not be varied independently of compatible weights.

Quality and relevance metadata include a configuration SHA-256 digest so results
from different configurations can be distinguished. Keep the corresponding YAML
or its Git commit alongside an experiment report; the digest is not a snapshot.
Existing saved observations are historical records and are not reclassified by
editing the config.

Numerical invariants and transport contracts remain in code: pixel ranges,
coordinate dimensions, matrix singularity tolerances, image minimum dimensions,
audio sample formats/rates, protocol limits, and local-access checks. They are
not inference tuning parameters. Runtime constructor arguments (for example, a
scheduler timeout in a test) can explicitly override their YAML defaults.

## Provisional live quality gates

Before local relevance inference, reject a frame if any condition holds:

| Setting | Rejection condition |
| --- | --- |
| `quality.max_brisque` | BRISQUE >60 |
| `quality.min_sharpness` | Laplacian variance <20 |
| `quality.max_extreme_fraction` | Dark fraction >0.80 OR bright fraction >0.80 |
| `quality.combined_brisque_threshold`, `quality.combined_sharpness_threshold` | BRISQUE >45 AND sharpness <50 |

Dark pixels have intensity ≤16 and bright pixels ≥239. Missing BRISQUE does not
invent a pass/fail score; exposure and sharpness still apply. A surviving frame
must then meet book similarity ≥0.26. These are initial whole-frame heuristics,
not calibrated book-readability guarantees. Restart the backend after editing.
