# Experiment configuration

[`config.yml`](../config.yml) is the source of tuning defaults. Python modules
import the validated settings from `library_claim.config`. The browser receives
only `capture.browser` through `/api/health`; it has no duplicate camera defaults.

| Section | Controls |
| --- | --- |
| `identification` | Guidance cooldown/timeout/session limit, recapture attempt limit, maximum detections, geometry IoU/margin, partial/fuzzy text matching, bounded crop SIFT/RANSAC thresholds |
| `quality` | Assessment resolution, brightness cutoffs, BRISQUE assets/ranking scale, local exposure/sharpness gate |
| `tracking` | SIFT, FLANN, keyframe budget, Lucas–Kanade, RANSAC and geometric verification |
| `selection` | New coverage, detail gain, BRISQUE improvement, sharpness improvement |
| `capture` | Quality/logging/remote cadences, evidence freshness, browser resolution and JPEG quality |
| `scheduler` | Default worker count, pending budget and timeout |
| `inference` | Crusoe text/vision models and their fallback models (tried in order when a call fails), token budgets, temperature, Qwen thinking and request timeouts |
| `speech` | Deepgram STT/TTS models, speech request chunk length and timeouts |
| `drex` | Drex model, endpoint, client name, timeouts, decision probability thresholds and guidance cooldowns |
| `measurement` | Depth sampling inset/window and Depth Pro model (the AR depth-grid size is under `capture.browser`) |
| `pricing` | Default country, per-country currency and price providers, appraisal keywords and threshold, conversion markets, exchange-rate endpoint and request timeouts |
| `items` | Non-book detection on/off, per-frame cap, detection cadence, association overlap and the wording that releases a framed piece from appraisal |

For quieter console output, change `capture.summary_log_interval_s` (default 5
seconds). It controls the combined Rich table only; scoring, tracking, and image
saving retain their independent cadences. Debug logs retain per-frame details.

Change a value, restart the backend, and refresh the browser. Configuration is
loaded once per process; existing workers and cached prompt embeddings are not
mutated mid-sweep.
Missing/unknown keys, invalid ranges, and nonfinite numeric
values raise startup errors instead of silently using fallback settings.

Model IDs now come from YAML. The previous `CRUSOE_MODEL`, `CRUSOE_VISION_MODEL`,
`DEEPGRAM_STT_MODEL`, and `DEEPGRAM_TTS_MODEL` environment variables are no longer
read. `.env` retains API credentials and the optional `CRUSOE_BASE_URL` endpoint.
No credentials are exposed through the health response or config file.

A Drex frame check (`library_claim/decisions/frame_check.py`) decides which frames
go to spine reading: `drex.book_visible_min` is the probability a book must
reach. A frame whose main issue is in `drex.capture_defects` (blur, glare, dark) is read only
when its title looks legible (`drex.legible_min`); otherwise the agent asks for a better view. Drex is called at most once per `capture.quality_interval_s`, with one
request in flight and only the newest frame waiting.
`capture.remote_interval_s` and `capture.frame_max_age_s` bound dispatch and pending
frame lifetime; inference retains one pending candidate and one in-flight job.

Quality metadata includes a configuration SHA-256 digest so results
from different configurations can be distinguished. Keep the corresponding YAML
or its Git commit alongside an experiment report; the digest is not a snapshot.
Existing saved observations are historical records and are not reclassified by
editing the config.

Numerical invariants and transport contracts remain in code: pixel ranges,
coordinate dimensions, matrix singularity tolerances, image minimum dimensions,
audio sample formats/rates, protocol limits, and local-access checks. They are
not inference tuning parameters. Runtime constructor arguments (for example, a
scheduler timeout in a test) can explicitly override their YAML defaults.

## Local gate

Before a Drex frame check, reject a frame if either condition holds:

| Setting | Rejection condition |
| --- | --- |
| `quality.min_sharpness` | Laplacian variance <20 |
| `quality.max_extreme_fraction` | Dark fraction >0.80 OR bright fraction >0.80 |

Dark pixels have intensity ≤16 and bright pixels ≥239. These whole-frame checks
are instant and free; perceptual quality and book presence are decided by Drex.
Restart the backend after editing.

## Phone AR depth (Android)

On Android phones whose Chrome supports WebXR `immersive-ar` with `depth-sensing`
and `camera-access`, the camera button starts an AR session and sends a
`capture.browser.depth_grid_width` × `depth_grid_height` depth grid with every frame.
Saved frames keep it as `.claim-data/depth/<frame_id>.npz`, and books are measured
immediately (`scale_method: webxr_depth`). Other browsers use the normal camera;
their saved book frames are measured in the background with Depth Pro.

WebXR needs a secure context. Serve the phone over USB so it can use `localhost`:

```sh
adb reverse tcp:4177 tcp:4177
.venv/bin/python -m uvicorn live_demo.server:app --host 127.0.0.1 --port 4177
# then open http://localhost:4177 in Chrome on the phone
```

`adb reverse` is what makes the phone work at all, and not only for WebXR. The
backend answers requests from `127.0.0.1` and `localhost` only: CORS rejects any
other origin and the live WebSocket refuses any other client address. Forwarding
the port makes the phone's `localhost` the laptop's, which satisfies the secure
context, the origin check and the socket check together. A phone reaching the
laptop over Wi-Fi at `http://192.168.x.x:4177` is refused on all three counts,
and would get no camera or microphone permission without TLS in any case.

The AR depth path does not depend on `measurement.enabled`. A phone in an AR
session sends its depth grid with every frame, and those books are measured from
it immediately. `measurement.enabled` controls only the Depth Pro fallback used
for frames that arrive without a depth grid, such as a desktop browser or an
Android device whose Chrome lacks `depth-sensing`. Turn it on to measure those
too, at the cost of loading Depth Pro.

Capture settings are worth lowering for a phone on a slower link.
`capture.browser` holds `frame_width`, `jpeg_quality` and `frame_interval_ms`;
at their defaults a sweep pushes roughly 6 Mbit/s, which is free over USB.

While the camera is on the page holds a screen wake lock, so the phone does not
lock partway through a sweep, and takes it back when the tab becomes visible
again. A browser without the Wake Lock API still sweeps normally.

## Pricing locale

`pricing.markets` maps a country code to its currency and the price providers
tried for it, in order. Adding a country is a configuration entry, not a code
change; `pricing.default_country` picks the market used when a sweep has no
confirmed country. The same books can be priced for another market by passing a
country to `build_packet`, which is how the packet shows one sweep valued in two
locales.

Prices come only from a source that answers with a retrievable URL. The Google
Books provider sends the market's country code, so the figure and its currency
are that market's own. Set `GOOGLE_BOOKS_API_KEY` in `.env`; without it the
shared anonymous quota refuses most requests and the book is reported unpriced
in the review queue rather than given a figure with nothing behind it.

When no local price exists, `pricing.convert_from` names the markets tried next.
A figure taken from another market is converted at a dated European Central Bank
reference rate and labelled `converted`, keeping the original currency and the
rate's source on the line.

Books whose readings contain `pricing.appraisal_keywords` (signed, first
edition, rare and so on) are never priced automatically, and a figure above
`pricing.appraisal_threshold` is withheld for a human appraiser. Both outcomes
leave the money fields empty and put the line in the review queue.

## Non-book contents

Admitted frames are inspected for the room's non-book contents on a cadence of
`items.detection_interval_s`, not on every frame: furniture does not move, and a
second vision call per frame would double the sweep's cost. `items.enabled`
turns the pass off entirely, leaving the packet's item list empty.

Two sightings merge into one object only when they agree on category and their
boxes overlap by `items.association_min_iou` in a verified shared basis, the
same geometry rule the books use. Appearance alone never merges two records, and
one object cannot absorb two detections from the same frame. Sightings that
cannot be related leave separate records carrying a possible-duplicate reason.

Framed art and mirrors are never valued automatically. They are recorded as
needing appraisal unless the policyholder says the piece is a print, which is
what `items.print_statements` lists. Every other object needs a retail price
provider under `pricing.item_providers`; no keyless retail API exists for
furniture and electronics, so that list ships empty and those lines are reported
unpriced with the reason rather than carrying an invented figure.
