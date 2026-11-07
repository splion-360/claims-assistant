# Image quality assessment

`QualityAssessor` combines OpenCV diagnostics with its pretrained BRISQUE model.
It loads the bundled model once per instance and performs inference locally.
Install the project requirements; use only the headless contrib OpenCV package,
since installing multiple OpenCV distributions can conflict over the `cv2` module.

```sh
.venv/bin/python -m library_claim.capture path/to/frame.jpg
.venv/bin/python -m library_claim.capture path/to/frame.jpg --previous path/to/previous.jpg
```

For application use, call `QualityAssessor().assess(bgr_image)` with decoded uint8
BGR or grayscale pixels. Reuse the assessor across calls on one worker; avoid
concurrent access to the same OpenCV model object. Run this synchronous CPU work
off the async event loop when integrating it into capture. An optional normalized
`Box` selects a book or shelf crop. A previous image must match the current image
shape and channel layout.

| Signal | Interpretation |
|---|---|
| `sharpness` | Grayscale Laplacian variance; larger means more high-frequency detail, which can also arise from noise |
| `dark_fraction` | Fraction of grayscale pixels at or below 16 by default |
| `bright_fraction` | Fraction at or above 239; white surfaces can contribute, so this is not a glare diagnosis |
| `temporal_change` | Mean absolute pixel difference divided by 255; affected by movement, exposure, and scene changes, not a speed estimate |
| `brisque` | Raw pretrained perceptual-quality estimate; lower is better, not a readability probability |
| `brisque_quality_score` | `clip(1 - brisque / 100, 0, 1)` for higher-better ranking; not calibrated confidence |

The quality result also includes assessed dimensions, region, cutoffs, OpenCV
version, method identifier, unavailable-signal reasons, and elapsed milliseconds.
The existing `Quality` contract carries raw diagnostics into book observations.
The complete result can be retained in a processing-job result for provenance.
The legacy `glare_fraction` field remains unset because bright pixels alone do not
establish glare.

Frames/crops are reduced to a maximum edge of 640 pixels, preserving aspect ratio
and never upscaling. Inputs and assessed regions need at least 32 pixels per edge.
Compare sharpness and BRISQUE only under consistent preprocessing and similar
region content. Full-frame quality cannot establish whether a small spine is
readable. Selecting the same coordinates in consecutive frames does not establish
that the same object remains there; tracking is implemented in `tracking/` and keeps its own coordinate associations.

Constant images make BRISQUE statistics degenerate, so their score is absent with
an explicit reason. Model computation failures and nonfinite results likewise
remain unknown while other measurements survive. Raw out-of-range BRISQUE scores
are preserved, and only the convenience ranking score is clipped.

There is no combined pass threshold or automatic voice instruction yet. The
intensity cutoffs define descriptive pixel fractions, not routing thresholds.
Real sweep footage and ground truth are required before calibrating rejection or
guidance policies. The current browser path is unchanged; this increment delivers
the local assessor and a CLI for collecting measurements.

Validation:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_quality.py' -v
```

Tests exercise the real pretrained model on deterministic synthetic imagery,
blur sensitivity of the sharpness signal, exposure extremes, temporal change,
region preprocessing, invalid inputs, unavailable scores, and SQLite round trips.
These verify implementation behavior, not book-readability accuracy.

## Live camera logging

The browser's existing camera frames are scored locally at most once per second,
inside the single local tracking worker per WebSocket, with no frame backlog. CPU work runs
in a background thread. The previous scored frame supplies temporal change;
a camera mode change resets that history. This diagnostic path is independent of
remote vision requests and does not gate them yet.

Run from the repository root:

```sh
.venv/bin/python -m uvicorn live_demo.server:app --host 127.0.0.1 --port 4177 --log-level info
```

Open `http://127.0.0.1:4177`, connect the session and enable the camera. Look for
`Camera quality` multiline summaries in that terminal. Normal logs show raw
BRISQUE, sharpness, dark/bright pixel percentages, frame change, and elapsed time.
Full JSON with frame/session references is available with `--log-level debug`.
Logs exclude images and credentials.
The existing session setup still requires the configured Crusoe API key.

Tracking now receives up to five camera frames per second; see [tracking](tracking/README.md) for its independent cadence and limitations.

## Local relevance diagnostics

The live backend can label sampled candidate views using a pretrained local
CLIP model. See [setup, score semantics, and evaluation](relevance/README.md).
It logs books, empty background, or uncertain, and stores the
assessment with selected frames. This pilot does not yet gate remote inference.

## Experiment settings

Edit the root [`config.yml`](../../config.yml) for quality, tracking, geometry,
selection, relevance, scheduling, camera cadence, and model generation settings.
The shared loader validates the file at startup; modules import those settings.
See [configuration usage](../../docs/configuration.md) for reload and model setup.
