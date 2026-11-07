# Book-only similarity filter — 2026-11-06

The live diagnostic classifier now uses one rule: take the maximum similarity
across 16 book prompts and three crops; classify as `books` at or above 0.24,
otherwise `uncertain`. Empty-background prompts and the margin requirement have
been removed. This change does not wire relevance into Crusoe routing.

The prompts cover front/back covers, spines, open pages, notebooks, composition
notebooks, stacks, shelves, upright/flat placement, angled views, partial views,
occlusion, and a person holding a book. These are descriptions for pretrained
CLIP, not trained classes or a guarantee of configuration coverage. The INT8
image encoder and FP32 text encoder are unchanged. Prompts and the sole decision
threshold remain in root `config.yml`.

## Development comparison

We reused the corrected 89 confirmed positives from `recent-crop-audit` and the
63 confirmed negatives from `nonbook-crop-audit`. Six unresolved earlier frames
were excluded. These correlated captures are development data, not a held-out
validation set. The new prompt list was fixed before scoring; nine thresholds
from 0.22 through 0.30 were inspected, and 0.24 was selected after comparison.
No model training or fine-tuning occurred.

| Decision rule | Detected / 89 positives | False positives / 63 negatives | Recall | Sample precision |
| --- | ---: | ---: | ---: | ---: |
| Previous four prompts + background margin | 43 | 0 | 48.3% | 100% |
| Original four prompts, book-only threshold 0.24 | 51 | 2 | 57.3% | 96.2% |
| Expanded 16 prompts, threshold 0.22 | 88 | 48 | 98.9% | 64.7% |
| Expanded 16 prompts, threshold 0.23 | 74 | 20 | 83.1% | 78.7% |
| **Expanded 16 prompts, threshold 0.24** | **66** | **7** | **74.2%** | **90.4%** |
| Expanded 16 prompts, threshold 0.25 | 60 | 5 | 67.4% | 92.3% |
| Expanded 16 prompts, threshold 0.26 | 50 | 4 | 56.2% | 92.6% |

At 0.24, 23 positives remain below threshold, and 7/63 negatives (11.1%) pass.
Compared with the old margin rule, this recovers 23 positive frames at the cost
of seven false-positive frames. The 0.24 choice is the lowest inspected threshold
with at least 90% empirical precision in this sample. That is a development
tradeoff, not a promised production precision target; fresh captures are needed
before generalizing it. Rejecting `uncertain` still risks missing small books.

## Previously discussed misses

| Frame | New maximum book score | Result |
| --- | ---: | --- |
| 58 — large blurred foreground notebook | 0.2719 | Books |
| 56 — left-edge notebook | 0.2554 | Books |
| 63 — right-edge notebook | 0.2592 | Books |
| 80 — large partial notebook | 0.2620 | Books |
| 81 — large partial notebook | 0.2718 | Books |

Scores are cosine similarities, not probabilities. They are maxima across all
crops and prompts, so they are not directly interchangeable with the old
margin-selected crop scores. Adding prompts can increase both true detections
and false positives.

## Verification and artifacts

All 152 assessments from the final live implementation matched the experiment's
scores within 1e-6 and exactly matched its threshold decisions. The backend suite
passed all 116 tests, including threshold equality, abstention below threshold,
invalid score rejection, strongest-crop selection, configuration overrides,
logging, quality admission, and persisted metadata integration.

Private artifacts remain under `.claim-data/book-only-experiment/`:

- `prompts.json`: the candidate descriptions fixed before scoring.
- `results.json`: comparison configuration, per-image/per-prompt/per-crop scores,
  original-four-prompt maxima, winning prompts, labels, and threshold sweep.
- `live-verification.json`: final configuration snapshot and full live assessment
  metadata on every labelled image.
- `run.py`: local replay of the fixed manifests and threshold sweep.

Run from the repository root:

```sh
PYTHONPATH=. .venv/bin/python .claim-data/book-only-experiment/run.py
```

The script uses the existing labelled cohort outputs and cached pinned model
weights. It overwrites `results.json`; preserve that artifact before a new
comparison. Images and SQLite rows are unchanged, and no image was sent to a
remote inference service. Historical evaluation tools use current configuration
on rerun; older reports require their recorded configuration/commit to reproduce.

## Subsequent background diagnostics and remote admission

Ten descriptive empty-background prompts were added after this experiment for
logging only, preserving the book threshold at 0.24. A replay of all 152 labelled
images confirmed identical book decisions and scores within 1e-6. The current
pipeline gates automatic Crusoe camera calls on same-frame quality/relevance and
retains existing rate/freshness constraints. Exact replay output is retained at
`.claim-data/book-only-experiment/background-filter-verification.json`. This is
local replay plus mocked-provider integration verification, not a live paid API
load test. The comparison table above remains applicable to book admission.
