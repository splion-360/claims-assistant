# Drex decision layer and metric spine measurement — design

Date: 2026-11-13. Target: Drexathon (Nace AI), build window 2026-11-14 11:30–16:30.

## Intent

Finish capabilities 1 (live sweep agent) and 2 (books from the spine) of the
take-home brief, with Nace AI's Drex decision model making every decision in
the pipeline. Generative models remain only where text must be produced: Qwen
reads spines and phrases conversational replies; Deepgram handles speech.

The test space is a single shelf with about five books. The take-home pass bars
(60 books, two shelving units, Drex-vs-baseline comparisons) are not goals.

## Goals

1. **Live capture guidance (cap. 1).** The agent notices blur, glare, darkness,
   distance, occlusion and missed shelf areas, and says so while the user is still
   there.
2. **Voice corrections (cap. 1).** "Skip those, they're not mine", "that's a first
   edition", "that's signed", "no, that's *Dune*" change the inventory.
3. **Duplicate merging by Drex (cap. 2).** Drex decides whether a new detection is
   an already-known book.
4. **Catalogue resolution (cap. 2).** A legible title/author resolves to an Open
   Library work, and to an edition only when spine evidence supports it.
5. **Metric spine measurement (cap. 2).** Height and thickness in centimetres for
   every detected book, identified or not, plus total shelf run in metres, each
   with its scale method recorded.

## Non-goals

- Pricing, non-book items, room area, claim packet (capabilities 3–6).
- Drex-vs-old-method benchmarks or research write-ups.
- The `needs_appraisal` decision as a Drex call (D4 is dropped). It is set only by
  a user's "first edition"/"signed" statement through D5.

## Anti-goals (failure even if it "works")

- A title, author, edition, ISBN or dimension filled with no evidence behind it.
- **Measurements derived from anything the user says.** User statements never set
  dimensions or scale. A user's title correction is recorded as a
  `user_statement` and sent to review; it does not overwrite the visual reading.
- Asking the user to pull books out, hold books, scan barcodes or photograph books
  one at a time.
- Guidance that nags: repeated or overlapping spoken prompts.

## Constraints

- Phone: OnePlus 15R (Android, Chrome). Its WebXR depth support is unverified.
- Server: laptop with RTX 3050 Laptop GPU (4 GB), 16 cores, 27 GB RAM.
- Drex: `POST https://console.nace.ai/v1/systemone`, header
  `Authorization: Bearer $DREX_API_KEY` (already in `.env`), plus
  `X-Nace-Client: library-claim/1`. Model `drex-max-v1.1` (only Max models accept
  images, up to 8 per request). Question types: `noul` (yes/no → probability),
  `choice` (labels → probabilities), `score`. Responses contain `answers`,
  `usage.input_tokens`, `evaluation_time_ms`, `request_id`. Errors: 401, 402,
  422, 429/529 (honour `retry-after-ms`), 500 (retry once).
  Measured on 2026-11-13: one 1024×576 JPEG frame with three D1 questions →
  HTTP 200, 0.72 s round trip, 803 input tokens (576 image), correct answers.
- Project conventions in `agent_start.md` and `AGENTS.md` apply (module
  docstrings, Google-style docstrings, YAGNI/KISS/DRY, intent comments).
- All tuning values go in `config.yml` (new `drex` section), validated in
  `library_claim/config.py`.

## Architecture (agreed)

```
Strip 1 – live, per frame
Phone camera ─▶ local gate ─pass▶ ◆D1 frame check ─ok▶ SIFT/LK tracking ─new▶ evidence store ─▶ Strip 2
 (+ depth*)     sharpness/          (Drex Max, ≤1/s)                        (SQLite + JPEG)
                brightness ─fail▶ instant "slow down"      D1 issue ─▶ Strip 3 (D6)

Strip 2 – background, per saved frame
Qwen-VL spine boxes + text ─▶ scale (depth → cm) ─▶ ◆D2 same book? ─▶ Open Library ─▶ ◆D3 which match? ─▶ inventory UI

Strip 3 – live voice
Mic ─▶ Deepgram STT ─▶ ◆D5 intent ─▶ correction tools ─▶ Strip 2 records (+ invalidate context)
D1 issues, missing titles, user turns ─▶ ◆D6 speak now / which prompt ─▶ phrase ─▶ Deepgram TTS ─▶ speaker
```
\* Depth comes from WebXR when available, otherwise from Depth Pro on the server.

### Shared Drex client

New module `library_claim/decisions/drex.py`: one async function that sends
`state`, `questions` and optional `images`, returns typed answers, and logs
`request_id`, latency and input tokens. It raises a safe error (no response
bodies or keys) like `provider.InferenceError`. Callers treat a Drex failure as
"no decision" and fall back to the conservative outcome listed per decision
below; the live loop never crashes on a Drex error. Tests inject a fake client.

### D1 — frame check (replaces the CLIP relevance classifier)

- **Where:** inside `CameraTracking`'s admission path, where
  `CameraRelevance.assess` is called today. It runs for the same frames that are
  sent to CLIP today (frames with a selection reason, or uncertain tracking),
  after the local gate.
- **Local gate kept:** the existing extreme-exposure and minimum-sharpness checks
  in `camera_relevance.py`. BRISQUE no longer gates admission. `QualityAssessor`
  stays because evidence selection (`capture/selection.py`) compares BRISQUE and
  sharpness between views.
- **Must not block tracking.** At most one D1 request in flight and at most one per
  second (`capture.quality_interval_s`); a newer candidate replaces a waiting one.
- **Request:** the frame JPEG as one image; state = `"Live camera frame from a
  home-library inventory sweep."` Questions:
  - `book_visible` (noul): "Is at least one physical book visible in the image?"
  - `spine_legible` (noul): "Could a person read the title text on a visible book
    spine or cover in this image?"
  - `issue` (choice): none, blur, glare, dark, far, occluded, no_books (wording as
    in the 2026-11-13 smoke test).
- **Outcome:** `book_visible ≥ drex.book_visible_min` (start 0.5) → frame is a book
  candidate (replaces `label == 'books'`), offered to `CameraFilter` and sent to
  Qwen. `issue` and `spine_legible` go to D6.
- **Saving vs. admission (my call).** `Frame` records are immutable and tracking
  references must be saved immediately, so D1 gates only Qwen admission, not
  saving:
  - Frames with a tracking/selection reason are saved at once, as today. D1 runs
    afterwards on the saved bytes and decides only whether they go to Qwen.
  - Frames whose only possible reason was `book_candidate` (uncertain tracking)
    are held in the single D1 slot and saved only if D1 says `book_visible`.
  - A frame displaced from the slot by a newer one is not sent to Qwen; it stays
    saved only if it had a tracking reason.
  - D1 results (model, request ID, probabilities, latency, tokens) are stored as a
    `frame_check` record linked to the frame ID, not in `Frame.relevance_metadata`.
    The builder chooses the record shape (a `ProcessingJob` with task
    `frame_check`, or a new record type).
- **On Drex failure:** frame is not sent to Qwen; D6 is not triggered.

### D6 — speak now, and which prompt (replaces fixed rules in `book_guidance.py`)

- **Triggered by:** a D1 result with `issue ≠ none/no_books`, a run of local-gate
  failures, a visible book with no title after Qwen extraction, a book box touching
  the frame edge that never later appears whole (missed area), or a sweep phase
  change.
- **State (JSON):** recent D1 issues with timestamps, local-gate failure streak,
  visible missing-title books, last prompt and when it was spoken, whether the user
  or agent is currently speaking, sweep phase.
- **Question:** `prompt` (choice): stay_quiet, slow_down, move_closer,
  reduce_glare, more_light, reposition (occlusion), pan_back (missed area),
  read_title (steady for a missing title).
- **Phrasing (builder's note, my call):** each label maps to a fixed sentence
  template, reusing the wording in `capture_instruction()`. Templates are instant
  and cannot invent facts; Qwen phrasing is used only for conversational replies.
- **Hard limits in code, not Drex (my call):** "never interrupt the user" always
  wins. Limits are split by prompt kind:
  - `read_title` keeps the existing `guidance_cooldown_s` (30 s) and
    `guidance_max_requests` (3 per sweep).
  - Capture-quality prompts (slow_down, move_closer, reduce_glare, more_light,
    reposition, pan_back) get a new `drex.quality_prompt_cooldown_s` (start 8 s)
    and no per-sweep cap; the same label is not repeated within
    `drex.same_prompt_cooldown_s` (start 20 s).
  - The instant local "slow down" from the gate shares the quality-prompt cooldown.
- **State machine (my call):** `read_title` enters the existing
  Approaching → Capturing → Resuming flow through `request_capture(book)`, with the
  target book chosen by the existing `GuidancePolicy` rule. All other labels are
  speech only and do not change the sweep phase. The fixed closing lines in
  `finish_capture` stay as they are and do not go through D6. Sweep phase changes
  are not a D6 trigger.
- **On Drex failure:** stay quiet.

### Metric scale and spine measurement

Scale is measured, never estimated by a language model and never taken from the
user. Each measurement records `scale_method`.

- **Primary — WebXR depth (`scale_method = "webxr_depth"`), if the 15R supports
  it.** The browser starts an `immersive-ar` session with `depth-sensing`
  (`cpu-optimized`), `camera-access` and `dom-overlay`. On each frame interval it
  reads the camera image (raw camera access), encodes the JPEG as today, and sends
  alongside it a downsampled depth grid in metres plus the camera intrinsics
  derived from the view's projection matrix. The server stores the depth grid and
  intrinsics as a sidecar to the saved frame. HTTPS requirement: serve the phone
  over `adb reverse tcp:<port> tcp:<port>` so `http://localhost` is a secure
  context. If the session is not supported, the existing `getUserMedia` path runs
  unchanged.
- **Fallback — Depth Pro on the server (`scale_method = "depth_pro"`).** For saved
  frames that have book detections but no WebXR depth, run Apple Depth Pro
  (`apple/DepthPro-hf` via transformers), which returns metric depth and focal
  length in pixels. Runs in the background scheduler, only on frames sent to Qwen.
  Fit on the 4 GB GPU is unverified: try fp16 on GPU, fall back to CPU.
- **Measurement (both sources):** for each spine-view detection, take the
  endpoints of the box's long axis and short axis at its centre, sample the median
  depth in a small window at each endpoint, back-project to 3-D camera
  coordinates with the intrinsics, and take the Euclidean distances. Long axis →
  `spine_height_cm`, short axis → `spine_thickness_cm`. Books with `view` other
  than spine, or with `cut_off`/`occlusion` issues, are not measured.
- **Aggregation:** per book, the median across its measured observations.
  Shelf run (m) = sum of book thicknesses (median per book) for non-excluded
  books.
- **Storage:** new optional fields on `BookObservation`
  (`spine_height_cm`, `spine_thickness_cm`, `scale_method`) — JSON payload fields,
  no SQL migration, as with earlier observation fields. Book-level values are
  computed in code when building the inventory.
- **Known limitation:** Qwen boxes are axis-aligned, so leaning books overstate
  height. Accepted for now.

### D2 — same book or separate (replaces the margin rule in `association_choices`)

- **Kept in code:** all hard blocks in `identification/storage.py`
  (`conflicting` title/author, `uncertain_book`, view mismatch without shared
  geometry, no geometry and no compatible title). These remain vetoes.
- **Drex decides among the survivors.** For each detection, the top 3 surviving
  candidates (by today's score) go into one request: images = the detection crop
  plus each candidate's best crop (≤ 4 images); state = readings, views, IoU or
  visual-match score per candidate. One `noul` question per candidate:
  "Is the book in image 1 the same physical book as the book in image *k*?"
- **Outcome:** highest probability ≥ `drex.same_book_min` (start 0.8) → reuse
  that ID (`association_method = "drex"`); all ≤ `drex.different_book_max`
  (start 0.2) → new provisional book; otherwise `unresolved` (review drawer, as
  today). The existing co-visible exact-title multiplicity rule stays.
- **Both merge paths (my call):** D2 replaces the margin rule in `save_analysis`
  and in `review.reconcile()`'s automatic fragment merging, so no heuristic margin
  makes merge decisions. Drex calls happen before `save_analysis` opens its
  database transaction (split candidate gathering from writing).
- **On Drex failure:** `unresolved`; fragments stay unmerged.

### D3 — catalogue match (new)

- **When:** a book has a title reading (from Qwen) and no catalogue match yet, or
  its best title reading changed.
- **Candidates:** Open Library `search.json` by title (+ author when read), top 5
  works. Generic User-Agent; no user details sent.
- **Question:** `match` (choice): one label per candidate (title, authors, first
  publication year) plus `none`. State = the spine readings (title, author,
  publisher, visible_text).
- **Outcome:** a candidate with probability ≥ `drex.catalogue_min` (start 0.7) →
  store the work key, catalogue URL and retrieval time on the book. Edition and
  ISBN are filled only when the spine's publisher reading matches the publisher of
  exactly one edition of that work (Open Library editions endpoint); otherwise
  they stay empty. `none` or below threshold → no match, review issue with reason.
- **On Drex or Open Library failure:** no match; retry on the next reading change.

### D5 — what the user meant (new voice corrections)

- **When:** every final user transcript during a sweep, before the conversation
  model replies.
- **Question:** `intent` (choice): exclude_books, mark_first_edition,
  mark_signed, correct_title, question_or_chat.
- **Target resolution:** "that/those" refers to books in the most recent saved
  frame with detections. If there is more than one, a second Drex `choice`
  question picks the book (labels = book IDs, described by left-to-right position
  and readings), or `all_visible` for exclusion. Confidence below
  `drex.target_min` (start 0.6) → the agent asks which book.
- **Effects:**
  - exclude_books → `Book.excluded = True` (the user decides ownership; it is
    logged as a `user_statement` observation).
  - mark_first_edition / mark_signed → `user_statement` observation and
    `needs_appraisal = True`.
  - correct_title → Qwen extracts the stated title (structured call); stored as a
    `user_statement` observation and a review issue. The visual reading is not
    overwritten.
  - question_or_chat → normal conversation path.
  - Every effect calls `SweepController.invalidate_context()`.
- **On Drex failure:** treat as question_or_chat.

## Removals

- From the live path: `CameraRelevance` CLIP call and its model loading. The
  `library_claim/capture/relevance/` package, its tests, the `relevance` config
  section and the `.cache/book-relevance` model are deleted once D1 works.
- Leftover insurance-intake code (`lookup_policy`, policy fields, `policies.py`,
  `policy_directory.py`) is removed last, only if time remains.

## Left to the builder

- Module layout beyond the Drex client, exact Drex wording beyond what is given,
  threshold tuning on the five test books, depth-grid resolution, how Depth Pro is
  loaded and cached, UI display of measurements and catalogue links, and test
  structure (unit tests with a fake Drex client; one opt-in live smoke script).

## Open items

- **OnePlus 15R WebXR depth support** — pending a test of the WebXR depth-sensing
  sample on the phone. If unsupported, Depth Pro becomes the only scale source and
  the browser capture code is unchanged.
