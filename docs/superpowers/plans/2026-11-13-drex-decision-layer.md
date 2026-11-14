# Drex Decision Layer and Metric Spine Measurement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish capabilities 1 and 2 with Nace Drex making every pipeline decision and books measured in centimetres from depth.

**Architecture:** One synchronous Drex client (`library_claim/decisions/drex.py`) is called from worker threads (`asyncio.to_thread`) by five decision modules: D1 frame check, D6 guidance prompt, D2 same-book, D3 catalogue match, D5 correction intent. Metric scale comes from a per-frame depth map (WebXR sidecar or Depth Pro), turned into spine centimetres by `library_claim/measurement/spine.py`. Qwen stays only for spine text extraction, stated-title extraction and conversational replies.

**Tech Stack:** Python 3, FastAPI, httpx, pydantic v2, OpenCV, numpy, SQLite; Drex `drex-max-v1.1`; Open Library; transformers + torch (Depth Pro); browser WebXR (`immersive-ar`, `depth-sensing`, `camera-access`, `dom-overlay`).

**Spec:** `docs/superpowers/specs/2026-11-13-drex-decision-layer-design.md`

## Global Constraints

- Drex endpoint `POST https://console.nace.ai/v1/systemone`; headers `Authorization: Bearer $DREX_API_KEY`, `X-Nace-Client: library-claim/1`; model `drex-max-v1.1`; at most 8 images per request.
- Drex errors: 429/529 → wait `retry-after-ms` (fallback 1000 ms) and retry once; 500 → retry once; 401/402/422 → no retry. Errors never include keys or response bodies.
- Every Drex caller has a conservative failure outcome (listed per task); the live loop never raises on a Drex error.
- User statements never set dimensions or scale. A stated title never overwrites a visual reading.
- No field is filled without evidence: titles/editions/ISBNs only from spine readings or a Drex-accepted catalogue match; dimensions only from depth.
- Thresholds (all in `config.yml` → `drex` section): `book_visible_min 0.5`, `same_book_min 0.8`, `different_book_max 0.2`, `catalogue_min 0.7`, `target_min 0.6`, `quality_prompt_cooldown_s 8`, `same_prompt_cooldown_s 20`, `gate_failure_streak 3`.
- Conventions from `agent_start.md` / `AGENTS.md`: one-line module docstring, Google-style docstrings on every function including tests, intent comments, YAGNI/KISS/DRY.
- Tests use `unittest`. Run all with `.venv/bin/python -m unittest discover -s tests`. No test makes a network call; Drex and Open Library are faked with `httpx.MockTransport` or injected callables.
- Out of scope: pricing, non-book items, room area, claim packet, Drex-vs-baseline benchmarks, a Drex appraisal decision (D4).

## Review Focus

1. Drex returns 402 (credit exhausted) mid-sweep → camera keeps tracking, no frames reach Qwen, agent stays quiet, one error line in the UI, no crash. Test in Task 3.
2. The user talks while a D6 prompt would fire → nothing is spoken. Test in Task 5.
3. A spine box whose edges overlap the background (depth jumps to a far wall) → measurement uses inset endpoints and is not inflated. Test in Task 6.
4. A frame with no depth available and Depth Pro failing to load → books are saved without dimensions and a review reason "No metric depth for this frame"; no crash. Test in Task 7.
5. "Skip those" with no saved frame in the last 30 s → nothing is excluded and the agent asks which books. Test in Task 11.

---

### Task 1: Drex client and configuration

**Files:**
- Create: `library_claim/decisions/__init__.py`, `library_claim/decisions/drex.py`, `scripts/drex_smoke.py`
- Modify: `library_claim/config.py` (add `DrexConfig`, `MeasurementConfig`, register in `AppConfig`), `config.yml`, `.env.example` (add `DREX_API_KEY=`), `docs/configuration.md` (two table rows)
- Test: `tests/test_drex.py`, `tests/test_config.py` (extend)

**Interfaces:**
- Produces:
  - `class DrexError(RuntimeError)` — safe message only.
  - `class Answer(BaseModel)`: `type: Literal['noul','choice','score']`, `noul: float | None`, `choice: str | None`, `probabilities: dict[str, float]`, `confidence: float | None`.
  - `class DrexResult(BaseModel)`: `model: str`, `answers: dict[str, Answer]`, `request_id: str | None`, `input_tokens: int`, `latency_ms: float`.
  - `def noul(instructions: str) -> dict` and `def choice(instructions: str, criteria: dict[str, str]) -> dict` — question builders.
  - `def decide(questions: dict[str, dict], *, state: object, images: list[bytes] | None = None, client: httpx.Client | None = None) -> DrexResult` — images are JPEG bytes, sent as `data:image/jpeg;base64,...`.
  - `CONFIG.drex`: `model, base_url, client_name, timeout_s, connect_timeout_s` + the thresholds in Global Constraints.
  - `CONFIG.measurement`: `depth_inset 0.15`, `depth_window_px 5`, `depth_pro_model "apple/DepthPro-hf"`, `depth_grid_width 96`, `depth_grid_height 54`.

- [ ] **Step 1: Write failing tests** in `tests/test_drex.py`:
  - `test_decide_sends_images_and_parses_answers`: MockTransport asserts URL `https://console.nace.ai/v1/systemone`, header `X-Nace-Client == 'library-claim/1'`, body `images[0].startswith('data:image/jpeg;base64,')`, `model == 'drex-max-v1.1'`; returns the smoke-test response shape (`answers.book_visible.noul = 0.002`, `usage.input_tokens = 803`, `request_id = 'req_x'`). Assert `result.answers['book_visible'].noul == 0.002`, `result.input_tokens == 803`.
  - `test_rate_limit_retries_once_after_retry_after`: first response 429 with `retry-after-ms: 1`, second 200 → succeeds, 2 requests seen.
  - `test_payment_required_raises_safe_error`: 402 with body containing `"secret"` → `DrexError`; `'secret' not in str(error)`; one request seen.
  - `test_missing_key_raises`: `DREX_API_KEY` unset (patch `os.environ`) → `DrexError`.
  - In `tests/test_config.py`: `test_drex_thresholds_loaded`: `CONFIG.drex.same_book_min == 0.8` and `CONFIG.drex.different_book_max == 0.2`.
- [ ] **Step 2: Run** `.venv/bin/python -m unittest tests.test_drex tests.test_config -v` → FAIL (module/attribute missing).
- [ ] **Step 3: Implement** config classes and `drex.py`. Load `.env` the way `provider.py` does. Log `request_id`, latency, `input_tokens` at INFO.
- [ ] **Step 4: Implement** `scripts/drex_smoke.py`: sends one JPEG path from argv with the D1 questions (import them from Task 2 once it exists; until then inline) and prints the answers. Not run by tests.
- [ ] **Step 5: Run** the two test modules → PASS; run full suite → 156+ pass.
- [ ] **Step 6: Commit** `git commit -m "Add Drex decision client and configuration"`.

---

### Task 2: D1 frame check decision and its record

**Files:**
- Create: `library_claim/decisions/frame_check.py`
- Modify: `library_claim/domain/models.py` (`ProcessingJob.task` adds `'frame_check'`)
- Test: `tests/test_frame_check.py`

**Interfaces:**
- Consumes: `decide`, `noul`, `choice`, `DrexError` (Task 1).
- Produces:
  - `ISSUES = ('none','blur','glare','dark','far','occluded','no_books')`
  - `class FrameCheck(BaseModel)`: `book_visible: float`, `spine_legible: float`, `issue: str`, `issue_probabilities: dict[str, float]`, `model: str`, `request_id: str | None`, `latency_ms: float`, `input_tokens: int`.
  - `def check_frame(jpeg: bytes, *, client: httpx.Client | None = None) -> FrameCheck` — questions and wording exactly as in the spec D1 section.
  - `def is_book_candidate(check: FrameCheck | None) -> bool` — `check is not None and check.book_visible >= CONFIG.drex.book_visible_min`.
  - `def save_frame_check(database: Path, sweep_id: str, frame_id: str, check: FrameCheck) -> None` — writes `ProcessingJob(id=f'check_{frame_id}', task='frame_check', status='succeeded', model=check.model, result=check.model_dump())`; idempotent (existing ID → no-op).

- [ ] **Step 1: Write failing tests**: `test_check_frame_maps_answers` (fake transport → `issue == 'no_books'`, `book_visible == 0.002`); `test_candidate_threshold` (`0.5` → True, `0.49` → False, `None` → False); `test_save_frame_check_is_idempotent` (temp DB with a saved `Frame` via `LiveCaptureStore`; save twice → one `frame_check` job).
- [ ] **Step 2: Run** `.venv/bin/python -m unittest tests.test_frame_check -v` → FAIL.
- [ ] **Step 3: Implement** `frame_check.py` and the `task` literal.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `"Add Drex frame check decision and persisted result"`.

---

### Task 3: Wire D1 into the live loop (replaces CLIP admission)

**Files:**
- Create: `live_demo/frame_checker.py`
- Rename/modify: `live_demo/camera_relevance.py` → `live_demo/camera_gate.py`
- Modify: `live_demo/camera_tracking.py`, `live_demo/camera_filter.py`, `live_demo/camera_logging.py`, `live_demo/server.py` (`track_camera`, `live_voice` setup)
- Test: `tests/test_frame_checker.py`; update `tests/test_camera_filter.py`, `tests/test_tracking_transport.py`, `tests/test_live_persistence.py`, `tests/test_camera_logging.py` to the new types.

**Interfaces:**
- Consumes: `check_frame`, `is_book_candidate`, `save_frame_check`, `FrameCheck` (Task 2).
- Produces:
  - `def local_gate(quality: QualityAssessment) -> str | None` in `camera_gate.py` — returns `'extreme_exposure'` or `'insufficient_texture_or_sharpness'` using the existing thresholds; BRISQUE checks removed.
  - `@dataclass(frozen=True) class PendingCheck` in `camera_tracking.py`: `jpeg, frame_id, timestamp, captured_at, result: TrackingResult, quality: QualityAssessment, width, height, saved: bool`.
  - `CameraTracking.__init__(store=None)` (no `relevance`); after `assess()`, `self.pending_check: PendingCheck | None` and `self.gate_failure: str | None` describe the frame. `pending_check` is set for frames that today would reach CLIP (selection reason or uncertain tracking) and pass `local_gate`. `saved` is True when a tracking reason saved it.
  - `CameraTracking.save_book_candidate(pending: PendingCheck) -> None` — saves with reason `'book_candidate'`.
  - `CameraFilter.offer(candidate: CameraCandidate, check: FrameCheck | None)` — same semantics with `is_book_candidate(check)` replacing the CLIP label.
  - `class FrameChecker` in `frame_checker.py`: `__init__(check=check_frame, *, min_interval_s=CONFIG.capture.quality_interval_s)`; `async def submit(pending: PendingCheck, on_result: Callable[[PendingCheck, FrameCheck | None], Awaitable[None]]) -> None`. One request in flight; one waiting slot where a newer `submit` replaces the waiting one; requests start at most once per `min_interval_s`; `check` runs via `asyncio.to_thread`; `DrexError` → `on_result(pending, None)`.
- Server `on_result`: if candidate and not `pending.saved` → `save_book_candidate`; if saved → `save_frame_check`; then `gate.offer(...)`/`gate.take(...)`/`capture(...)` exactly as `track_camera` does today; pass `check` (or the gate failure) to D6 (Task 5 hook; until then no-op). On `None` with a `DrexError` of 402/401, send one UI error `"Drex decisions unavailable; check credit."` per camera activation.

- [ ] **Step 1: Write failing tests** in `tests/test_frame_checker.py`:
  - `test_newer_submission_replaces_waiting_one`: fake `check` blocks on an event; submit A, B, C; release → only A and C checked.
  - `test_interval_spacing`: two sequential submissions with `min_interval_s=0.2` start ≥ 0.2 s apart.
  - `test_drex_failure_reports_none`: fake `check` raises `DrexError('x')` → `on_result` called with `None`, no exception escapes. *(Review Focus 1)*
  - In `tests/test_tracking_transport.py`: `test_uncertain_frame_not_saved_until_checked`: uncertain-tracking frame → `pending_check.saved is False` and no frame file written; `save_book_candidate` then writes it.
- [ ] **Step 2: Run** these tests → FAIL.
- [ ] **Step 3: Implement** `camera_gate.py`, `PendingCheck`, tracking changes, `FrameChecker`, filter change, logging change (summary shows last D1 `issue`/`book_visible` instead of CLIP label).
- [ ] **Step 4: Wire** `server.py`: create one `FrameChecker` per camera activation next to `CameraTracking`; in `track_camera`, after `stream.assess`, submit `stream.pending_check` if present.
- [ ] **Step 5: Run** full suite → PASS (update the listed existing tests to construct `CameraTracking(store)` and `FrameCheck` instead of `RelevanceAssessment`).
- [ ] **Step 6: Manual check**: start server, point laptop camera at books for 30 s; log shows `frame_check` entries ~1/s and Qwen captures firing.
- [ ] **Step 7: Commit** `"Admit camera frames with Drex frame checks instead of CLIP"`.

---

### Task 4: Remove the CLIP relevance stack

**Files:**
- Delete: `library_claim/capture/relevance/` (package + README), `tests/test_relevance.py`, `tests/test_relevance_audit.py`, `tests/test_relevance_edge_audit.py`, `docs/evaluations/relevance-*.md`
- Modify: `library_claim/config.py` and `config.yml` (remove `relevance` and `experiments` sections), `requirements.txt` (remove `onnxruntime`, `tokenizers` if no other import remains — verify with `grep -rn "onnxruntime\|tokenizers" --include='*.py' .`), `library_claim/capture/README.md` (drop the CLIP paragraph), `docs/configuration.md`, `library_claim/domain/models.py`/`live_capture.py` only if they import relevance types.

- [ ] **Step 1:** `grep -rn "relevance" --include='*.py' library_claim live_demo tests` and remove every remaining import/use.
- [ ] **Step 2: Run** full suite → PASS; `.venv/bin/python -c "import live_demo.server"` succeeds.
- [ ] **Step 3:** Delete `.cache/book-relevance` (untracked; tell the user, do not commit).
- [ ] **Step 4: Commit** `"Remove CLIP relevance classifier and its evaluations"`.

---

### Task 5: D6 guidance decision

**Files:**
- Create: `live_demo/guidance_decision.py`
- Modify: `live_demo/server.py` (D6 hook from Task 3, local-gate streak, missing-title path in `capture()`)
- Test: `tests/test_guidance_decision.py`

**Interfaces:**
- Consumes: `decide`, `choice` (Task 1); `FrameCheck` (Task 2); existing `GuidancePolicy`, `capture_instruction`, `request_capture`, `guide`.
- Produces:
  - `QUALITY_PROMPTS: dict[str, str]` — fixed copy:
    - `slow_down`: "Slow down a little so I can read the spines."
    - `move_closer`: "Move a little closer to this shelf so I can read the titles."
    - `reduce_glare`: "Shift the camera slightly to the side to avoid the glare."
    - `more_light`: "It's too dark here to read the spines."
    - `reposition`: "Something is blocking the books; shift the camera so I can see them."
    - `pan_back`: "Pan back a little; I only saw part of that shelf."
  - `LABELS = ('stay_quiet', *QUALITY_PROMPTS, 'read_title')`
  - `@dataclass class GuidanceSignals`: `recent_issues: list[tuple[str, float]]`, `gate_failure_streak: int`, `missing_title_books: int`, `edge_cut_books: int`, `last_prompt: str | None`, `seconds_since_prompt: float | None`, `phase: str`.
  - `def choose_prompt(signals: GuidanceSignals, *, client: httpx.Client | None = None) -> str` — one Drex `choice` question `prompt` over `LABELS` with state `asdict(signals)`; `DrexError` → `'stay_quiet'`.
  - `class QualityPromptLimiter`: `allow(label: str, now: float) -> bool` and `record(label: str, now: float) -> None` — kind cooldown `quality_prompt_cooldown_s`, same-label cooldown `same_prompt_cooldown_s`.
  - `def edge_cut_count(inventory: list[dict]) -> int` — books whose every observation has `'cut_off'` in `issues`.
- Server rules:
  - Triggers: D1 result with `issue not in ('none','no_books')`; `gate_failure_streak >= CONFIG.drex.gate_failure_streak`; after `save_analysis`, any missing-title or edge-cut book.
  - Never call `guide` while the user is speaking (voice state active or `text_task` running) or `spoken_turn_id` is set.
  - `read_title` → `book = session.guidance_policy.choose(...)`; if a book → `request_capture(book)` (existing limits apply). Quality labels → `limiter.allow` then `guide(QUALITY_PROMPTS[label])`. `stay_quiet` → nothing.
  - A local gate streak at the threshold speaks `slow_down` directly through the limiter (no Drex call).

- [ ] **Step 1: Write failing tests**: `test_choose_prompt_maps_choice` (fake transport returns `choice: 'reduce_glare'`); `test_drex_failure_is_quiet`; `test_limiter_kind_and_label_cooldowns` (record `slow_down` at 0 → `move_closer` at 7 False, at 9 True; `slow_down` at 19 False, at 21 True); `test_edge_cut_count`; `test_no_prompt_while_user_speaking` — a helper `should_speak(user_speaking: bool, agent_speaking: bool) -> bool` in `guidance_decision.py` returns False for either. *(Review Focus 2)*
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Wire server.** **Step 5: Run** full suite → PASS.
- [ ] **Step 6: Commit** `"Let Drex choose live capture guidance"`.

---

### Task 6: Spine measurement from a depth map

**Files:**
- Create: `library_claim/measurement/__init__.py`, `library_claim/measurement/spine.py`
- Modify: `library_claim/domain/models.py` (`BookObservation` adds `spine_height_cm: PositiveFloat | None = None`, `spine_thickness_cm: PositiveFloat | None = None`, `scale_method: Literal['webxr_depth','depth_pro'] | None = None`)
- Test: `tests/test_spine_measurement.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) class DepthMap`: `depth_m: np.ndarray` (H×W float32, 0 = unknown), `fx, fy, cx, cy: float` in **source-image pixels**, `image_width, image_height: int`, `method: Literal['webxr_depth','depth_pro']`.
  - `@dataclass(frozen=True) class SpineMeasurement`: `height_cm: float`, `thickness_cm: float`, `method: str`.
  - `def measure_spine(bbox: Box, view: str | None, issues: list[str], depth: DepthMap) -> SpineMeasurement | None`
  - `def book_dimensions(observations: list[BookObservation]) -> tuple[float, float, str] | None` — medians of measured observations; method of the majority.
  - `def shelf_run_m(thicknesses_cm: list[float]) -> float`.
- Algorithm for `measure_spine` (the tests do not fully determine it):
  ```
  return None unless view == 'spine' and not {'cut_off','occlusion'} & set(issues)
  long axis = vertical if box height_px >= width_px else horizontal
  for each axis (long → height, short → thickness):
      endpoints = centre ± (half_length * (1 - depth_inset))   # inset avoids background
      depth at each endpoint = median of non-zero depth in a depth_window_px window (depth grid coords scaled from image coords)
      return None if either depth is missing
      back-project p = ((u-cx)*Z/fx, (v-cy)*Z/fy, Z); length = |p1-p2| / (1 - depth_inset)
  return SpineMeasurement(height_cm=long*100, thickness_cm=short*100, method=depth.method)
  ```

- [ ] **Step 1: Write failing tests** (synthetic depth, 1000×1000 image, `fx=fy=1000`, `cx=cy=500`):
  - `test_fronto_parallel_spine`: constant depth 1.0 m; box `x 0.48–0.52, y 0.40–0.60` → height ≈ 20.0 cm, thickness ≈ 4.0 cm (±1%).
  - `test_background_at_box_edges_does_not_inflate`: same box, depth 1.0 inside the inner 80 %, 3.0 m in the outer 10 % rim → height within 2 % of 20.0. *(Review Focus 3)*
  - `test_tilted_spine_uses_depth_gradient`: depth varies linearly 0.9→1.1 m top→bottom → height > fronto-parallel result.
  - `test_non_spine_or_cut_off_not_measured`: `view='cover'` → None; `issues=['cut_off']` → None.
  - `test_missing_depth_returns_none`: all zeros → None.
  - `test_book_dimensions_median_and_shelf_run`: observations 20/21/22 cm → 21; `shelf_run_m([3.0, 2.5]) == 0.055`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `"Measure spine height and thickness from metric depth"`.

---

### Task 7: Depth sources, storage, and measurement in the pipeline

**Files:**
- Create: `library_claim/storage/depth.py`, `library_claim/measurement/depth_pro.py`
- Modify: `library_claim/identification/storage.py` (`save_analysis` gains `depth: DepthMap | None = None`; `inventory` adds dimensions), `live_demo/server.py` (`_pin_evidence_photo`: load sidecar, else Depth Pro), `live_demo/app.js` (show `height × thickness cm` on book cards and total shelf run), `requirements.txt` (`torch`, `transformers` pinned to current stable versions that ship `DepthProForDepthEstimation`)
- Test: `tests/test_depth_storage.py`, extend `tests/test_book_observations.py`

**Interfaces:**
- Consumes: `DepthMap`, `measure_spine`, `book_dimensions`, `shelf_run_m` (Task 6).
- Produces:
  - `def save_depth(root: Path, frame_id: str, depth: DepthMap) -> None` / `def load_depth(root: Path, frame_id: str) -> DepthMap | None` — `.npz` at `<root>/depth/<frame_id>.npz`.
  - `def estimate_depth(jpeg: bytes) -> DepthMap` — lazily loads `CONFIG.measurement.depth_pro_model` once (fp16 on CUDA if available, else fp32 CPU); uses the model's predicted focal length for `fx = fy`, principal point at image centre; raises `DepthUnavailable(RuntimeError)` on load/inference failure.
  - `save_analysis(...)` sets the three measurement fields on each observation when `depth` is given and `measure_spine` returns a value; when `depth is None`, adds `ReviewIssue(reason='No metric depth for this frame')` per book.
  - `inventory()` entries gain `spine_height_cm`, `spine_thickness_cm`, `scale_method` (from `book_dimensions`), and `excluded`/`needs_appraisal` passthrough from `Book`. A new `def inventory_totals(books: list[dict]) -> dict` returns `{'book_count', 'books_identified', 'books_unidentified', 'shelf_run_m'}` over non-excluded books.
- Server: `depth = load_depth(...)`; if None → `await asyncio.to_thread(estimate_depth, frame)`; `DepthUnavailable` → `None`.

- [ ] **Step 1: Write failing tests**: `test_depth_round_trip`; `test_save_analysis_stores_measurements` (constant-depth `DepthMap`, one spine detection → observation has `spine_height_cm` ≈ expected); `test_missing_depth_adds_review_reason` *(Review Focus 4)*; `test_inventory_totals_exclude_excluded_books`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** storage, measurement in `save_analysis`, inventory fields, totals.
- [ ] **Step 4: Implement** `depth_pro.py`; install deps; run once on a saved frame: `.venv/bin/python -c "from library_claim.measurement.depth_pro import estimate_depth; ..."` and record GPU memory/latency in the commit message. If CUDA OOM, confirm the CPU fallback works.
- [ ] **Step 5: Wire** server and UI. **Step 6: Run** full suite → PASS.
- [ ] **Step 7: Commit** `"Store depth, measure books in the pipeline, and show dimensions"`.

---

### Task 8: WebXR depth capture on Android (only if the 15R depth sample works)

**Files:**
- Modify: `live_demo/app.js` (AR capture path), `live_demo/index.html` (AR overlay root), `live_demo/server.py` (parse optional `depth` on `video` messages; pass to tracking), `live_demo/camera_tracking.py` (`assess(..., depth: DepthMap | None = None)`; saving writes the sidecar via `save_depth`)
- Test: `tests/test_depth_payload.py`; manual phone test

**Interfaces:**
- Consumes: `DepthMap`, `save_depth` (Tasks 6–7).
- Produces: `def parse_depth_payload(payload: dict, image_width: int, image_height: int) -> DepthMap` in `library_claim/storage/depth.py`. Payload: `{"width": 96, "height": 54, "mm": "<base64 uint16 little-endian>", "fx", "fy", "cx", "cy"}`, intrinsics already in JPEG pixels. Rejects wrong sizes, non-finite intrinsics, or grids larger than 160×120 with `ValueError`.
- Browser:
  - If `navigator.xr?.isSessionSupported('immersive-ar')` → camera button starts `immersive-ar` with `requiredFeatures: ['depth-sensing','camera-access','dom-overlay']`, `depthSensing: {usagePreference: ['cpu-optimized'], dataFormatPreference: ['luminance-alpha','float32']}`, `domOverlay: {root: <app root>}`. Otherwise the existing `getUserMedia` path.
  - Every `frame_interval_ms`: camera texture via `XRWebGLBinding.getCameraImage(view.camera)` → draw to canvas at `frame_width` → JPEG as today. Depth grid: for each of 96×54 normalized view points call `depthInfo.getDepthInMeters(u, v)` → uint16 mm.
  - Intrinsics from `view.projectionMatrix` P (column-major) for a JPEG of W×H: `fx = P[0]*W/2`, `fy = P[5]*H/2`, `cx = (1 - P[8])*W/2`, `cy = (1 + P[9])*H/2`.
  - Phone access: `adb reverse tcp:<port> tcp:<port>`, open `http://localhost:<port>` (secure context; server's localhost check still passes).

- [ ] **Step 1: Write failing tests**: `test_parse_depth_payload_scales_mm`; `test_parse_rejects_oversized_grid`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** parser and server/tracking plumbing. **Step 4: Run** → PASS.
- [ ] **Step 5: Implement** the browser AR path.
- [ ] **Step 6: Manual phone test**: measure the 5 books with a tape; sweep; record measured vs. tape in the commit message.
- [ ] **Step 7: Commit** `"Capture WebXR depth on Android for metric spine scale"`.

---

### Task 9: D2 same-book decisions

**Files:**
- Create: `library_claim/decisions/same_book.py`
- Modify: `library_claim/identification/storage.py` (`association_choices` takes `decide_same`), `library_claim/identification/review.py` (`reconcile` passes it through)
- Test: `tests/test_same_book.py`; update `tests/test_book_observations.py` cases that relied on the 0.15 margin to inject a fake `decide_same`.

**Interfaces:**
- Consumes: `decide`, `noul`, `DrexError` (Task 1).
- Produces:
  - `@dataclass(frozen=True) class SameBookCandidate`: `book_id: str`, `crop_jpeg: bytes`, `title: str | None`, `author: str | None`, `view: str | None`, `score: float`, `method: str`.
  - `def same_book_probabilities(detection_crop: bytes, detection: BookDetection, candidates: list[SameBookCandidate], *, client=None) -> list[float]` — one request, images = detection crop + up to 3 candidate crops, one `noul` per candidate (`same_k`), state = readings/views/scores; raises `DrexError`.
  - `def choose_same_book(probabilities: list[float]) -> int | None | Literal['unresolved']` — index if max ≥ `same_book_min`; `None` (new book) if all ≤ `different_book_max`; else `'unresolved'`.
  - `association_choices(..., decide_same: Callable[[bytes, BookDetection, list[SameBookCandidate]], list[float]] = same_book_probabilities)` — existing vetoes unchanged; surviving candidates sorted by today's score, top 3 passed; `association_method = 'drex'` on reuse; `DrexError` → `'unresolved'`. The 0.15 margin rule is deleted. The co-visible exact-title rule stays. Crops come from the existing evidence images (encode the bbox region with `cv2.imencode('.jpg', ...)`).
  - `save_analysis` calls `association_choices` before `repo.add_many` (already the case — keep it that way).

- [ ] **Step 1: Write failing tests**: `test_choose_same_book_thresholds` (`[0.9,0.3]→0`, `[0.1,0.15]→None`, `[0.5]→'unresolved'`); `test_association_uses_drex_decision` (fake `decide_same` returning `[0.95]` → reuse with method `'drex'`); `test_drex_failure_leaves_unresolved`; `test_conflicting_titles_never_sent_to_drex` (fake asserts it is not called).
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Run** full suite → PASS.
- [ ] **Step 5: Commit** `"Decide physical-copy association with Drex"`.

---

### Task 10: D3 catalogue resolution

**Files:**
- Create: `library_claim/identification/catalogue.py`
- Modify: `library_claim/domain/models.py` (`BookObservation` adds `catalogue_url: str | None = None`, `retrieved_at: AwareDatetime | None = None`), `library_claim/identification/storage.py` (`inventory` adds `catalogue: {title, author, edition, isbn, url, retrieved_at} | None` from the latest `provenance='catalogue'` observation; visual readings stay from visual observations only), `live_demo/server.py` (background resolution after `save_analysis`), `live_demo/app.js` (catalogue link on card)
- Test: `tests/test_catalogue.py`

**Interfaces:**
- Consumes: `decide`, `choice` (Task 1); `normalize`, `text_relation` (matching.py).
- Produces:
  - `@dataclass(frozen=True) class WorkCandidate`: `key: str`, `title: str`, `authors: list[str]`, `first_publish_year: int | None`.
  - `def search_works(title: str, author: str | None, *, client=None) -> list[WorkCandidate]` — `GET https://openlibrary.org/search.json?title=&author=&limit=5&fields=key,title,author_name,first_publish_year`, `User-Agent: library-claim/1`.
  - `def matching_edition(work_key: str, publisher: str | None, *, client=None) -> dict | None` — `GET https://openlibrary.org{work_key}/editions.json?limit=50`; returns `{'edition': edition title or key, 'isbn': first isbn_13 else isbn_10}` only when exactly one edition's normalized publisher equals the normalized spine publisher.
  - `def choose_work(readings: dict, candidates: list[WorkCandidate], *, client=None) -> WorkCandidate | None` — Drex `choice` with labels `c0..c4` + `none`; accept if probability ≥ `catalogue_min`.
  - `def resolve_book(database: Path, sweep_id: str, book: dict, *, http=None, drex=None) -> None` — appends a `BookObservation(provenance='catalogue', title, author, edition, isbn, catalogue_url=f'https://openlibrary.org{key}', retrieved_at=now)` using the book's latest observation's `frame_id` and `bbox`; or a `ReviewIssue(field='catalogue', reason='No confident catalogue match')`. Any HTTP/Drex error → returns without writing.
- Server: after each `save_analysis`, for inventory books with a title whose title differs from the last resolved title, `track(asyncio.to_thread(resolve_book, ...))` then refresh inventory.

- [ ] **Step 1: Write failing tests** (MockTransport for Open Library and Drex): `test_search_parses_candidates`; `test_choose_work_rejects_low_probability` (0.6 → None); `test_none_label_returns_none`; `test_edition_only_with_unique_publisher_match` (two editions same publisher → None; one → isbn); `test_resolve_writes_catalogue_observation`; `test_resolve_failure_writes_nothing`.
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Wire server/UI.** **Step 5: Run** full suite → PASS.
- [ ] **Step 6: Commit** `"Resolve legible spines to Open Library works with Drex"`.

---

### Task 11: D5 voice corrections

**Files:**
- Create: `library_claim/decisions/corrections.py`, `library_claim/identification/corrections.py`
- Modify: `live_demo/server.py` (`submit()` during a library sweep), `live_demo/live_tools.py` (system instruction: corrections are applied by the system; acknowledge them briefly), `library_claim/identification/storage.py` (`inventory` hides excluded books from totals, shows them greyed)
- Test: `tests/test_corrections.py`

**Interfaces:**
- Consumes: `decide`, `choice` (Task 1); `provider.structured` (Qwen); `SweepController.invalidate_context`; `inventory`.
- Produces:
  - `INTENTS = ('exclude_books','mark_first_edition','mark_signed','correct_title','question_or_chat')`
  - `def classify_intent(text: str, *, client=None) -> str` — `DrexError` → `'question_or_chat'`.
  - `def resolve_target(text: str, books: list[dict], *, allow_all: bool, client=None) -> list[str] | None` — `books` are the inventory entries seen in the target frame, described by left-to-right order (bbox `x_min`) and readings; one book → `[id]` without a Drex call; labels = book IDs (+ `all_visible` when `allow_all`); below `target_min` or `DrexError` → `None`.
  - `class StatedTitle(BaseModel)`: `title: str | None`; `async def extract_stated_title(text: str) -> str | None` via `provider.structured`.
  - `def apply_correction(database: Path, sweep_id: str, intent: str, book_ids: list[str], frame_id: str, statement: str, stated_title: str | None = None) -> None`:
    - every intent appends a `BookObservation(provenance='user_statement', visible_text=statement[:500], bbox=<book's bbox in frame_id>)`;
    - `exclude_books` → `Book.excluded = True` (`repo.update`);
    - `mark_first_edition` / `mark_signed` → `Book.needs_appraisal = True`;
    - `correct_title` → observation `title=stated_title` plus `ReviewIssue(field='title', reason='User-stated title differs from spine reading')`; the visual reading is untouched.
- Server `submit()` (library sweep only, before `respond`): `intent = await asyncio.to_thread(classify_intent, text)`; if not chat → target frame = most recent saved frame with detections within 30 s; none → `guide("Which books do you mean? Point the camera at them.")` and stop *(Review Focus 5)*; `resolve_target` `None` → `guide("Which book do you mean?")`; else `apply_correction` → `SweepController(repo, session_id).invalidate_context()` → refresh inventory → continue to `respond` so Qwen acknowledges.

- [ ] **Step 1: Write failing tests**: `test_classify_intent_failure_is_chat`; `test_single_visible_book_needs_no_drex`; `test_low_target_confidence_returns_none`; `test_exclude_sets_flag_and_records_statement`; `test_first_edition_sets_needs_appraisal`; `test_correct_title_keeps_visual_reading` (inventory title still the spine reading; review reason present); `test_no_recent_frame_asks_instead_of_excluding` — a pure helper `target_frame(frames: list[tuple[str, float]], now: float) -> str | None` with a 30 s window. *(Review Focus 5)*
- [ ] **Step 2: Run** → FAIL. **Step 3: Implement.** **Step 4: Wire server.** **Step 5: Run** full suite → PASS.
- [ ] **Step 6: Manual check**: sweep the 5 books, say "skip that one, it's not mine" → card greyed, totals drop by one.
- [ ] **Step 7: Commit** `"Apply spoken inventory corrections decided by Drex"`.

---

### Task 12 (only if time remains): Remove insurance-intake leftovers

**Files:**
- Delete/modify: `policies.py`, `policy_directory.py`, `lookup_policy` and policy fields in `live_demo/server.py` and `live_demo/live_tools.py`, related tests.

- [ ] **Step 1:** `grep -rn "lookup_policy\|policies\|policy_directory" --include='*.py' .` and remove the paths only the intake flow uses.
- [ ] **Step 2: Run** full suite → PASS; manual sweep still works.
- [ ] **Step 3: Commit** `"Remove unused insurance intake code"`.
