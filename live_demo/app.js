const transcriptEl = document.querySelector("#transcript");
const notesEl = document.querySelector("#notes");
const pinboardEl = document.querySelector("#pinboard");
const stampEl = document.querySelector("#stamp");
const penEl = document.querySelector("#pen");
const neededListEl = document.querySelector("#neededList");
const teamFeedEl = document.querySelector("#teamFeed");
const readinessEl = document.querySelector("#readiness");
const callStatus = document.querySelector("#callStatus");
const modelLabel = document.querySelector("#modelLabel");
const pageDate = document.querySelector("#pageDate");
const micButton = document.querySelector("#micButton");
const cameraButton = document.querySelector("#cameraButton");
const cameraStage = document.querySelector("#cameraStage");
const cameraPreview = document.querySelector("#cameraPreview");
const frameCanvas = document.querySelector("#frameCanvas");
const newIntakeButton = document.querySelector("#newIntakeButton");
const textForm = document.querySelector("#textForm");
const textInput = document.querySelector("#textInput");

const DEFAULT_API_ORIGIN = "http://127.0.0.1:4177";
const API_ORIGIN = window.location.protocol === "file:" ? DEFAULT_API_ORIGIN : window.location.origin;
const WS_ORIGIN = API_ORIGIN.replace(/^http/, "ws");
// Tracking receives intermediate views; quality and remote inference keep separate cadences.
let cameraConfig = null;

if (window.location.protocol === "file:") {
  window.location.replace(`${API_ORIGIN}/index.html`);
}

let liveSocket = null;
let connectionPromise = null;
let generation = 0;
let processing = false;
let cameraPending = false;
let resetting = false;
let cameraStream = null;
let frameTimer = null;
let arSession = null;
let sessionId = null;
let state = null;
let writing = false;
const seenNotes = new Set();
const voice = window.ClaimVoice ? new window.ClaimVoice({
  send: message => {
    if (liveSocket?.readyState === WebSocket.OPEN) liveSocket.send(JSON.stringify(message));
  },
  status: label => {
    micButton.querySelector('.round-label').textContent = label === 'Voice off' ? 'Talk' : label;
    micButton.classList.toggle('active', label !== 'Voice off');
  },
  error: message => appendSystem(message),
}) : null;

const routeLabels = {
  emergency_escalation: ["Escalate to human", "danger"],
  policy_review: ["Policy review", "warning"],
  needs_docs: ["Needs docs", "warning"],
  special_investigation: ["SIU review", "info"],
  ready_for_adjuster: ["Ready for adjuster", "success"],
};

const blockerQuestions = {
  policyholder_name: "Name?",
  policy_number: "Policy number?",
  contact_method: "Best contact?",
  date_of_loss: "When did it happen?",
  loss_location: "Where?",
  loss_description: "What happened?",
};

const teamLabels = {
  lookup_policy: "Policy desk",
  sync_claim_packet: "Claim writer",
  pin_evidence_photo: "Evidence",
};

const emptyState = {
  route: "needs_docs",
  progress: 0,
  fields: {},
  transcript: [],
  events: [],
  tool_activity: [],
  missing_blockers: [],
  documents: [],
  evidence_photos: [],
  camera_notes: [],
  policy: null,
  handoff: {},
  packet_markdown: "# Library sweep evidence\n\nNo evidence yet.",
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function now() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function isFilled(field) {
  if (!field || field.status === "missing") return false;
  return !/^(missing:|not captured|not specified|unknown|none)/i.test(String(field.value || "").trim());
}

function shortBlocker(blocker) {
  if (blockerQuestions[blocker]) return blockerQuestions[blocker];
  let text = String(blocker).replace(/\s*\([^)]*\)/g, "").trim();
  if (text.length > 42) text = `${text.slice(0, 40).trim()}...`;
  return text.endsWith("?") ? text : `${text}?`;
}

function setStatus(text, tone = "") {
  callStatus.textContent = text;
  callStatus.className = `pill ${tone}`.trim();
}

function setWriting(next) {
  writing = next;
  penEl.hidden = !next;
}

function setState(nextState) {
  const previous = state || emptyState;
  state = {
    ...emptyState,
    ...nextState,
    tool_activity: nextState.tool_activity ?? previous.tool_activity,
  };
  render();
}

function render() {
  renderTranscript();
  renderNotes();
  renderPinboard();
  renderNeeded();
  renderTeam();
  renderStamp();
  document.querySelector("#downloadPacket").href = sessionId ? `${API_ORIGIN}/api/sessions/${sessionId}/packet` : "#";
}

function renderTranscript() {
  transcriptEl.innerHTML = (state.transcript || [])
    .map((turn) => {
      const cls = turn.speaker === "Agent" ? "agent" : turn.speaker === "System" ? "system" : "claimant";
      const who = turn.speaker === "Agent" ? "AI" : turn.speaker === "System" ? "!" : "You";
      return `<article class="turn ${cls} ${turn.streaming ? "streaming" : ""}"><span class="who">${who}</span><p>${escapeHtml(turn.text)}</p></article>`;
    })
    .join("");
  transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

function buildNotes() {
  if (state.library_sweep) return [{key:"sweep", cls:"aside", text:state.capture_instruction || "Sweep slowly across the shelves. Open Books to review readings."}];
  const f = state.fields || {};
  const notes = [];
  const name = isFilled(f.claimant) ? f.claimant.value : "";
  const policy = isFilled(f.policy) ? f.policy.value : "";
  if (name || policy) {
    notes.push({ key: `title:${name}|${policy}`, cls: "title", text: [name, policy].filter(Boolean).join("  ·  ") });
  }
  const record = state.policy;
  if (record) {
    if (record.found) {
      const active = record.status === "active";
      const extras = (record.coverages || []).slice(0, 1).join("");
      notes.push({
        key: `policy:${record.policy_number}:${record.status}`,
        cls: active ? "check" : "flag urgent",
        text: active
          ? `${record.policy_line}, active. ${extras}`
          : `${record.policy_line}, ${record.status}. Human review before anything else.`,
      });
    } else {
      notes.push({ key: `policy:notfound:${record.policy_number}`, cls: "flag urgent", text: `Policy ${record.policy_number || ""} not found, confirm the number.` });
    }
  }
  const when = isFilled(f.date) ? f.date.value : "";
  const where = isFilled(f.location) ? f.location.value : "";
  if (when || where) {
    notes.push({ key: `whenwhere:${when}|${where}`, cls: "", text: [where, when].filter(Boolean).join(", ") });
  }
  if (isFilled(f.description)) {
    notes.push({ key: `desc:${f.description.value}`, cls: "", text: f.description.value });
  }
  if (isFilled(f.injuries)) {
    const urgent = f.injuries.status === "urgent";
    notes.push({ key: `inj:${f.injuries.value}`, cls: urgent ? "flag urgent" : "", text: urgent ? `Injury: ${f.injuries.value}` : f.injuries.value });
  }
  if (isFilled(f.contact)) {
    notes.push({ key: `contact:${f.contact.value}`, cls: "aside", text: `Reach at ${f.contact.value}` });
  }
  if (isFilled(f.photos)) {
    notes.push({ key: `ev:${f.photos.value}`, cls: "aside", text: `Has: ${f.photos.value}` });
  }
  if (isFilled(f.police)) {
    notes.push({ key: `rep:${f.police.value}`, cls: "aside", text: `Report: ${f.police.value}` });
  }
  if (claimantHasSpoken()) {
    const seenQuestions = new Set();
    for (const blocker of state.missing_blockers || []) {
      const question = shortBlocker(blocker);
      const dedupe = question.toLowerCase().replace(/[^a-z]/g, "");
      if (seenQuestions.has(dedupe) || seenQuestions.size >= 3) continue;
      seenQuestions.add(dedupe);
      notes.push({ key: `blank:${blocker}`, cls: "blank", text: question, blank: true });
    }
  }
  if (!notes.length) {
    notes.push({ key: "empty", cls: "aside", text: "Sweep slowly across the shelves. Open Books to review readings." });
  }
  return notes;
}

function claimantHasSpoken() {
  return (state.transcript || []).some((turn) => turn.speaker === "Claimant" && String(turn.text || "").trim());
}

function renderNotes() {
  const notes = buildNotes();
  notesEl.innerHTML = notes
    .map((note) => {
      const fresh = !seenNotes.has(note.key);
      seenNotes.add(note.key);
      return `<div class="note ${note.cls} ${fresh ? "ink-in" : ""}">${escapeHtml(note.text)}${note.blank ? '<span class="blank-line"></span>' : ""}</div>`;
    })
    .join("");
}

let bookTab = "review";
let selectedBookId = null;
let compareBookId = null;
let reviewBusy = false;

function bookNeedsReview(book) {
  return book.needs_review ?? (book.status === "unresolved" || !book.title || Boolean(book.review_reasons?.length));
}

function observationUrl(observation) {
  return `${API_ORIGIN}/api/sessions/${encodeURIComponent(sessionId)}/books/evidence/${encodeURIComponent(observation.id)}`;
}

function bestObservation(book) {
  return [...(book.observations || [])].sort((a,b) => (b.sharpness || 0) - (a.sharpness || 0))[0];
}

function bookGroups(books) {
  const groups = [];
  for (const book of books) {
    // Suggestions group cards for inspection; they never merge identity records.
    const group = groups.find(items => items.every(other =>
      ((book.candidates || []).includes(other.id) || (other.candidates || []).includes(book.id)) &&
      !(book.frame_ids || []).some(id => (other.frame_ids || []).includes(id))));
    if (group) group.push(book); else groups.push([book]);
  }
  return groups;
}

function cropImage(book, className = "") {
  const image = bestObservation(book);
  return image ? `<img class="${className}" src="${observationUrl(image)}" alt="Observed crop: ${escapeHtml(book.title || "unreadable book")}" loading="lazy">` : '<span class="missing-crop">No clear image yet</span>';
}

function bookSize(book) {
  // Sizes come only from measured depth; say so when a book has none yet.
  if (!book.spine_height_cm) return "Not measured yet";
  const source = book.scale_method === "webxr_depth" ? "AR depth" : "Depth Pro";
  return `${Number(book.spine_height_cm).toFixed(1)} × ${Number(book.spine_thickness_cm).toFixed(1)} cm · ${source}`;
}

function renderBookInventory() {
  const books = state.book_inventory || [];
  const groups = bookGroups(books);
  const requiresReview = group => group.length > 1 || group.some(bookNeedsReview);
  const reviewCount = groups.filter(requiresReview).length;
  const totals = state.inventory_totals || {};
  const run = totals.shelf_run_m ? ` · shelf run ${Number(totals.shelf_run_m).toFixed(2)} m` : "";
  document.querySelector("#openBooks").textContent = `Books · ${groups.length - reviewCount} identified · ${reviewCount} need review${run}`;
  document.querySelector("#reviewTab").setAttribute?.("aria-pressed", String(bookTab === "review"));
  document.querySelector("#identifiedTab").setAttribute?.("aria-pressed", String(bookTab === "identified"));
  const visible = groups.filter(group => requiresReview(group) === (bookTab === "review"));
  const inventoryEl = document.querySelector("#bookInventory");
  inventoryEl.hidden = false;
  inventoryEl.innerHTML = visible.length ? visible.map(group => {
    const book = group[0];
    return `<button class="book-card${book.excluded ? " excluded" : ""}" data-book-id="${escapeHtml(book.id)}" type="button">
      <span class="group-crops">${group.map(item => cropImage(item)).join("")}</span>
      <span class="book-card-copy"><small class="book-badge">${group.length > 1 ? "Possible same book" : bookNeedsReview(book) ? "Needs review" : "Title read"}</small>
      <strong>${escapeHtml(book.title || "Unidentified book")}</strong>
      <span>${escapeHtml(book.author || "Author unreadable")}</span>
      <small>${escapeHtml(bookSize(book))}</small>
      ${book.excluded ? "<small>Excluded by you · not counted</small>" : ""}
      ${book.needs_appraisal ? "<small>Needs human appraisal</small>" : ""}
      ${book.catalogue ? `<small>Open Library: ${escapeHtml(book.catalogue.title)}${book.catalogue.edition ? ` · ${escapeHtml(book.catalogue.edition)}` : ""}</small>` : ""}
      <small>${group.length > 1 ? `${group.length} sightings to compare · not a copy count` : `${Number(book.observation_count)} observation(s)`}</small>
      ${book.status === "unresolved" ? "<small>Association unresolved</small>" : ""}</span></button>`;
  }).join("") : `<p class="empty">${bookTab === "review" ? "No books need review yet." : "No identified books yet."}</p>`;
  renderBookDetail();
  document.querySelector("#captureInstruction").textContent = state.capture_instruction || "Sweep slowly across the shelves.";
}

function renderBookDetail() {
  const books = state.book_inventory || [];
  const book = books.find(item => item.id === selectedBookId);
  const detail = document.querySelector("#bookDetail");
  if (!book) { selectedBookId = null; detail.innerHTML = '<p class="detail-placeholder">Select a book or a possible-match group to inspect the images.</p>'; return; }
  const others = books.filter(item => item.id !== book.id && !(item.frame_ids || []).some(id => (book.frame_ids || []).includes(id)));
  const other = others.find(item => item.id === compareBookId) || others.find(item => (book.candidates || []).includes(item.id));
  compareBookId = other?.id || null;
  const panel = item => `<article class="comparison-card">${cropImage(item, "book-best")}<h3>${escapeHtml(item.title || "Title unreadable")}</h3><p>${escapeHtml(item.author || "Author unreadable")}</p><p>${escapeHtml(bookSize(item))}</p>${item.catalogue?.url ? `<p><a href="${escapeHtml(item.catalogue.url)}" target="_blank" rel="noopener">Open Library: ${escapeHtml(item.catalogue.title)}</a>${item.catalogue.isbn ? ` · ISBN ${escapeHtml(item.catalogue.isbn)}` : ""}</p>` : ""}<small>${Number(item.observation_count)} observation(s)</small></article>`;
  detail.innerHTML = `<h3>${other ? "Compare these sightings" : "Book details"}</h3>
    <p>${other ? "These may show the same book. Compare the visible design and text; similar titles alone do not establish identity." : "Visual reading only; not catalogue verification."}</p>
    <div class="book-comparison">${panel(book)}${other ? panel(other) : ""}</div>
    ${others.length ? `<label>Compare another sighting <select id="compareBook"><option value="">Select a sighting</option>${others.map(item => `<option value="${escapeHtml(item.id)}" ${item.id === compareBookId ? "selected" : ""}>${escapeHtml(item.title || "Unidentified book")} · ${escapeHtml(item.id.slice(0,6))}</option>`).join("")}</select></label>` : ""}
    ${other ? `<div class="review-actions"><button type="button" data-book-action="same" ${reviewBusy ? "disabled" : ""}>Same book</button><button type="button" data-book-action="separate" ${reviewBusy ? "disabled" : ""}>Separate books</button></div>` : ""}
    <button type="button" data-book-action="capture" ${reviewBusy ? "disabled" : ""}>Request a clearer view</button>
    <details><summary>Capture history (${Number(book.observation_count)})</summary><div class="book-history">${(book.observations || []).map(row => `<figure><img src="${observationUrl(row)}" alt="Previous book observation" loading="lazy"><figcaption>${escapeHtml(row.title || "Title unreadable")}</figcaption></figure>`).join("")}</div></details>`;
}

async function reviewBook(action) {
  if (reviewBusy || !selectedBookId) return;
  const status = document.querySelector("#bookActionStatus");
  reviewBusy = true;
  try {
    if (action === "capture") {
      if (!cameraStream) await startCamera();
      if (!cameraStream) throw new Error("Start the camera to request a clearer view.");
      await connectLive();
      liveSocket.send(JSON.stringify({type:"book_capture", book_id:selectedBookId}));
      document.querySelector("#booksDialog").close();
      document.querySelector("#openBooks").focus();
    } else {
      const target = document.querySelector("#compareBook").value;
      const result = await api(`/api/sessions/${sessionId}/books/review`, {method:"POST", body:JSON.stringify({source_id:selectedBookId,target_id:target,action})});
      selectedBookId = action === "same" ? target : selectedBookId;
      applyServerState(result.state);
      status.textContent = action === "same" ? "Observations combined. Original evidence retained." : "Books marked as separate.";
    }
  } catch (error) { status.textContent = error.message; }
  finally { reviewBusy = false; renderBookDetail(); }
}

function renderPinboard() {
  renderBookInventory();
  const photos = state.evidence_photos || [];
  const cards = photos.map(
    (photo, index) => `
      <figure class="polaroid" style="--tilt: ${index % 2 ? 2 : -2.5}deg" data-key="${escapeHtml(photo.id)}">
        <img src="${photo.data_url}" alt="Camera frame pinned as evidence" />
        <figcaption>${escapeHtml(photo.caption)}</figcaption>

        <span class="tag ${photo.confirmed ? "" : "unconfirmed"}">${photo.claimant_description ? `Claimant says: ${escapeHtml(photo.claimant_description)} · ` : ""}${photo.confirmed ? "Supported by this image" : "Visual evidence; details in Books"}</span>
        <span class="tag">Seen on camera ${escapeHtml(photo.captured_at || "")} · ${escapeHtml(photo.evidence_type || "evidence")}</span>
      </figure>`
  );

  const currentKeys = [...pinboardEl.querySelectorAll("[data-key]")].map((el) => el.dataset.key).join("|");
  const nextKeys = photos.map((p) => p.id).join("|");
  if (currentKeys !== nextKeys) pinboardEl.innerHTML = cards.join("");
}

function renderStamp() {
  if (state.library_sweep) { stampEl.hidden = true; return; }
  const route = state.route;
  const [label, tone] = routeLabels[route] || [route, "warning"];
  stampEl.hidden = !claimantHasSpoken() || writing;
  stampEl.textContent = label;
  if (stampEl.dataset.route !== route) {
    stampEl.dataset.route = route;
    stampEl.className = `stamp ${tone}`;
    stampEl.style.animation = "none";
    void stampEl.offsetWidth;
    stampEl.style.animation = "";
  }
}

function renderNeeded() {
  if (state.library_sweep) {
    const count = (state.book_inventory || []).filter(bookNeedsReview).length;
    neededListEl.innerHTML = `<li>${count ? `${count} book record(s) need review. Open Books.` : "Continue the sweep; review books when needed."}</li>`;
    readinessEl.textContent = "Sweep in progress";
    return;
  }
  const items = [];
  for (const blocker of state.missing_blockers || []) {
    items.push({ text: shortBlocker(blocker).replace(/\?$/, ""), cls: "blocker" });
  }
  for (const doc of state.documents || []) {
    items.push({ text: `${doc.item} — ${doc.status || "unknown"}`, cls: doc.already_provided ? "done" : "" });
  }
  neededListEl.innerHTML = items.length
    ? items.map((item) => `<li class="${item.cls}"><span class="tick-box"></span><span>${escapeHtml(item.text)}</span></li>`).join("")
    : `<li class="empty">Nothing yet. The list fills in as the claim team reads the conversation.</li>`;
  const progress = Number(state.progress || 0);
  readinessEl.textContent = `${progress}% collected`;
  readinessEl.className = `pill ${progress >= 80 ? "" : progress >= 40 ? "warning" : "neutral"}`;
}

function renderTeam() {
  const activity = [...(state.tool_activity || [])].slice(-6).reverse();
  teamFeedEl.innerHTML = activity.length
    ? activity
        .map((item) => {
          const phase = item.phase || "running";
          let headline = item.headline || "";
          if (item.name === "sync_claim_packet" && phase === "done" && item.result) {
            const facts = (item.result.open_items || []).length;
            const docs = (item.result.open_documents || []).length;
            headline = `${String(item.result.routing_decision || "").replaceAll("_", " ")}: ${facts} open fact${facts === 1 ? "" : "s"}${docs ? ", documents still needed" : ""}`;
          }
          const meta = phase === "running" ? "working" : item.scheduling === "INTERRUPT" ? "interrupted the agent" : item.duration_ms != null ? `${item.duration_ms} ms` : "";
          return `<li><span class="team-dot ${phase}"></span><span><span class="team-name">${escapeHtml(teamLabels[item.name] || item.name)}</span> · ${escapeHtml(headline)}</span><span class="team-meta ${item.scheduling === "INTERRUPT" ? "interrupt" : ""}">${escapeHtml(meta)}</span></li>`;
        })
        .join("")
    : `<li class="empty">Camera observations and review activity appear here.</li>`;
  setWriting(processing || (state.tool_activity || []).some((item) => item.phase === "running"));
}

function appendSystem(text) {
  setState({ ...state, transcript: [...(state.transcript || []), { id: crypto.randomUUID(), speaker: "System", text }] });
}

function mergeTranscript(authoritative, local) {
  const byId = new Map((authoritative || []).map(turn => [turn.id, turn]));
  for (const turn of local || []) {
    if (!byId.has(turn.id)) byId.set(turn.id, turn);
  }
  return [...byId.values()];
}

function applyServerState(nextState) {
  if (nextState.session_id !== sessionId) return;
  setState({
    ...nextState,
    transcript: mergeTranscript(nextState.transcript, state?.transcript),
    tool_activity: mergeToolActivity(nextState.tool_activity, state?.tool_activity),
  });
}

function mergeToolActivity(authoritative, local) {
  const finished = new Set(["done", "error", "cancelled"]);
  const byId = new Map();
  for (const item of authoritative || []) byId.set(item.id, item);
  for (const item of local || []) {
    const current = byId.get(item.id);
    if (!current || (finished.has(item.phase) && !finished.has(current.phase))) byId.set(item.id, item);
  }
  return [...byId.values()];
}

function applyToolEvent(message) {
  const { type, ...entry } = message;
  const others = (state.tool_activity || []).filter((item) => item.id !== entry.id);
  setState({ ...state, tool_activity: [...others, entry] });
}

function upsertStreamingTurn(speaker, text, final = false, id = crypto.randomUUID()) {
  if (!String(text || "").trim()) return;
  const transcript = [...(state.transcript || [])];
  const index = transcript.findIndex(turn => turn.id === id);
  const turn = { id, speaker, text, streaming: !final };
  if (index >= 0) transcript[index] = turn;
  else transcript.push(turn);
  setState({ ...state, transcript });
}

async function api(path, options = {}) {
  const response = await fetch(`${API_ORIGIN}${path}`, { headers: { "Content-Type": "application/json" }, ...options });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `Request failed with status ${response.status}`);
  return payload;
}


function clearBusy() {
  processing = false;
  state.tool_activity = (state.tool_activity || []).map(item => item.phase === "running" ? { ...item, phase: "cancelled", headline: "Connection ended" } : item);
  render();
}

function disconnectLive() {
  voice?.stop(false);
  generation += 1;
  const socket = liveSocket;
  liveSocket = null;
  connectionPromise = null;
  stopCamera();
  if (socket) socket.close();
  if (state) clearBusy();
}

async function createSession(resume = false) {
  if (resetting) return;
  resetting = true;
  newIntakeButton.disabled = true;
  const previous = sessionId || sessionStorage.getItem("intakeSession");
  disconnectLive();
  const epoch = generation;
  seenNotes.clear();
  setStatus("Connecting", "neutral");
  sessionId = null;
  setState(emptyState);
  selectedBookId = null;
  document.querySelector("#booksDialog").close();
  pageDate.textContent = `Library sweep · ${new Date().toLocaleDateString([], { month: "short", day: "numeric" })}`;
  try {
    let payload;
    if (resume && previous) {
      try { payload = await api(`/api/sessions/${previous}`); } catch (_) { sessionStorage.removeItem("intakeSession"); }
    } else if (previous) {
      await api(`/api/sessions/${previous}`, { method: "DELETE" }).catch(() => {});
    }
    payload = payload || await api("/api/sessions", { method: "POST" });
    if (epoch !== generation) return;
    sessionId = payload.session_id;
    sessionStorage.setItem("intakeSession", sessionId);
    setState(payload.state);
    const health = await api("/api/health");
    modelLabel.textContent = "Library sweep assistant";
    cameraConfig = health.camera;
    micButton.disabled = !health.voice_enabled;
    micButton.title = health.voice_enabled ? "Start or stop voice" : "Set DEEPGRAM_API_KEY and restart to enable voice";
    setStatus(payload.has_api_key ? "Ready" : "API key required", payload.has_api_key ? "" : "danger");
    textInput.focus();
  } catch (error) {
    setStatus("Backend unavailable", "danger");
    appendSystem(error.message);
  } finally {
    resetting = false;
    newIntakeButton.disabled = false;
  }
}

function connectLive() {
  if (connectionPromise) return connectionPromise;
  if (!sessionId || resetting) return Promise.reject(new Error("Wait for the intake to be ready."));
  const epoch = generation;
  const socket = new WebSocket(`${WS_ORIGIN}/ws/live?session_id=${encodeURIComponent(sessionId)}`);
  liveSocket = socket;
  connectionPromise = new Promise((resolve, reject) => {
    let ready = false;
    const timeout = setTimeout(() => { reject(new Error("Live connection timed out.")); socket.close(); }, 20000);
    const current = () => epoch === generation && liveSocket === socket;
    socket.onmessage = (event) => {
      if (!current()) return;
      const message = JSON.parse(event.data);
      if (message.session_id !== sessionId) return;
      if (message.type === "ready") {
        ready = true;
        clearTimeout(timeout);
        setStatus("Live", "");
        resolve();
      } else if (message.type === "session") {
        modelLabel.textContent = "Library sweep assistant";
      } else if (message.type === "voice_state") {
        voice?.setReady(message.enabled);
        if (!message.enabled) {
          state.transcript = state.transcript.filter(t => !(t.speaker === 'Claimant' && t.streaming));
          render();
        }
      } else if (message.type === "interrupted") {
        voice?.interrupt();
      } else if (message.type === "audio_start") {
        voice?.begin(message.epoch);
      } else if (message.type === "audio_end") {
        voice?.end(message.epoch);
      } else if (message.type === "audio") {
        voice?.play(message);
      } else if (message.type === "processing") {
        processing = message.active;
        render();
      } else if (message.type === "transcript") {
        upsertStreamingTurn(message.speaker, message.text, message.final, message.id);
      } else if (message.type === "tool") {
        applyToolEvent(message);

      } else if (message.type === "state") {
        applyServerState(message.state);
      } else if (message.type === "error") {
        processing = false;
        appendSystem(message.message);
        if (!ready) { reject(new Error(message.message)); socket.close(); }
      }
    };
    socket.onerror = () => { clearTimeout(timeout); reject(new Error("Live connection failed.")); socket.close(); };
    socket.onclose = () => {
      clearTimeout(timeout);
      if (!ready) reject(new Error("Live connection ended before it was ready."));
      if (!current()) return;
      liveSocket = null;
      connectionPromise = null;
      stopCamera();
      voice?.stop(false);
      clearBusy();
      setStatus("Disconnected — reconnect to continue", "warning");
    };
  });
  return connectionPromise;
}


async function sendClaimantTurn(text) {
  voice?.interrupt();
  const epoch = generation;
  try {
    await connectLive();
    if (epoch !== generation) return;
    const id = crypto.randomUUID();
    liveSocket.send(JSON.stringify({ type: "text", text, id }));
    upsertStreamingTurn("Claimant", text, true, id);
    processing = true;
    render();
  } catch (error) {
    if (epoch !== generation) return;
    textInput.value = text;
    appendSystem(`${error.message} Your message is back in the input; send it again to retry.`);
  }
}



async function arDepthSupported() {
  // WebXR depth gives real metric scale on supported Android phones; everything else uses the plain camera.
  try { return Boolean(navigator.xr && window.XRWebGLBinding && await navigator.xr.isSessionSupported("immersive-ar")); }
  catch { return false; }
}

function depthGrid(depthInfo, width, height) {
  // Sample depth on a coarse grid of normalized view points, in millimetres (0 = unknown).
  const mm = new Uint16Array(width * height);
  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      let metres = 0;
      try { metres = depthInfo.getDepthInMeters((col + 0.5) / width, (row + 0.5) / height); } catch { metres = 0; }
      mm[row * width + col] = Number.isFinite(metres) ? Math.min(65535, Math.max(0, Math.round(metres * 1000))) : 0;
    }
  }
  let binary = "";
  const bytes = new Uint8Array(mm.buffer);
  for (let i = 0; i < bytes.length; i += 8192) binary += String.fromCharCode(...bytes.subarray(i, i + 8192));
  return btoa(binary);
}

function sendArFrame(gl, framebuffer, texture, view, depthInfo) {
  if (!liveSocket || liveSocket.readyState !== WebSocket.OPEN || liveSocket.bufferedAmount > 512000) return;
  const { width, height } = view.camera;
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  const pixels = new Uint8ClampedArray(width * height * 4);
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  // readPixels returns rows bottom-up; flip into a top-down canvas before scaling to the JPEG size.
  const source = document.createElement("canvas");
  source.width = width; source.height = height;
  const sourceCtx = source.getContext("2d");
  const flipped = new ImageData(width, height);
  for (let row = 0; row < height; row++) flipped.data.set(pixels.subarray((height - 1 - row) * width * 4, (height - row) * width * 4), row * width * 4);
  sourceCtx.putImageData(flipped, 0, 0);
  const jpegWidth = cameraConfig.frame_width;
  const jpegHeight = Math.round(height * jpegWidth / width);
  frameCanvas.width = jpegWidth; frameCanvas.height = jpegHeight;
  frameCanvas.getContext("2d").drawImage(source, 0, 0, jpegWidth, jpegHeight);
  const dataUrl = frameCanvas.toDataURL("image/jpeg", cameraConfig.jpeg_quality);
  // Intrinsics in JPEG pixels from the column-major projection matrix.
  const p = view.projectionMatrix;
  const depth = {
    width: cameraConfig.depth_grid_width, height: cameraConfig.depth_grid_height,
    mm: depthGrid(depthInfo, cameraConfig.depth_grid_width, cameraConfig.depth_grid_height),
    fx: p[0] * jpegWidth / 2, fy: p[5] * jpegHeight / 2,
    cx: (1 - p[8]) * jpegWidth / 2, cy: (1 + p[9]) * jpegHeight / 2,
  };
  liveSocket.send(JSON.stringify({ type: "video", data: dataUrl.split(",")[1], depth }));
}

async function startArCamera() {
  const session = await navigator.xr.requestSession("immersive-ar", {
    requiredFeatures: ["depth-sensing", "camera-access", "dom-overlay"],
    depthSensing: { usagePreference: ["cpu-optimized"], dataFormatPreference: ["luminance-alpha", "float32"] },
    domOverlay: { root: document.body },
  });
  const glCanvas = document.createElement("canvas");
  const gl = glCanvas.getContext("webgl2", { xrCompatible: true });
  session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) });
  const space = await session.requestReferenceSpace("local");
  const binding = new XRWebGLBinding(session, gl);
  const framebuffer = gl.createFramebuffer();
  let lastSent = -Infinity;
  const onFrame = (time, frame) => {
    session.requestAnimationFrame(onFrame);
    if (time - lastSent < cameraConfig.frame_interval_ms) return;
    const view = frame.getViewerPose(space)?.views?.[0];
    if (!view?.camera) return;
    const depthInfo = frame.getDepthInformation(view);
    const texture = binding.getCameraImage(view.camera);
    if (!depthInfo || !texture) return;
    lastSent = time;
    sendArFrame(gl, framebuffer, texture, view, depthInfo);
  };
  session.requestAnimationFrame(onFrame);
  session.addEventListener("end", () => { arSession = null; stopCamera(); }, { once: true });
  return session;
}

async function startCamera() {
  if (cameraPending || cameraStream) return;
  const epoch = generation;
  if (await arDepthSupported()) {
    try {
      cameraPending = true;
      await connectLive();
      cameraConfig = cameraConfig || (await api("/api/health")).camera;
      arSession = await startArCamera();
      // A track-less stand-in keeps the existing "camera is on" checks working in AR mode.
      cameraStream = { getTracks: () => [], ar: true };
      liveSocket.send(JSON.stringify({ type: "camera_state", enabled: true }));
      cameraButton.classList.add("active");
      cameraButton.querySelector(".round-label").textContent = "Stop camera";
      setStatus("AR camera on, measuring with depth", "");
      return;
    } catch (error) {
      appendSystem(`AR depth unavailable (${error.message}); using the normal camera.`);
      arSession = null;
    } finally { cameraPending = false; }
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    appendSystem("This browser cannot access a camera.");
    return;
  }
  try {
    cameraPending = true;
    await connectLive();
    cameraConfig = cameraConfig || (await api("/api/health")).camera;
    if (!cameraConfig) throw new Error("Camera configuration is unavailable.");
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: cameraConfig.ideal_width }, height: { ideal: cameraConfig.ideal_height }, facingMode: "environment" } });
    if (epoch !== generation || !liveSocket || liveSocket.readyState !== WebSocket.OPEN) { stream.getTracks().forEach(track => track.stop()); return; }
    cameraStream = stream;
    stream.getVideoTracks().forEach(track => track.addEventListener("ended", stopCamera, { once: true }));
    liveSocket.send(JSON.stringify({ type: "camera_state", enabled: true }));
    cameraPreview.srcObject = cameraStream;
    cameraStage.hidden = false;
    cameraButton.classList.add("active");
    cameraButton.querySelector(".round-label").textContent = "Stop camera";
    frameTimer = window.setInterval(sendFrame, cameraConfig.frame_interval_ms);
    setStatus("Camera on, agent can see", "");
  } catch (error) {
    const denied = error.name === "NotAllowedError" || /denied|permission/i.test(error.message);
    appendSystem(denied ? "Camera access was denied. Allow it for this site to show the damage." : `Camera failed: ${error.message}`);
    stopCamera();
  } finally { cameraPending = false; }
}

function stopCamera() {
  const wasOn = Boolean(cameraStream);
  if (arSession) { const session = arSession; arSession = null; session.end().catch(() => {}); }
  if (frameTimer) window.clearInterval(frameTimer);
  frameTimer = null;
  cameraStream?.getTracks().forEach((track) => track.stop());
  cameraStream = null;
  cameraPreview.srcObject = null;
  cameraStage.hidden = true;
  cameraButton.classList.remove("active");
  cameraButton.querySelector(".round-label").textContent = "Show camera";
  if (wasOn && liveSocket?.readyState === WebSocket.OPEN) {
    liveSocket.send(JSON.stringify({ type: "camera_state", enabled: false }));
    setStatus("Camera off, describing the scene", "");
  }
}

function sendFrame() {
  if (!cameraStream || !liveSocket || liveSocket.readyState !== WebSocket.OPEN) return;
  if (!cameraPreview.videoWidth || liveSocket.bufferedAmount > 512000) return;
  const scale = cameraConfig.frame_width / cameraPreview.videoWidth;
  frameCanvas.width = cameraConfig.frame_width;
  frameCanvas.height = Math.round(cameraPreview.videoHeight * scale);
  const ctx = frameCanvas.getContext("2d");
  ctx.drawImage(cameraPreview, 0, 0, frameCanvas.width, frameCanvas.height);
  const dataUrl = frameCanvas.toDataURL("image/jpeg", cameraConfig.jpeg_quality);
  liveSocket.send(JSON.stringify({ type: "video", data: dataUrl.split(",")[1] }));
}




micButton.addEventListener("click", async () => {
  if (voice?.enabled) { voice.stop(); return; }
  const epoch = generation;
  try {
    await connectLive();
    if (epoch === generation) await voice?.start();
  } catch (error) { appendSystem(error.message); }
});
cameraButton.addEventListener("click", () => (cameraStream ? stopCamera() : startCamera()));
newIntakeButton.addEventListener("click", () => createSession(false));
window.addEventListener("pagehide", disconnectLive);
textForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const value = textInput.value.trim();
  if (!value) return;
  textInput.value = "";
  sendClaimantTurn(value);
});
document.querySelector("#openPacket").addEventListener("click", () => { renderBookInventory(); document.querySelector("#booksDialog").showModal(); });

document.querySelector("#openBooks").addEventListener("click", () => { renderBookInventory(); document.querySelector("#booksDialog").showModal(); });
document.querySelector("#closeBooks").addEventListener("click", () => { document.querySelector("#booksDialog").close(); document.querySelector("#openBooks").focus(); });
document.querySelector("#reviewTab").addEventListener("click", () => { bookTab = "review"; selectedBookId = null; renderBookInventory(); });
document.querySelector("#identifiedTab").addEventListener("click", () => { bookTab = "identified"; selectedBookId = null; renderBookInventory(); });
document.querySelector("#bookInventory").addEventListener("click", event => { const button = event.target.closest("[data-book-id]"); if (button) { selectedBookId = button.dataset.bookId; compareBookId = null; renderBookDetail(); } });
document.querySelector("#bookDetail").addEventListener("click", event => { const button = event.target.closest("[data-book-action]"); if (button) void reviewBook(button.dataset.bookAction); });

document.querySelector("#bookDetail").addEventListener("change", event => { if (event.target.id === "compareBook") { compareBookId = event.target.value; renderBookDetail(); } });

createSession(true);
