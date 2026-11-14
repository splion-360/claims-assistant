// Read-only tables of what the sweep has recorded, refreshed while it runs.
const API_ORIGIN = window.location.origin;
const REFRESH_MS = 4000;

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function cell(value) {
  return value === null || value === undefined || value === "" ? '<span class="unknown">—</span>' : escapeHtml(value);
}

function centimetres(value) {
  return value ? `${Number(value).toFixed(1)} cm` : '<span class="unknown">not measured</span>';
}

// A figure is shown only with the source behind it; otherwise the line says why not.
function price(valuation, missing) {
  if (!valuation) return `<span class="unknown">${escapeHtml(missing)}</span>`;
  if (valuation.needs_appraisal) return '<span class="unknown">awaiting appraisal</span>';
  const amount = valuation.replacement;
  if (!amount) return '<span class="unknown">no source</span>';
  const converted = amount.converted ? ` · converted from ${escapeHtml(amount.converted_from)}` : "";
  const label = `${escapeHtml(amount.currency)} ${Number(amount.amount).toFixed(2)}`;
  const source = amount.url
    ? `<a href="${escapeHtml(amount.url)}" target="_blank" rel="noopener">${escapeHtml(amount.source)}</a>`
    : escapeHtml(amount.source);
  return `<span class="price">${label}</span><br><small>${source}${converted}</small>`;
}

// Art waits for a person; everything else waits for a retail provider that is not configured.
function itemPrice(item) {
  if (item.needs_appraisal) return '<span class="unknown">awaiting appraisal</span>';
  return price(item.valuation, "no retail source configured");
}

function statusTag(entry) {
  if (entry.needs_appraisal) return '<span class="tag appraisal">appraisal</span>';
  if (entry.needs_review) return '<span class="tag review">review</span>';
  if (entry.excluded) return '<span class="tag">excluded</span>';
  return '<span class="tag ok">recorded</span>';
}

// Detection issues are recorded as codes; a reviewer reads words.
const ISSUE_WORDS = {
  blur: "blurred", glare: "glare", dark: "too dark", occlusion: "partly hidden",
  cut_off: "cut off", unreadable_text: "text unreadable", uncertain_book: "may not be a book",
  uncertain_item: "may not be this kind of object",
};

function readable(reason) {
  return ISSUE_WORDS[reason] || reason.replace(/_/g, " ");
}

function reasons(entry) {
  const list = (entry.review_reasons || []).map(readable);
  return list.length ? `<span class="reasons">${escapeHtml(list.join(" · "))}</span>` : cell("");
}

// A crop of the sighting, so a line can be judged against what the camera saw.
function thumbnail(sessionId, kind, observationId, label) {
  if (!observationId) return '<span class="unknown">—</span>';
  const url = `${API_ORIGIN}/api/sessions/${encodeURIComponent(sessionId)}/${kind}/evidence/${encodeURIComponent(observationId)}`;
  return `<img src="${escapeHtml(url)}" alt="${escapeHtml(label)}" loading="lazy">`;
}

function size(book) {
  if (!book.spine_height_cm && !book.spine_thickness_cm) return '<span class="unknown">not measured</span>';
  return `${Number(book.spine_height_cm || 0).toFixed(1)} &times; ${Number(book.spine_thickness_cm || 0).toFixed(1)} cm`;
}

function renderBooks(sessionId, books) {
  document.querySelector("#bookCount").textContent =
    `${books.length} recorded · ${books.filter((b) => b.title).length} identified`;
  document.querySelector("#books").innerHTML = books.length
    ? books.map((book) => {
        const shot = (book.observations || []).at(-1)?.id;
        return `<tr>
        <td class="shot">${thumbnail(sessionId, "books", shot, book.title || "Unidentified book")}</td>
        <td>${statusTag(book)}</td>
        <td><strong>${cell(book.title)}</strong><br><small>${cell(book.author)}</small>
          ${book.catalogue?.isbn ? `<br><small>ISBN ${escapeHtml(book.catalogue.isbn)}</small>` : ""}
          <br><small>${Number(book.observation_count)} sighting(s)</small></td>
        <td>${size(book)}</td>
        <td>${price(book.valuation, "not priced yet")}</td>
        <td>${reasons(book)}</td></tr>`;
      }).join("")
    : '<tr><td colspan="6" class="empty">No books recorded yet.</td></tr>';
}

function renderItems(sessionId, items) {
  document.querySelector("#itemCount").textContent = `${items.length} recorded`;
  document.querySelector("#items").innerHTML = items.length
    ? items.map((item) => `<tr>
        <td class="shot">${thumbnail(sessionId, "items", item.observation_ids?.at(-1), item.description || "Object")}</td>
        <td>${statusTag(item)}</td>
        <td><strong>${cell(item.category)}</strong><br><small>${cell(item.description)}</small>
          ${item.brand_model ? `<br><small>${escapeHtml(item.brand_model)}</small>` : ""}
          ${item.material ? `<br><small>${escapeHtml(item.material)}</small>` : ""}
          <br><small>${Number(item.observation_count)} sighting(s)</small></td>
        <td>${itemPrice(item)}</td>
        <td>${reasons(item)}
          <div class="actions" data-item="${escapeHtml(item.id)}">
            <button type="button" data-action="confirm">Confirm</button>
            <button type="button" data-action="appraise">Appraise</button>
            <button type="button" data-action="exclude">Not mine</button>
          </div></td></tr>`).join("")
    : '<tr><td colspan="5" class="empty">No non-book contents recorded yet.</td></tr>';
}

// Settling a review is the one thing this page writes back.
async function settleItem(sessionId, itemId, action, button) {
  button.disabled = true;
  try {
    const response = await fetch(
      `${API_ORIGIN}/api/sessions/${encodeURIComponent(sessionId)}/items/${encodeURIComponent(itemId)}/review`,
      { method: "POST", credentials: "include",
        headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action }) });
    if (!response.ok) throw new Error(`The sweep refused that (${response.status}).`);
    renderItems(sessionId, (await response.json()).state.item_inventory || []);
  } catch (error) {
    document.querySelector("#meta").textContent = error.message;
    button.disabled = false;
  }
}

document.querySelector("#items").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  const sessionId = currentSession();
  if (sessionId) settleItem(sessionId, button.parentElement.dataset.item, button.dataset.action, button);
});

function currentSession() {
  return new URLSearchParams(window.location.search).get("session")
    || sessionStorage.getItem("intakeSession");
}

async function refresh() {
  const sessionId = currentSession();
  const meta = document.querySelector("#meta");
  if (!sessionId) { meta.textContent = "No sweep open. Start one on the main page first."; return; }
  try {
    const response = await fetch(`${API_ORIGIN}/api/sessions/${sessionId}`, { credentials: "include" });
    if (!response.ok) throw new Error(`The sweep is no longer open (${response.status}).`);
    const state = (await response.json()).state;
    renderBooks(sessionId, state.book_inventory || []);
    renderItems(sessionId, state.item_inventory || []);
    meta.textContent = `Sweep ${sessionId.slice(0, 8)} · updated ${new Date().toLocaleTimeString()}`;
  } catch (error) {
    meta.textContent = error.message;
  }
}

refresh();
window.setInterval(refresh, REFRESH_MS);
