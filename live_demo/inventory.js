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

function reasons(entry) {
  const list = entry.review_reasons || [];
  return list.length ? `<span class="reasons">${escapeHtml(list.join(" · "))}</span>` : cell("");
}

function renderBooks(books) {
  document.querySelector("#bookCount").textContent =
    `${books.length} recorded · ${books.filter((b) => b.title).length} identified`;
  document.querySelector("#books").innerHTML = books.length
    ? books.map((book) => `<tr>
        <td>${statusTag(book)}</td>
        <td>${cell(book.title)}</td>
        <td>${cell(book.author)}</td>
        <td>${cell(book.catalogue?.isbn)}</td>
        <td class="num">${centimetres(book.spine_height_cm)}</td>
        <td class="num">${centimetres(book.spine_thickness_cm)}</td>
        <td>${price(book.valuation, "not priced yet")}</td>
        <td class="num">${cell(book.observation_count)}</td>
        <td>${reasons(book)}</td></tr>`).join("")
    : '<tr><td colspan="9" class="empty">No books recorded yet.</td></tr>';
}

function renderItems(items) {
  document.querySelector("#itemCount").textContent = `${items.length} recorded`;
  document.querySelector("#items").innerHTML = items.length
    ? items.map((item) => `<tr>
        <td>${statusTag(item)}</td>
        <td>${cell(item.category)}</td>
        <td>${cell(item.description)}</td>
        <td>${cell(item.material)}</td>
        <td>${cell(item.brand_model)}</td>
        <td>${itemPrice(item)}</td>
        <td class="num">${cell(item.observation_count)}</td>
        <td>${reasons(item)}</td></tr>`).join("")
    : '<tr><td colspan="8" class="empty">No non-book contents recorded yet.</td></tr>';
}

async function refresh() {
  const sessionId = new URLSearchParams(window.location.search).get("session")
    || sessionStorage.getItem("intakeSession");
  const meta = document.querySelector("#meta");
  if (!sessionId) { meta.textContent = "No sweep open. Start one on the main page first."; return; }
  try {
    const response = await fetch(`${API_ORIGIN}/api/sessions/${sessionId}`, { credentials: "include" });
    if (!response.ok) throw new Error(`The sweep is no longer open (${response.status}).`);
    const state = (await response.json()).state;
    renderBooks(state.book_inventory || []);
    renderItems(state.item_inventory || []);
    meta.textContent = `Sweep ${sessionId.slice(0, 8)} · updated ${new Date().toLocaleTimeString()}`;
  } catch (error) {
    meta.textContent = error.message;
  }
}

refresh();
window.setInterval(refresh, REFRESH_MS);
