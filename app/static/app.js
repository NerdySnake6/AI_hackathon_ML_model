const samples = [
  "1 сезон тьмы",
  "интерстелар смотреть онлайн",
  "10 троллейбус ижевск",
  "смотреть фильмы 2025 года онлайн",
  "кухня ремонт",
  "машаи медведь новые серии",
  "slovo pacana",
  "bynthcntkkfh",
];

const reviewQueue = [];
let lastResult = null;
let lastBatch = [];

const els = {
  serviceStatus: document.getElementById("serviceStatus"),
  sampleQueries: document.getElementById("sampleQueries"),
  singleQuery: document.getElementById("singleQuery"),
  batchQueries: document.getElementById("batchQueries"),
  labelSingle: document.getElementById("labelSingle"),
  labelBatch: document.getElementById("labelBatch"),
  sendToReview: document.getElementById("sendToReview"),
  downloadBatch: document.getElementById("downloadBatch"),
  downloadReview: document.getElementById("downloadReview"),
  clearReview: document.getElementById("clearReview"),
  resultPane: document.getElementById("resultPane"),
  decisionBadge: document.getElementById("decisionBadge"),
  messageBox: document.getElementById("messageBox"),
  reviewCount: document.getElementById("reviewCount"),
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatValue(value) {
  if (value === null || value === undefined || value === "") {
    return "—";
  }
  return escapeHtml(value);
}

function percent(value) {
  return `${Math.round((Number(value) || 0) * 100)}%`;
}

function videoLabel(result) {
  if (result.domain_label === "uncertain") {
    return "неясно";
  }
  return result.is_prof_video ? "да" : "нет";
}

function setMessage(text) {
  els.messageBox.textContent = text;
  els.messageBox.classList.toggle("visible", Boolean(text));
}

function setDecision(decision) {
  const value = decision || "ready";
  els.decisionBadge.textContent = value;
  els.decisionBadge.className = `badge ${value}`;
}

function showView(name) {
  document.querySelectorAll("[data-view]").forEach((view) => {
    view.classList.toggle("active", view.dataset.view === name);
  });
  document.querySelectorAll("[data-view-button]").forEach((button) => {
    button.classList.toggle("active", button.dataset.viewButton === name);
  });
  if (name === "review") {
    renderReview();
  }
}

async function healthCheck() {
  try {
    const response = await fetch("/health");
    if (!response.ok) {
      throw new Error("bad status");
    }
    els.serviceStatus.textContent = "API работает";
  } catch (error) {
    els.serviceStatus.textContent = "API недоступен";
  }
}

async function labelSingle() {
  const query = els.singleQuery.value.trim();
  if (!query) {
    setMessage("Пустой запрос не обрабатывается");
    return;
  }

  setMessage("");
  const response = await fetch("/label", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({query_text: query}),
  });
  if (!response.ok) {
    setMessage("Ошибка разметки запроса");
    return;
  }

  lastResult = await response.json();
  renderResult(lastResult);
  if (["review", "manual_required"].includes(lastResult.decision)) {
    addToReview(lastResult);
  }
}

async function labelBatch() {
  const queries = els.batchQueries.value
    .split("\n")
    .map((query) => query.trim())
    .filter(Boolean)
    .map((query_text, index) => ({query_id: `ui_${index + 1}`, query_text}));

  if (!queries.length) {
    setMessage("Нет запросов для обработки");
    return;
  }

  setMessage("");
  const response = await fetch("/label_batch", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({queries}),
  });
  if (!response.ok) {
    setMessage("Ошибка batch-разметки");
    return;
  }

  lastBatch = await response.json();
  lastBatch
    .filter((item) => ["review", "manual_required"].includes(item.decision))
    .forEach(addToReview);
  renderBatch(lastBatch);
}

function renderResult(result) {
  setDecision(result.decision);
  const candidates = result.candidates || [];
  const candidateRows = candidates.length
    ? candidates.map((candidate, index) => `
      <tr>
        <td>${index + 1}</td>
        <td>${escapeHtml(candidate.canonical_title)}</td>
        <td>${escapeHtml(candidate.content_type)}</td>
        <td>${formatValue(candidate.title_id)}</td>
        <td>${percent(candidate.rank_score)}</td>
        <td>${escapeHtml(candidate.matched_alias)}</td>
      </tr>
    `).join("")
    : `<tr><td colspan="6">Кандидатов нет</td></tr>`;

  const reasons = (result.reasons || []).length
    ? result.reasons.map((reason) => `<span class="chip">${escapeHtml(reason)}</span>`).join("")
    : `<span class="chip">no_reasons</span>`;

  els.resultPane.innerHTML = `
    <div class="summary">
      <div class="metric">
        <div class="metric-label">Видеоконтент</div>
        <div class="metric-value">${videoLabel(result)}</div>
      </div>
      <div class="metric">
        <div class="metric-label">Тип</div>
        <div class="metric-value">${formatValue(result.content_type)}</div>
      </div>
      <div class="metric">
        <div class="metric-label">Тайтл</div>
        <div class="metric-value">${formatValue(result.title)}</div>
      </div>
      <div class="metric">
        <div class="metric-label">Confidence</div>
        <div class="metric-value">${percent(result.confidence)}</div>
        <div class="confidence-track">
          <div class="confidence-fill" style="width: ${percent(result.confidence)}"></div>
        </div>
      </div>
    </div>
    <table>
      <tbody>
        <tr><th>Query</th><td>${escapeHtml(result.query)}</td></tr>
        <tr><th>Normalized</th><td>${escapeHtml(result.normalized_query)}</td></tr>
        <tr><th>Domain</th><td>${escapeHtml(result.domain_label)}</td></tr>
        <tr><th>Title ID</th><td>${formatValue(result.title_id)}</td></tr>
        <tr><th>Model</th><td>${escapeHtml(result.model_version)}</td></tr>
      </tbody>
    </table>
    <h2 class="section-title">Кандидаты</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Название</th>
            <th>Тип</th>
            <th>Title ID</th>
            <th>Score</th>
            <th>Alias</th>
          </tr>
        </thead>
        <tbody>${candidateRows}</tbody>
      </table>
    </div>
    <h2 class="section-title">Reasons</h2>
    <div class="chips">${reasons}</div>
  `;
}

function renderBatch(results) {
  setDecision("batch");
  const rows = results.map((item) => `
    <tr>
      <td>${escapeHtml(item.query)}</td>
      <td><span class="badge ${item.decision}">${escapeHtml(item.decision)}</span></td>
      <td>${item.is_prof_video ? "да" : "нет"}</td>
      <td>${formatValue(item.content_type)}</td>
      <td>${formatValue(item.title)}</td>
      <td>${formatValue(item.title_id)}</td>
      <td>${percent(item.confidence)}</td>
    </tr>
  `).join("");
  const autoCount = results.filter((item) => item.decision === "auto_accept").length;
  const reviewCount = results.filter((item) => ["review", "manual_required"].includes(item.decision)).length;
  const nonVideoCount = results.filter((item) => item.decision === "non_video").length;

  els.resultPane.innerHTML = `
    <div class="summary">
      <div class="metric"><div class="metric-label">Всего</div><div class="metric-value">${results.length}</div></div>
      <div class="metric"><div class="metric-label">Auto</div><div class="metric-value">${autoCount}</div></div>
      <div class="metric"><div class="metric-label">Review</div><div class="metric-value">${reviewCount}</div></div>
      <div class="metric"><div class="metric-label">Non-video</div><div class="metric-value">${nonVideoCount}</div></div>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Query</th>
            <th>Decision</th>
            <th>Video</th>
            <th>Type</th>
            <th>Title</th>
            <th>Title ID</th>
            <th>Confidence</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

function addToReview(result) {
  const exists = reviewQueue.some((item) => item.query === result.query);
  if (!exists) {
    reviewQueue.push(result);
  }
  updateReviewCount();
}

function updateReviewCount() {
  els.reviewCount.textContent = reviewQueue.length
    ? `В очереди: ${reviewQueue.length}`
    : "Очередь пуста";
}

function renderReview() {
  setDecision("review_queue");
  updateReviewCount();
  if (!reviewQueue.length) {
    els.resultPane.innerHTML = `<div class="empty">Очередь проверки пуста</div>`;
    return;
  }

  const rows = reviewQueue.map((item, index) => `
    <tr>
      <td>${index + 1}</td>
      <td>${escapeHtml(item.query)}</td>
      <td>${escapeHtml(item.decision)}</td>
      <td>${formatValue(item.title)}</td>
      <td>${formatValue(item.title_id)}</td>
      <td>${percent(item.confidence)}</td>
    </tr>
  `).join("");

  els.resultPane.innerHTML = `
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Query</th>
            <th>Decision</th>
            <th>Title</th>
            <th>Title ID</th>
            <th>Confidence</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

function toCsv(rows) {
  const header = ["query", "decision", "is_prof_video", "content_type", "title", "title_id", "confidence"];
  const lines = [header.join(",")];
  rows.forEach((item) => {
    const values = [
      item.query,
      item.decision,
      item.is_prof_video,
      item.content_type || "",
      item.title || "",
      item.title_id || "",
      item.confidence,
    ].map((value) => `"${String(value).replaceAll('"', '""')}"`);
    lines.push(values.join(","));
  });
  return lines.join("\n");
}

function downloadCsv(filename, rows) {
  if (!rows.length) {
    setMessage("Нет данных для скачивания");
    return;
  }
  const blob = new Blob([toCsv(rows)], {type: "text/csv;charset=utf-8"});
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

document.querySelectorAll("[data-view-button]").forEach((button) => {
  button.addEventListener("click", () => showView(button.dataset.viewButton));
});

samples.forEach((sample) => {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = sample;
  button.title = sample;
  button.addEventListener("click", () => {
    els.singleQuery.value = sample;
    labelSingle();
  });
  els.sampleQueries.appendChild(button);
});

els.labelSingle.addEventListener("click", labelSingle);
els.labelBatch.addEventListener("click", labelBatch);
els.sendToReview.addEventListener("click", () => {
  if (!lastResult) {
    setMessage("Сначала разметьте запрос");
    return;
  }
  addToReview(lastResult);
  setMessage("Запрос добавлен в review");
});
els.downloadBatch.addEventListener("click", () => downloadCsv("batch_predictions.csv", lastBatch));
els.downloadReview.addEventListener("click", () => downloadCsv("review_queue.csv", reviewQueue));
els.clearReview.addEventListener("click", () => {
  reviewQueue.splice(0, reviewQueue.length);
  renderReview();
});

healthCheck();
labelSingle();
