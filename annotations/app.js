const DATA_URL = "finding_rubric/items_blinded.jsonl";
const STORAGE_PREFIX = "baikal-finding-rubric-v1:";
const REQUIRED_FIELDS = ["grounded", "relevance", "distinctness", "report_usefulness"];
const CSV_FIELDS = [
  "sample_id",
  "annotator_id",
  "grounded",
  "relevance",
  "distinctness",
  "report_usefulness",
  "grounded_notes",
  "relevance_notes",
  "distinctness_notes",
  "report_usefulness_notes",
];

const RUBRICS = [
  {
    key: "grounded",
    title: "Groundedness",
    prompt: "Is the finding supported by the supplied evidence?",
    options: [
      ["no", "No", "Unsupported, contradicted, not tied to evidence, or absence-only"],
      ["yes", "Yes", "Requested factual information is supported by the supplied evidence"],
    ],
  },
  {
    key: "relevance",
    title: "Relevance",
    prompt: "How directly does this finding contribute to the research question?",
    options: [
      ["none", "None", "Unrelated or does not help answer the research question"],
      ["minimal", "Minimal", "Barely related; mostly off-topic"],
      ["partial", "Partial", "Tangential; misses the main analytical goals"],
      ["substantial", "Substantial", "Mostly relevant with minor gaps"],
      ["full", "Full", "Directly addresses an important part of the question"],
    ],
  },
  {
    key: "distinctness",
    title: "Distinctness",
    prompt: "How much new information does it add beyond earlier findings?",
    options: [
      ["none", "None", "Duplicate or near-verbatim rephrase"],
      ["minimal", "Minimal", "Mostly repeats with trivial wording changes"],
      ["partial", "Partial", "Heavy overlap but adds a small new detail"],
      ["substantial", "Substantial", "Mostly new angle or evidence; some overlap"],
      ["full", "Full", "Clearly new insight not covered earlier"],
    ],
  },
  {
    key: "report_usefulness",
    title: "Report usefulness",
    prompt: "Is this finding worth including in a final research report?",
    options: [
      ["none", "None", "Noise, redundant, or only states information was not found"],
      ["minimal", "Minimal", "Marginally informative; unlikely to help the reader"],
      ["partial", "Partial", "Somewhat helpful but low priority"],
      ["substantial", "Substantial", "Useful; requested aspect or complementary insight"],
      ["full", "Full", "Highly useful; directly answers or adds valuable insight"],
    ],
  },
];

let items = [];
let annotatorId = "";
let state = null;
let saveTimer = null;

const setup = document.querySelector("#setup");
const setupForm = document.querySelector("#setup-form");
const setupInput = document.querySelector("#setup-annotator");
const workspace = document.querySelector("#workspace");
const itemNav = document.querySelector("#item-nav");

function emptyRating() {
  return {
    grounded: "",
    relevance: "",
    distinctness: "",
    report_usefulness: "",
    grounded_notes: "",
    relevance_notes: "",
    distinctness_notes: "",
    report_usefulness_notes: "",
  };
}

function storageKey(id) {
  return `${STORAGE_PREFIX}${id}`;
}

function loadState(id) {
  try {
    const stored = JSON.parse(localStorage.getItem(storageKey(id)));
    if (stored?.ratings && stored.version === 1) {
      return {
        ...stored,
        currentIndex: Math.min(Math.max(stored.currentIndex || 0, 0), items.length - 1),
      };
    }
  } catch (error) {
    console.warn("Could not read saved progress", error);
  }
  return {
    version: 1,
    annotatorId: id,
    currentIndex: 0,
    ratings: {},
    updatedAt: new Date().toISOString(),
  };
}

function saveState() {
  if (!annotatorId || !state) return;
  state.updatedAt = new Date().toISOString();
  localStorage.setItem(storageKey(annotatorId), JSON.stringify(state));
  const status = document.querySelector("#save-status");
  status.textContent = "Saved";
  status.classList.add("save-status");
}

function queueSave() {
  const status = document.querySelector("#save-status");
  status.textContent = "Saving…";
  status.classList.remove("save-status");
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveState, 120);
}

function ratingFor(sampleId) {
  if (!state.ratings[sampleId]) state.ratings[sampleId] = emptyRating();
  return state.ratings[sampleId];
}

function isComplete(sampleId) {
  const rating = state?.ratings?.[sampleId];
  return Boolean(rating && REQUIRED_FIELDS.every((field) => rating[field]));
}

function completedCount() {
  return items.filter((item) => isComplete(item.sample_id)).length;
}

async function loadItems() {
  if (Array.isArray(window.ANNOTATION_ITEMS)) {
    items = window.ANNOTATION_ITEMS;
    if (!items.length) throw new Error("The annotation dataset is empty.");
    return;
  }
  const response = await fetch(DATA_URL);
  if (!response.ok) throw new Error(`Could not load ${DATA_URL} (${response.status})`);
  const text = await response.text();
  items = text
    .split(/\r?\n/)
    .filter(Boolean)
    .map((line) => JSON.parse(line));
  if (!items.length) throw new Error("The annotation dataset is empty.");
}

function showToast(message) {
  const toast = document.querySelector("#toast");
  toast.textContent = message;
  toast.classList.add("visible");
  setTimeout(() => toast.classList.remove("visible"), 2200);
}

function updateProgress() {
  const complete = completedCount();
  document.querySelector("#progress-text").textContent =
    `${complete} of ${items.length} complete`;
  document.querySelector("#progress-bar").style.width =
    `${items.length ? (100 * complete) / items.length : 0}%`;
  document.querySelector("#annotator-label").textContent = annotatorId || "Annotator";
}

function renderItemNav() {
  itemNav.replaceChildren();
  items.forEach((item, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "item-button";
    button.textContent = String(index + 1);
    button.title = `${item.sample_id}${isComplete(item.sample_id) ? " — complete" : ""}`;
    button.setAttribute("aria-label", `Item ${index + 1}`);
    if (isComplete(item.sample_id)) button.classList.add("complete");
    if (index === state.currentIndex) {
      button.classList.add("current");
      button.setAttribute("aria-current", "page");
    }
    button.addEventListener("click", () => goToItem(index));
    itemNav.append(button);
  });
  itemNav.querySelector(".current")?.scrollIntoView({ block: "nearest", inline: "nearest" });
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function stringifyCell(cell) {
  if (Array.isArray(cell)) return stringifyCell(cell[0]);
  if (cell && typeof cell === "object" && "value" in cell) return String(cell.value ?? "");
  if (cell && typeof cell === "object") return JSON.stringify(cell);
  return String(cell ?? "");
}

function normalizedRows(table) {
  const columns = table.columns || [];
  const rows = table.rows || [];
  if (!columns.length || !rows.length) return [];

  if (rows.every((row) => Array.isArray(row))) {
    return rows.map((row) => row.map(stringifyCell));
  }
  if (rows.every((row) => row && typeof row === "object" && !("value" in row))) {
    return rows.map((row) => columns.map((column) => stringifyCell(row[column])));
  }
  if (
    rows.every((row) => row && typeof row === "object" && "value" in row) &&
    rows.length % columns.length === 0
  ) {
    const output = [];
    for (let index = 0; index < rows.length; index += columns.length) {
      output.push(rows.slice(index, index + columns.length).map(stringifyCell));
    }
    return output;
  }
  return [];
}

function renderTableEvidence(table) {
  const block = element("section", "evidence-block");
  const heading = element(
    "h3",
    "evidence-title",
    `${table.id}${table.title ? ` — ${table.title}` : ""}`,
  );
  block.append(heading);

  if (table.unavailable) {
    block.append(element("p", "evidence-meta", "Table content unavailable."));
    return block;
  }

  const rows = normalizedRows(table);
  if (!rows.length) {
    const pre = element("pre", "", JSON.stringify(table.rows || [], null, 2));
    block.append(pre);
    return block;
  }

  const wrap = element("div", "table-wrap");
  const tableNode = document.createElement("table");
  const thead = document.createElement("thead");
  const headerRow = document.createElement("tr");
  (table.columns || []).forEach((column) => {
    headerRow.append(element("th", "", String(column)));
  });
  thead.append(headerRow);
  tableNode.append(thead);

  const tbody = document.createElement("tbody");
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    row.forEach((cell) => tr.append(element("td", "", cell)));
    tbody.append(tr);
  });
  tableNode.append(tbody);
  wrap.append(tableNode);
  block.append(wrap);
  if (table.rows_truncated) {
    block.append(
      element("p", "evidence-meta", `Showing 100 of ${table.total_rows} stored rows.`),
    );
  }
  return block;
}

function renderEvidence(container, item) {
  container.replaceChildren();
  const evidence = item.evidence || {};
  const tables = evidence.tables_cited || [];
  const passages = evidence.passages_cited || [];
  const retrieved = evidence.retrieval_evidence || [];
  const sqlRows = evidence.sql_result_preview || [];

  tables.forEach((table) => container.append(renderTableEvidence(table)));

  passages.forEach((passage) => {
    const block = element("section", "evidence-block");
    block.append(
      element(
        "h3",
        "evidence-title",
        `${passage.id}${passage.title ? ` — ${passage.title}` : ""}`,
      ),
    );
    block.append(element("div", "evidence-text", passage.text || "(No text available)"));
    container.append(block);
  });

  retrieved.forEach((record, index) => {
    const block = element("section", "evidence-block");
    block.append(element("h3", "evidence-title", `Retrieved evidence ${index + 1}`));
    if (record.score !== null && record.score !== undefined) {
      block.append(element("div", "evidence-meta", `Retrieval score: ${record.score}`));
    }
    block.append(element("div", "evidence-text", record.text));
    container.append(block);
  });

  if (evidence.sql && evidence.sql !== "(none)") {
    const block = element("section", "evidence-block");
    block.append(element("h3", "evidence-title", "SQL execution"));
    block.append(
      element("div", "evidence-meta", `${evidence.sql_row_count || 0} rows returned`),
    );
    block.append(element("pre", "", evidence.sql));
    if (sqlRows.length) {
      block.append(element("pre", "", JSON.stringify(sqlRows, null, 2)));
    }
    container.append(block);
  } else if (sqlRows.length) {
    const block = element("section", "evidence-block");
    block.append(element("h3", "evidence-title", "SQL result preview"));
    block.append(element("pre", "", JSON.stringify(sqlRows, null, 2)));
    container.append(block);
  }

  if (!container.children.length) {
    container.append(element("p", "evidence-meta", "No inspectable evidence supplied."));
  }
  return { tables: tables.length, passages: passages.length, retrieved: retrieved.length };
}

function renderPreviousFindings(container, findings) {
  container.replaceChildren();
  if (!findings.length) {
    container.append(element("p", "evidence-meta", "No earlier findings."));
    return;
  }
  findings.forEach((finding, index) => {
    const block = element("section", "prior-finding");
    block.append(element("h3", "evidence-title", `Earlier finding ${index + 1}`));
    block.append(element("p", "evidence-meta", finding.sub_question || ""));
    block.append(element("p", "", finding.answer || ""));
    container.append(block);
  });
}

function renderRubrics(container, sampleId) {
  container.replaceChildren();
  const current = ratingFor(sampleId);
  RUBRICS.forEach((rubric) => {
    const card = element("section", "rubric-card");
    const header = element("div", "rubric-header");
    header.append(element("h2", "", rubric.title));
    header.append(element("p", "", rubric.prompt));
    card.append(header);

    const options = element("div", `options${rubric.options.length === 2 ? " binary" : ""}`);
    rubric.options.forEach(([value, label, hint]) => {
      const option = element("div", "option");
      const input = document.createElement("input");
      input.type = "radio";
      input.name = rubric.key;
      input.id = `${rubric.key}-${value}`;
      input.value = value;
      input.checked = current[rubric.key] === value;
      const optionLabel = document.createElement("label");
      optionLabel.htmlFor = input.id;
      optionLabel.append(element("strong", "", label));
      optionLabel.append(element("span", "", hint));
      option.append(input, optionLabel);
      options.append(option);
    });
    card.append(options);

    const notes = document.createElement("details");
    notes.className = "notes-toggle";
    if (current[`${rubric.key}_notes`]) notes.open = true;
    notes.append(element("summary", "", "Add optional note"));
    const textarea = document.createElement("textarea");
    textarea.name = `${rubric.key}_notes`;
    textarea.placeholder = "Explain an unclear or borderline rating…";
    textarea.value = current[`${rubric.key}_notes`] || "";
    notes.append(textarea);
    card.append(notes);
    container.append(card);
  });
}

function evidenceHint(counts) {
  const parts = [];
  if (counts.tables) parts.push(`${counts.tables} table${counts.tables === 1 ? "" : "s"}`);
  if (counts.passages) {
    parts.push(`${counts.passages} passage${counts.passages === 1 ? "" : "s"}`);
  }
  if (counts.retrieved) parts.push(`${counts.retrieved} retrieved excerpt${counts.retrieved === 1 ? "" : "s"}`);
  return parts.join(" · ") || "SQL evidence";
}

function renderCurrentItem() {
  const item = items[state.currentIndex];
  const fragment = document.querySelector("#item-template").content.cloneNode(true);

  fragment.querySelector("#item-position").textContent =
    `Item ${state.currentIndex + 1} of ${items.length}`;
  fragment.querySelector("#item-dataset").textContent = item.dataset;
  fragment.querySelector("#research-question").textContent = item.research_question;
  fragment.querySelector("#sub-question").textContent = item.sub_question;
  fragment.querySelector("#finding").textContent = item.finding;

  const counts = renderEvidence(fragment.querySelector("#evidence"), item);
  fragment.querySelector("#evidence-summary").textContent = evidenceHint(counts);
  renderPreviousFindings(fragment.querySelector("#previous-findings"), item.previous_findings || []);
  fragment.querySelector("#prior-count").textContent =
    `${(item.previous_findings || []).length} prior`;
  renderRubrics(fragment.querySelector("#rubric-fields"), item.sample_id);

  const form = fragment.querySelector("#rating-form");
  form.addEventListener("input", (event) => {
    const rating = ratingFor(item.sample_id);
    rating[event.target.name] = event.target.value;
    queueSave();
    updateProgress();
    renderItemNav();
    updateCompletionHint(item.sample_id);
  });

  const previous = fragment.querySelector("#previous-item");
  const next = fragment.querySelector("#next-item");
  previous.disabled = state.currentIndex === 0;
  next.textContent = state.currentIndex === items.length - 1 ? "Finish" : "Next →";
  previous.addEventListener("click", () => goToItem(state.currentIndex - 1));
  next.addEventListener("click", () => {
    if (state.currentIndex < items.length - 1) {
      goToItem(state.currentIndex + 1);
    } else if (completedCount() === items.length) {
      showToast("All items complete — export your ratings.");
    } else {
      goToNextIncomplete();
    }
  });

  workspace.replaceChildren(fragment);
  updateCompletionHint(item.sample_id);
  renderItemNav();
  updateProgress();
  window.scrollTo({ top: 0, behavior: "instant" });
}

function updateCompletionHint(sampleId) {
  const hint = document.querySelector("#completion-hint");
  if (!hint) return;
  hint.textContent = isComplete(sampleId)
    ? "Item complete. Progress saved automatically."
    : "Complete all four ratings to finish this item.";
  hint.style.color = isComplete(sampleId) ? "var(--success)" : "";
}

function goToItem(index) {
  if (index < 0 || index >= items.length) return;
  saveState();
  state.currentIndex = index;
  saveState();
  renderCurrentItem();
}

function goToNextIncomplete() {
  const next = items.findIndex(
    (item, index) => index > state.currentIndex && !isComplete(item.sample_id),
  );
  const wrapped = items.findIndex((item) => !isComplete(item.sample_id));
  const target = next >= 0 ? next : wrapped;
  if (target >= 0) goToItem(target);
  else showToast("All items are complete.");
}

function csvEscape(value) {
  const text = String(value ?? "");
  return /[",\r\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function exportRows() {
  return items.map((item) => {
    const rating = state.ratings[item.sample_id] || emptyRating();
    return {
      sample_id: item.sample_id,
      annotator_id: annotatorId,
      ...rating,
    };
  });
}

function download(content, filename, type) {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.append(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

function safeFilename(value) {
  return value.replace(/[^a-z0-9_-]+/gi, "-").replace(/^-|-$/g, "") || "annotator";
}

function exportCsv() {
  saveState();
  const rows = exportRows();
  const csv = [
    CSV_FIELDS.join(","),
    ...rows.map((row) => CSV_FIELDS.map((field) => csvEscape(row[field])).join(",")),
  ].join("\n");
  download(
    csv,
    `finding_ratings_${safeFilename(annotatorId)}.csv`,
    "text/csv;charset=utf-8",
  );
  showToast(`Exported ${completedCount()} complete ratings.`);
}

function exportJson() {
  saveState();
  const payload = {
    schema_version: 1,
    annotator_id: annotatorId,
    exported_at: new Date().toISOString(),
    completed: completedCount(),
    total: items.length,
    ratings: exportRows(),
  };
  download(
    JSON.stringify(payload, null, 2),
    `finding_ratings_${safeFilename(annotatorId)}.json`,
    "application/json",
  );
  showToast("Exported JSON backup.");
}

function openSetup() {
  setup.classList.remove("hidden");
  setupInput.value = annotatorId;
  setTimeout(() => setupInput.focus(), 0);
}

function startSession(id) {
  annotatorId = id.trim();
  if (!annotatorId) return;
  state = loadState(annotatorId);
  setup.classList.add("hidden");
  renderCurrentItem();
}

setupForm.addEventListener("submit", (event) => {
  event.preventDefault();
  startSession(setupInput.value);
});
document.querySelector("#change-annotator").addEventListener("click", () => {
  saveState();
  openSetup();
});
document.querySelector("#next-incomplete").addEventListener("click", goToNextIncomplete);
document.querySelector("#export-csv").addEventListener("click", exportCsv);
document.querySelector("#export-json").addEventListener("click", exportJson);

window.addEventListener("beforeunload", saveState);
document.addEventListener("keydown", (event) => {
  if (!state) return;
  if (event.target.matches("input, textarea")) return;
  if (event.key === "ArrowLeft") goToItem(state.currentIndex - 1);
  if (event.key === "ArrowRight") goToItem(state.currentIndex + 1);
});

loadItems()
  .then(() => {
    document.querySelector("#setup-submit").disabled = false;
    const lastAnnotator = localStorage.getItem(`${STORAGE_PREFIX}last-annotator`);
    if (lastAnnotator) setupInput.value = lastAnnotator;
    setupForm.addEventListener("submit", () => {
      localStorage.setItem(`${STORAGE_PREFIX}last-annotator`, setupInput.value.trim());
    });
    setupInput.focus();
  })
  .catch((error) => {
    setup.classList.add("hidden");
    workspace.innerHTML = `
      <div class="content-card">
        <h1>Could not load annotation items</h1>
        <p>${error.message}</p>
        <p>Start a local server from the repository root, then open
        <code>/annotation/</code>.</p>
      </div>`;
    console.error(error);
  });
