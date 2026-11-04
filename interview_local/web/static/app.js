const SUPPORTED_EXTENSIONS = [".wav", ".mp3", ".m4a"];
const UNSUPPORTED_FORMAT_MESSAGE = "Please select a WAV, MP3 or M4A file.";
const POLL_INTERVAL_MS = 1000;
const PHASE_ORDER = ["speech_recognition", "speaker_recognition", "llm"];
const REASON_LABELS = {
  overlapping_speech: "simultaneous speech",
  ambiguous_speaker_boundary: "unclear speaker boundary",
  no_matching_speaker: "no speaker match",
  unassigned: "no speaker assigned",
  speaker_changed_by_llm: "speaker changed by LLM",
};

const $ = (id) => document.getElementById(id);
let currentDocument = null;
let currentSelection = null;

class ApiError extends Error {}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const body = await response.json();
  if (!response.ok) {
    throw new ApiError(body.error || `Request failed (${response.status})`);
  }
  return body;
}

function sendJson(path, method, value) {
  return api(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(value),
  });
}

function showView(id) {
  for (const view of ["start-view", "progress-view", "editor-view"]) {
    $(view).hidden = view !== id;
  }
}

function showError(element, message) {
  element.textContent = message;
  element.hidden = !message;
}

function formatTime(seconds) {
  const total = Math.floor(seconds);
  const hours = String(Math.floor(total / 3600)).padStart(2, "0");
  const minutes = String(Math.floor((total % 3600) / 60)).padStart(2, "0");
  const secs = String(total % 60).padStart(2, "0");
  return `${hours}:${minutes}:${secs}`;
}

async function showStart() {
  showView("start-view");
  const list = $("run-list");
  list.replaceChildren();
  const { runs } = await api("/api/runs");
  if (!runs.length) {
    list.append(Object.assign(document.createElement("li"), { textContent: "No transcripts yet." }));
  }
  for (const run of runs) {
    const item = document.createElement("li");
    const button = Object.assign(document.createElement("button"), { type: "button", textContent: run.run_id });
    button.addEventListener("click", () => openRun(run.run_id).catch(reportStartError));
    item.append(button);
    list.append(item);
  }
}

function reportStartError(error) {
  showView("start-view");
  showError($("upload-error"), error.message);
}

async function startUpload(event) {
  event.preventDefault();
  showError($("upload-error"), "");
  const file = $("audio-file").files[0];
  const extension = file ? file.name.slice(file.name.lastIndexOf(".")).toLowerCase() : "";
  if (!SUPPORTED_EXTENSIONS.includes(extension)) {
    showError($("upload-error"), UNSUPPORTED_FORMAT_MESSAGE);
    return;
  }
  const query = new URLSearchParams({ name_a: $("name-a").value, name_b: $("name-b").value });
  try {
    const job = await api(`/api/runs?${query}`, {
      method: "POST",
      headers: { "X-Audio-Filename": encodeURIComponent(file.name), "Content-Type": "application/octet-stream" },
      body: file,
    });
    showProgress(job.job_id);
  } catch (error) {
    showError($("upload-error"), error.message);
  }
}

function showProgress(jobId) {
  showView("progress-view");
  showError($("progress-error"), "");
  $("progress-back").hidden = true;
  $("progress-detail").textContent = "Uploading finished. Starting …";
  pollJob(jobId);
}

async function pollJob(jobId) {
  let job;
  try {
    job = await api(`/api/jobs/${jobId}`);
  } catch (error) {
    showError($("progress-error"), error.message);
    $("progress-back").hidden = false;
    return;
  }
  renderPhase(job);
  if (job.state === "running") {
    setTimeout(() => pollJob(jobId), POLL_INTERVAL_MS);
  } else if (job.state === "failed") {
    showError($("progress-error"), job.error);
    $("progress-back").hidden = false;
  } else {
    await openRun(job.run_id).catch(reportStartError);
  }
}

function renderPhase(job) {
  const activeIndex = PHASE_ORDER.indexOf(job.phase);
  for (const item of $("phase-list").children) {
    const index = PHASE_ORDER.indexOf(item.dataset.phase);
    item.className = index < activeIndex ? "phase-done" : index === activeIndex ? `phase-${job.state}` : "";
  }
  $("progress-detail").textContent = job.detail;
}

async function openRun(runId) {
  const loaded = await api(`/api/runs/${encodeURIComponent(runId)}/document`);
  $("audio").src = `/api/runs/${encodeURIComponent(runId)}/audio`;
  $("export-md").href = `/api/runs/${encodeURIComponent(runId)}/export.md`;
  $("export-txt").href = `/api/runs/${encodeURIComponent(runId)}/export.txt`;
  showView("editor-view");
  renderDocument(loaded);
}

function runPath(suffix) {
  return `/api/runs/${encodeURIComponent(currentDocument.run_id)}/${suffix}`;
}

function renderDocument(loaded) {
  currentDocument = loaded;
  clearSelection();
  showError($("editor-error"), "");
  const container = $("paragraphs");
  container.replaceChildren(...loaded.paragraphs.map(renderParagraph));
}

function renderParagraph(paragraph) {
  const block = document.createElement("article");
  block.className = "paragraph";
  block.dataset.firstWord = paragraph.first_word;
  if (paragraph.speaker_uncertainty.length) {
    block.classList.add("speaker-uncertain");
  }
  block.append(renderParagraphHeader(paragraph));
  if (paragraph.speaker_uncertainty.length) {
    const reasons = paragraph.speaker_uncertainty.map((reason) => REASON_LABELS[reason] || reason);
    block.append(Object.assign(document.createElement("p"), {
      className: "speaker-reasons",
      textContent: `Check speaker: ${reasons.join(", ")}`,
    }));
  }
  const text = document.createElement("p");
  text.className = "paragraph-text";
  paragraph.words.forEach((word, index) => {
    if (index > 0) {
      text.append(" ");
    }
    text.append(...renderWord(word, index));
  });
  block.append(text);
  return block;
}

function renderParagraphHeader(paragraph) {
  const header = document.createElement("header");
  header.append(Object.assign(document.createElement("strong"), { textContent: paragraph.speaker_name }));
  header.append(` · ${formatTime(paragraph.start)}–${formatTime(paragraph.end)} `);
  const edit = Object.assign(document.createElement("button"), { type: "button", textContent: "Edit" });
  edit.addEventListener("click", () => startEditing(paragraph, header.parentElement));
  header.append(edit);
  const select = speakerSelect();
  const assign = Object.assign(document.createElement("button"), { type: "button", textContent: "Assign paragraph" });
  assign.addEventListener("click", () => reassign(paragraph.first_word, paragraph.last_word, select.value));
  header.append(" ", select, assign);
  return header;
}

function speakerSelect() {
  const select = document.createElement("select");
  for (const [label, name] of Object.entries(currentDocument.speakers)) {
    select.append(new Option(`to ${name}`, label));
  }
  return select;
}

function renderWord(word, index) {
  const nodes = [];
  const showOriginals = $("show-originals").checked;
  if (word.removed_before) {
    const removed = Object.assign(document.createElement("span"), {
      className: "removed",
      title: `Removed by edit: ${word.removed_before}`,
      textContent: showOriginals ? `[−${word.removed_before}] ` : "⌫ ",
    });
    nodes.push(removed);
  }
  const span = Object.assign(document.createElement("span"), { textContent: word.text });
  span.dataset.wordIndex = index;
  span.dataset.sourceIds = word.source_word_ids.join(",");
  span.className = "word";
  if (word.change !== "unchanged") {
    span.classList.add("llm-changed");
    span.title = word.change === "inserted" ? "Inserted (not in speech recognition)" : `Original: ${word.original_text}`;
  }
  if (word.low_asr_confidence) {
    span.classList.add("low-asr-confidence");
    span.title = [span.title, "Speech recognition uncertain"].filter(Boolean).join(" · ");
  }
  span.addEventListener("click", () => seek(word.seek_time));
  nodes.push(span);
  if (showOriginals && word.change === "replaced") {
    nodes.push(Object.assign(document.createElement("span"), {
      className: "original",
      textContent: ` [${word.original_text}]`,
    }));
  }
  return nodes;
}

function seek(seconds) {
  if (!window.getSelection().isCollapsed) {
    return;
  }
  const audio = $("audio");
  audio.currentTime = seconds;
  audio.play().catch((error) => showError($("editor-error"), `Playback failed: ${error.message}`));
}

function startEditing(paragraph, block) {
  const textArea = Object.assign(document.createElement("textarea"), { value: paragraph.text, rows: 5 });
  const save = Object.assign(document.createElement("button"), { type: "button", textContent: "Save" });
  const cancel = Object.assign(document.createElement("button"), { type: "button", textContent: "Cancel" });
  const text = block.querySelector(".paragraph-text");
  const editor = document.createElement("div");
  editor.className = "paragraph-editor";
  editor.append(textArea, save, cancel);
  text.replaceWith(editor);
  cancel.addEventListener("click", () => renderDocument(currentDocument));
  save.addEventListener("click", async () => {
    try {
      const path = runPath(`paragraphs/${paragraph.first_word}/text`);
      renderDocument(await sendJson(path, "PUT", { text: textArea.value }));
    } catch (error) {
      showError($("editor-error"), error.message);
    }
  });
}

async function reassign(firstWord, lastWord, speaker) {
  try {
    const body = { first_word: firstWord, last_word: lastWord, speaker };
    renderDocument(await sendJson(runPath("reassign"), "POST", body));
  } catch (error) {
    showError($("editor-error"), error.message);
  }
}

function wordSpanOf(node) {
  const element = node.nodeType === Node.ELEMENT_NODE ? node : node.parentElement;
  return element ? element.closest(".word") : null;
}

function updateSelection() {
  const selection = window.getSelection();
  if (selection.isCollapsed || !selection.rangeCount) {
    return;
  }
  const range = selection.getRangeAt(0);
  const block = wordSpanOf(range.startContainer)?.closest(".paragraph");
  const spans = [...$("paragraphs").querySelectorAll(".word")].filter((span) => range.intersectsNode(span));
  const sourceIds = spans.flatMap((span) => span.dataset.sourceIds.split(",").filter(Boolean).map(Number));
  if (!spans.length) {
    return;
  }
  if (!block || spans.some((span) => span.closest(".paragraph") !== block)) {
    clearSelection();
    showError($("editor-error"), "Select words within one paragraph.");
    return;
  }
  if (!sourceIds.length) {
    clearSelection();
    showError($("editor-error"), "The selection contains only inserted words.");
    return;
  }
  currentSelection = { firstWord: Math.min(...sourceIds), lastWord: Math.max(...sourceIds) };
  showError($("editor-error"), "");
  $("selection-label").textContent = `${spans.length} selected word(s)`;
  $("selection-speaker").replaceChildren(...speakerSelect().children);
  $("selection-bar").hidden = false;
}

function clearSelection() {
  currentSelection = null;
  $("selection-bar").hidden = true;
}

async function swapNames() {
  const labels = Object.keys(currentDocument.speakers);
  if (labels.length !== 2) {
    showError($("editor-error"), "Swapping needs exactly two speakers.");
    return;
  }
  const [first, second] = labels;
  const swapped = { [first]: currentDocument.speakers[second], [second]: currentDocument.speakers[first] };
  try {
    renderDocument(await sendJson(runPath("speakers"), "PUT", swapped));
  } catch (error) {
    showError($("editor-error"), error.message);
  }
}

$("upload-form").addEventListener("submit", startUpload);
$("progress-back").addEventListener("click", () => showStart());
$("editor-back").addEventListener("click", () => showStart());
$("swap-names").addEventListener("click", swapNames);
$("show-originals").addEventListener("change", () => renderDocument(currentDocument));
$("paragraphs").addEventListener("mouseup", updateSelection);
$("selection-assign").addEventListener("click", () => {
  reassign(currentSelection.firstWord, currentSelection.lastWord, $("selection-speaker").value);
});
showStart().catch(reportStartError);
