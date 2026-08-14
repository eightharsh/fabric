/* Operator dashboard — BATCH mode: inspect many frames at once, grade together. */
"use strict";
const $ = (id) => document.getElementById(id);
let frames = [];          // { file, el, img, boxLayer, badge, meta, result }
let done = 0, total = 0;
let view = "boxes";       // "boxes" (original + boxes) | "heatmap"

/* ── endpoint / health / controls (shared with the console) ──────────────── */
(function initEndpoint() {
  const saved = localStorage.getItem("fd_endpoint");
  if (saved) $("endpoint").value = saved;
  else if (location.hostname) $("endpoint").value = `${location.protocol}//${location.hostname}:8000/predict`;
})();
$("endpoint").addEventListener("change", () => { localStorage.setItem("fd_endpoint", $("endpoint").value.trim()); checkHealth(); });
$("ppm").addEventListener("change", () => localStorage.setItem("fd_ppm", $("ppm").value));
$("category").addEventListener("change", () => localStorage.setItem("fd_category", $("category").value));

function populateCategories(list, dflt) {
  const sel = $("category");
  if (sel.dataset.filled || !list?.length) return;
  const saved = localStorage.getItem("fd_category");
  sel.innerHTML = list.map((c) => `<option>${c}</option>`).join("");
  sel.value = saved && list.includes(saved) ? saved : (dflt || list[0]);
  sel.dataset.filled = "1";
}
async function checkHealth() {
  const base = $("endpoint").value.trim().replace(/\/predict\/?$/, "");
  try {
    const res = await fetch(base + "/health");
    if (!res.ok) throw new Error();
    const d = await res.json();
    $("status").className = "pill ok";
    $("statusText").textContent = `online · ${d.category}`;
    populateCategories(d.available, d.category);
    if (!$("ppm").dataset.filled) { $("ppm").value = localStorage.getItem("fd_ppm") || d.pixels_per_mm || 5; $("ppm").dataset.filled = "1"; }
  } catch { $("status").className = "pill down"; $("statusText").textContent = "backend offline"; }
}
checkHealth();

/* ── input: multi-upload + drop many ─────────────────────────────────────── */
$("upload").addEventListener("change", (e) => addFiles(e.target.files));
$("clearBtn").addEventListener("click", clearAll);
$("viewSeg").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  view = b.dataset.v;
  document.querySelectorAll("#viewSeg button").forEach((x) => x.classList.toggle("on", x === b));
  frames.forEach(applyView);
});
const grid = $("grid");
["dragenter", "dragover"].forEach((ev) => grid.addEventListener(ev, (e) => { e.preventDefault(); grid.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => grid.addEventListener(ev, (e) => { e.preventDefault(); grid.classList.remove("drag"); }));
grid.addEventListener("drop", (e) => addFiles(e.dataTransfer?.files));
window.addEventListener("paste", (e) => {
  const imgs = [...(e.clipboardData?.items || [])].filter((i) => i.type.startsWith("image/")).map((i) => i.getAsFile());
  if (imgs.length) addFiles(imgs);
});

function clearAll() {
  frames = []; done = 0; total = 0;
  $("grid").innerHTML = '<div class="empty" id="empty">Drop or upload several fabric images — they’re inspected in parallel and graded together.</div>';
  $("metrics").innerHTML = ""; $("progWrap").style.display = "none";
  setVerdict("idle", "READY");
}

/* ── one card per frame ──────────────────────────────────────────────────── */
function makeCard(file) {
  $("empty")?.remove();
  const el = document.createElement("div");
  el.className = "frame";
  el.innerHTML =
    '<div class="thumb"><img alt=""><div class="boxLayer"></div><div class="spin"><div class="sp"></div></div></div>' +
    `<div class="meta"><div class="fn"></div><div class="ln"><span class="ty">inspecting…</span><span class="pt"></span></div></div>`;
  const img = el.querySelector("img");
  img.src = URL.createObjectURL(file);
  el.querySelector(".fn").textContent = file.name || "frame";
  $("grid").appendChild(el);
  return { file, el, img, boxLayer: el.querySelector(".boxLayer"), spin: el.querySelector(".spin"),
           ty: el.querySelector(".ty"), pt: el.querySelector(".pt"), result: null };
}

function applyView(fr) {
  const d = fr.result;
  if (!d) return;
  fr.img.src = view === "heatmap" ? (d.heatmap_png || d.original_png) : (d.original_png || d.heatmap_png);
  fr.boxLayer.style.display = view === "heatmap" ? "none" : "";  // boxes belong to the decision view
}

function fillCard(fr, d) {
  fr.result = d;
  fr.spin.style.display = "none";
  const defective = d.is_defective === null ? d.num_defects > 0 : d.is_defective;
  // build the box overlay once; applyView chooses image + box visibility
  const size = d.image_size || 224;
  fr.boxLayer.innerHTML = "";
  (d.boxes || []).forEach((b) => {
    const bx = document.createElement("div");
    bx.className = "bbox";
    bx.style.left = (b.x / size) * 100 + "%"; bx.style.top = (b.y / size) * 100 + "%";
    bx.style.width = (b.w / size) * 100 + "%"; bx.style.height = (b.h / size) * 100 + "%";
    if (b.type) bx.innerHTML = `<span class="lbl">${b.type}</span>`;
    fr.boxLayer.appendChild(bx);
  });
  applyView(fr);
  const badge = document.createElement("span");
  badge.className = "badge " + (defective ? "bad" : "ok");
  badge.textContent = defective ? "DEFECT" : "PASS";
  fr.el.querySelector(".thumb").appendChild(badge);
  const top = (d.boxes || []).slice().sort((a, b) => b.area - a.area)[0];
  fr.ty.className = "ty " + (defective ? "bad" : "ok");
  fr.ty.textContent = defective ? (top?.type || "defect") : "pass";
  fr.pt.textContent = defective ? `${d.defect_points ?? 0} pt · ${d.anomaly_score}` : `${d.anomaly_score}`;
}
function failCard(fr, msg) {
  fr.spin.style.display = "none";
  fr.ty.className = "ty bad"; fr.ty.textContent = "error"; fr.pt.textContent = msg.slice(0, 18);
}

/* ── batch run (small concurrency pool) + aggregate ──────────────────────── */
function addFiles(list) {
  const files = [...(list || [])].filter((f) => f.type.startsWith("image/"));
  if (!files.length) return;
  const newFrames = files.map(makeCard);
  frames.push(...newFrames);
  total += newFrames.length;
  $("progWrap").style.display = "block";
  runPool(newFrames, analyzeOne, 3);
}

async function analyzeOne(fr) {
  try {
    const fd = new FormData();
    fd.append("file", fr.file);
    fd.append("category", $("category").value);
    fd.append("pixels_per_mm", $("ppm").value);
    const res = await fetch($("endpoint").value.trim(), { method: "POST", body: fd });
    if (!res.ok) throw new Error("HTTP " + res.status);
    fillCard(fr, await res.json());
  } catch (e) { failCard(fr, e.message); }
  done++;
  updateSummary();
}

async function runPool(items, worker, concurrency) {
  let i = 0;
  const run = async () => { while (i < items.length) await worker(items[i++]); };
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, run));
}

function setVerdict(cls, text) { $("verdict").className = "verdict " + cls; $("verdictText").textContent = text; }
function metric(k, v) { return `<div class="metric"><div class="k">${k}</div><div class="v">${v}</div></div>`; }

function batchGrade(totalPoints, n, imageSize, ppm) {
  const frameMm = imageSize / ppm;                    // one square frame's side in mm
  const lengthYd = (n * frameMm) / 914.4;             // frames stacked along the roll
  const widthIn = frameMm / 25.4;
  const p100 = lengthYd > 0 && widthIn > 0 ? (totalPoints * 3600) / (lengthYd * widthIn) : 0;
  return { p100: Math.round(p100), grade: p100 <= 40 ? "FIRST" : "SECOND" };
}

function updateSummary() {
  const results = frames.map((f) => f.result).filter(Boolean);
  const defective = results.filter((d) => (d.is_defective === null ? d.num_defects > 0 : d.is_defective)).length;
  const points = results.reduce((s, d) => s + (d.defect_points || 0), 0);
  const size = results[0]?.image_size || 224;
  const ppm = parseFloat($("ppm").value) || 5;
  const g = batchGrade(points, frames.length, size, ppm);

  const pct = total ? Math.round((done / total) * 100) : 0;
  $("prog").style.width = pct + "%";
  if (done >= total) setTimeout(() => { $("progWrap").style.display = "none"; }, 600);

  if (defective > 0) setVerdict("bad", `${defective} DEFECTIVE`);
  else if (done > 0) setVerdict("ok", "ALL PASS");
  else setVerdict("idle", "INSPECTING…");

  $("metrics").innerHTML =
    metric("Frames", `${done}/${total}`) +
    metric("Defective", defective) +
    metric("Pass", Math.max(0, done - defective)) +
    metric("Penalty pts", points) +
    metric("Points / 100yd²", g.p100) +
    metric("Batch grade", g.grade);
}
