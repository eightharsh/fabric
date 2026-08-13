/* Fabric Inspect — console logic.
   Talks to the FastAPI /predict backend and renders verdict, stats, and the
   Original / Heatmap / Boxes result views. Boxes are drawn client-side from the
   returned coordinates so they stay crisp at any display size. */
"use strict";

const $ = (id) => document.getElementById(id);
let selectedFile = null;
let lastResult = null; // most recent /predict response, for the view toggle

/* ── Endpoint: remember the user's choice; else default to this page's host ── */
(function initEndpoint() {
  const saved = localStorage.getItem("fd_endpoint");
  if (saved) { $("endpoint").value = saved; }
  else if (location.hostname) {
    $("endpoint").value = `${location.protocol}//${location.hostname}:8000/predict`;
  }
})();
$("endpoint").addEventListener("change", () => {
  localStorage.setItem("fd_endpoint", $("endpoint").value.trim());
  checkHealth();
});

/* ── Category picker ─────────────────────────────────────────────────────── */
function populateCategories(list, dflt) {
  const sel = $("category");
  if (sel.dataset.filled || !list?.length) return;
  const saved = localStorage.getItem("fd_category");
  sel.innerHTML = list.map((c) => `<option value="${c}">${c}</option>`).join("");
  sel.value = saved && list.includes(saved) ? saved : (dflt || list[0]);
  sel.dataset.filled = "1";
}
function pillCategory() {
  if ($("status").classList.contains("ok")) {
    $("statusText").textContent = `online · ${$("category").value}`;
  }
}
$("category").addEventListener("change", () => {
  localStorage.setItem("fd_category", $("category").value);
  pillCategory();
});
$("ppm").addEventListener("change", () => localStorage.setItem("fd_ppm", $("ppm").value));

/* ── Backend health → status pill + category list ────────────────────────── */
async function checkHealth() {
  const base = $("endpoint").value.trim().replace(/\/predict\/?$/, "");
  try {
    const res = await fetch(base + "/health", { method: "GET" });
    if (!res.ok) throw new Error();
    const d = await res.json();
    $("status").className = "pill ok";
    populateCategories(d.available, d.category);
    if (!$("ppm").dataset.filled) {
      $("ppm").value = localStorage.getItem("fd_ppm") || d.pixels_per_mm || 5;
      $("ppm").dataset.filled = "1";
    }
    pillCategory();
  } catch {
    $("status").className = "pill down";
    $("statusText").textContent = "backend offline";
  }
}
checkHealth();

/* ── Result-pane state machine ───────────────────────────────────────────── */
function showState(name) {
  for (const s of ["Empty", "Loading", "Error", "Result"]) {
    $("state" + s).classList.toggle("hide", s.toLowerCase() !== name);
  }
}

/* ── File selection (upload / camera / drop / paste) ─────────────────────── */
function setFile(f) {
  if (!f || !f.type.startsWith("image/")) return;
  selectedFile = f;
  $("preview").src = URL.createObjectURL(f);
  $("dzEmpty").classList.add("hide");
  $("dzLive").classList.add("hide");
  $("dzPreview").classList.remove("hide");
  stopCamera();
}
function clearFile() {
  selectedFile = null;
  $("dzPreview").classList.add("hide");
  $("dzEmpty").classList.remove("hide");
}
$("upload").addEventListener("change", (e) => setFile(e.target.files[0]));
$("camera").addEventListener("change", (e) => setFile(e.target.files[0]));
$("clearPreview").addEventListener("click", clearFile);

/* drag-and-drop + clipboard paste onto the dropzone */
const dz = $("dropzone");
["dragenter", "dragover"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) =>
  dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
dz.addEventListener("drop", (e) => { const f = e.dataTransfer?.files[0]; if (f) setFile(f); });
window.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) setFile(item.getAsFile());
});

/* ── Analyze flow ────────────────────────────────────────────────────────── */
async function runAnalyze(file, btn) {
  if (!file) return;
  const prev = btn ? btn.innerHTML : "";
  if (btn) { btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Analyzing…'; }
  showState("loading");
  try {
    const fd = new FormData();
    fd.append("file", file);
    if ($("category").value) fd.append("category", $("category").value);
    if ($("ppm").value) fd.append("pixels_per_mm", $("ppm").value);
    const res = await fetch($("endpoint").value.trim(), { method: "POST", body: fd });
    if (!res.ok) {
      let detail = "HTTP " + res.status;
      try { const j = await res.json(); if (j.detail) detail = j.detail; } catch { /* ignore */ }
      throw new Error(detail);
    }
    renderResult(await res.json());
  } catch (err) {
    renderError(err.message);
  } finally {
    if (btn) { btn.disabled = false; btn.innerHTML = prev; }
  }
}
$("analyze").addEventListener("click", () => runAnalyze(selectedFile, $("analyze")));

/* ── Live camera (getUserMedia; needs localhost or https) ────────────────── */
let stream = null;
async function startCamera() {
  // getUserMedia only exists in a secure context (HTTPS or localhost). Over plain
  // http on a phone/LAN the browser hides it entirely — tell the user precisely.
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    $("liveHint").innerHTML =
      "Live camera needs <b>HTTPS</b> (or localhost). Over http on a phone/LAN the browser " +
      "blocks the live feed — use the <b>Camera</b> snapshot or <b>Upload</b> instead, or open this app over https.";
    return;
  }
  // Prefer the rear camera, but fall back to any camera so desktops don't fail.
  const attempts = [
    { video: { facingMode: { ideal: "environment" } }, audio: false },
    { video: true, audio: false },
  ];
  let lastErr = null;
  for (const constraints of attempts) {
    try { stream = await navigator.mediaDevices.getUserMedia(constraints); lastErr = null; break; }
    catch (e) { lastErr = e; }
  }
  if (!stream) {
    $("liveHint").textContent = "Could not open camera: " + (lastErr?.message || lastErr);
    return;
  }
  $("video").srcObject = stream;
  $("dzEmpty").classList.add("hide");
  $("dzPreview").classList.add("hide");
  $("dzLive").classList.remove("hide");
  $("liveHint").textContent = "Point at the fabric, then Capture & analyze.";
}
function stopCamera() {
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  $("dzLive").classList.add("hide");
  if (!selectedFile) $("dzEmpty").classList.remove("hide");
  $("liveHint").textContent = "";
}
$("liveBtn").addEventListener("click", startCamera);
$("stopCam").addEventListener("click", stopCamera);
$("capture").addEventListener("click", () => {
  const v = $("video");
  if (!v.videoWidth) return;
  const c = $("canvas");
  c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
  c.toBlob((blob) => {
    if (blob) runAnalyze(new File([blob], "capture.png", { type: "image/png" }), $("capture"));
  }, "image/png");
});

/* ── Rendering ───────────────────────────────────────────────────────────── */
const OK_ICON = '<svg viewBox="0 0 24 24" fill="none"><path d="M20 6L9 17l-5-5" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const BAD_ICON = '<svg viewBox="0 0 24 24" fill="none"><path d="M12 8v5M12 16.5v.5" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle cx="12" cy="12" r="9" stroke="currentColor" stroke-width="1.6"/></svg>';

function renderError(msg) {
  $("errorMsg").textContent = `${msg} — is the backend running and reachable at the endpoint on the left?`;
  showState("error");
}

function renderResult(d) {
  lastResult = d;
  const defective = d.is_defective === null ? d.num_defects > 0 : d.is_defective;

  $("verdict").className = "verdict " + (defective ? "bad" : "ok");
  $("verdict").innerHTML = (defective ? BAD_ICON : OK_ICON) +
    (defective ? `Defective · ${d.num_defects} region${d.num_defects === 1 ? "" : "s"}` : "Pass · no defect");

  const parts = [];
  if (d.category) parts.push(d.category);
  if (d.model) parts.push(d.model);
  if (d.latency_ms != null) parts.push(`${d.latency_ms} ms`);
  $("latencyTag").textContent = parts.join(" · ");

  const tile = (k, v, sub) =>
    `<div class="stat"><div class="k">${k}</div><div class="v">${v}${sub ? `<small> ${sub}</small>` : ""}</div></div>`;
  $("stats").innerHTML =
    tile("Anomaly score", d.anomaly_score) +
    tile("Threshold", d.threshold != null ? d.threshold : "—") +
    tile("Regions", d.num_defects) +
    tile("Penalty pts", d.defect_points != null ? d.defect_points : "—");

  const hasLayers = !!(d.original_png && d.heatmap_png);
  $("viewTabs").classList.toggle("hide", !hasLayers);
  setView(hasLayers ? "boxes" : "combined");

  if (d.boxes && d.boxes.length) {
    const typed = d.boxes.some((b) => b.type);
    let t = "<div class='boxes-title'>Defects (ASTM D5430 4-Point)</div><table class='data'>" +
      `<tr><th>#</th>${typed ? "<th>type</th>" : ""}<th>size (mm)</th><th>points</th><th>score</th></tr>`;
    d.boxes.forEach((b, i) => {
      const type = b.type
        ? `${b.type} <span class="muted">${Math.round((b.type_conf || 0) * 100)}%</span>`
        : "—";
      t += `<tr><td>${i + 1}</td>${typed ? `<td>${type}</td>` : ""}` +
        `<td>${b.size_mm ?? "—"}</td><td>${b.points ?? "—"}</td>` +
        `<td>${Number(b.score).toFixed(2)}</td></tr>`;
    });
    $("boxWrap").innerHTML = t + "</table>";
  } else $("boxWrap").innerHTML = "";

  showState("result");
}

/* view toggle: Original / Heatmap / Boxes */
function drawBoxLayer(show) {
  const layer = $("boxLayer");
  layer.innerHTML = "";
  const d = lastResult;
  if (!show || !d?.boxes?.length) return;
  const size = d.image_size || 224; // boxes are in the resized inference frame
  d.boxes.forEach((b) => {
    const el = document.createElement("div");
    el.className = "bbox";
    el.style.left = (b.x / size) * 100 + "%";
    el.style.top = (b.y / size) * 100 + "%";
    el.style.width = (b.w / size) * 100 + "%";
    el.style.height = (b.h / size) * 100 + "%";
    el.innerHTML = `<span class="lbl">${Number(b.score).toFixed(2)}</span>`;
    layer.appendChild(el);
  });
}
function setView(view) {
  const d = lastResult;
  if (!d) return;
  const src = view === "heatmap" ? d.heatmap_png
    : view === "combined" ? d.overlay_png
      : d.original_png || d.overlay_png;
  if (src) { $("overlay").src = src; $("overlay").style.display = "block"; }
  else $("overlay").style.display = "none";
  drawBoxLayer(view === "boxes");
  document.querySelectorAll("#viewTabs button").forEach((t) => {
    const on = t.dataset.view === view;
    t.classList.toggle("active", on);
    t.setAttribute("aria-selected", on ? "true" : "false");
  });
}
document.querySelectorAll("#viewTabs button").forEach((t) =>
  t.addEventListener("click", () => setView(t.dataset.view)));
