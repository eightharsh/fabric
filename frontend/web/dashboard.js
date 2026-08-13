/* Operator dashboard: drives the 3-monitor view from the real /predict output. */
"use strict";
const $ = (id) => document.getElementById(id);
let selectedFile = null;

/* ── endpoint (shared with the console via localStorage) ─────────────────── */
(function initEndpoint() {
  const saved = localStorage.getItem("fd_endpoint");
  if (saved) $("endpoint").value = saved;
  else if (location.hostname) $("endpoint").value = `${location.protocol}//${location.hostname}:8000/predict`;
})();
$("endpoint").addEventListener("change", () => {
  localStorage.setItem("fd_endpoint", $("endpoint").value.trim());
  checkHealth();
});
$("ppm").addEventListener("change", () => localStorage.setItem("fd_ppm", $("ppm").value));

/* ── health → status pill + category list ────────────────────────────────── */
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
    if (!$("ppm").dataset.filled) {
      $("ppm").value = localStorage.getItem("fd_ppm") || d.pixels_per_mm || 5;
      $("ppm").dataset.filled = "1";
    }
  } catch {
    $("status").className = "pill down";
    $("statusText").textContent = "backend offline";
  }
}
$("category").addEventListener("change", () => localStorage.setItem("fd_category", $("category").value));
checkHealth();

/* ── frame input: upload / camera / drop / paste ─────────────────────────── */
function setFile(f) {
  if (!f || !f.type.startsWith("image/")) return;
  selectedFile = f;
  stopCamera();
  $("rawImg").src = URL.createObjectURL(f);
  $("rawEmpty").style.display = "none";
  analyze();
}
$("upload").addEventListener("change", (e) => setFile(e.target.files[0]));
$("camera").addEventListener("change", (e) => setFile(e.target.files[0]));
const raw = $("raw");
["dragenter", "dragover"].forEach((ev) => raw.addEventListener(ev, (e) => { e.preventDefault(); raw.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => raw.addEventListener(ev, (e) => { e.preventDefault(); raw.classList.remove("drag"); }));
raw.addEventListener("drop", (e) => { const f = e.dataTransfer?.files[0]; if (f) setFile(f); });
window.addEventListener("paste", (e) => {
  const it = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (it) setFile(it.getAsFile());
});

/* ── live camera ─────────────────────────────────────────────────────────── */
let stream = null;
async function startCamera() {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    setStatus("bad", "LIVE NEEDS HTTPS"); return;
  }
  try { stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { ideal: "environment" } } }); }
  catch (e) { setStatus("bad", "CAMERA ERROR"); return; }
  const v = $("video");
  v.srcObject = stream; v.style.display = "block";
  $("rawImg").style.display = "none"; $("rawEmpty").style.display = "none";
  const tick = () => {
    if (!stream) return;
    const c = $("canvas");
    if (v.videoWidth) {
      c.width = v.videoWidth; c.height = v.videoHeight;
      c.getContext("2d").drawImage(v, 0, 0);
      c.toBlob((b) => { if (b) analyze(new File([b], "f.png", { type: "image/png" })); }, "image/png");
    }
    setTimeout(tick, 1200);   // ~1 fps: DINOv3 inference is the bottleneck
  };
  tick();
}
function stopCamera() {
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  $("video").style.display = "none"; $("rawImg").style.display = "";
  $("liveBtn").textContent = "Live";
  $("liveBtn").classList.remove("pri");
}
$("liveBtn").addEventListener("click", () => {
  if (stream) { stopCamera(); return; }        // toggle off
  startCamera();
  $("liveBtn").textContent = "Stop";
  $("liveBtn").classList.add("pri");
});

/* ── analyze + render the 3 monitors ─────────────────────────────────────── */
function setStatus(cls, text) {
  $("statusBig").className = "status " + cls;
  $("statusBigText").textContent = text;
}
async function analyze(fileArg) {
  const file = fileArg || selectedFile;
  if (!file) return;
  setStatus("idle", "INSPECTING…");
  try {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("category", $("category").value);
    fd.append("pixels_per_mm", $("ppm").value);
    const res = await fetch($("endpoint").value.trim(), { method: "POST", body: fd });
    if (!res.ok) throw new Error("HTTP " + res.status);
    render(await res.json());
  } catch (e) { setStatus("bad", "REQUEST FAILED"); }
}

function metric(k, v) { return `<div class="metric"><div class="k">${k}</div><div class="v">${v}</div></div>`; }

function render(d) {
  const defective = d.is_defective === null ? d.num_defects > 0 : d.is_defective;
  if (d.original_png) { $("decImg").src = d.original_png; $("decEmpty").style.display = "none"; }
  if (d.heatmap_png) { $("heatImg").src = d.heatmap_png; $("heatEmpty").style.display = "none"; }

  // decision-panel boxes (scaled from image_size), with type labels
  const layer = $("boxLayer"); layer.innerHTML = "";
  const size = d.image_size || 224;
  (d.boxes || []).forEach((b) => {
    const el = document.createElement("div");
    el.className = "bbox";
    el.style.left = (b.x / size) * 100 + "%"; el.style.top = (b.y / size) * 100 + "%";
    el.style.width = (b.w / size) * 100 + "%"; el.style.height = (b.h / size) * 100 + "%";
    el.innerHTML = `<span class="lbl">${b.type ? b.type : "defect"}</span>`;
    layer.appendChild(el);
  });

  setStatus(defective ? "bad" : "ok", defective ? "DEFECT DETECTED" : "PASS");
  const top = (d.boxes || []).slice().sort((a, b) => b.area - a.area)[0];
  $("metrics").innerHTML =
    metric("Type", top?.type || "—") +
    metric("Score", d.anomaly_score) +
    metric("Threshold", d.threshold ?? "—") +
    metric("Defects", d.num_defects) +
    metric("Penalty pts", d.defect_points ?? "—") +
    metric("Category", d.category || "—") +
    metric("Model", d.model || "—") +
    metric("Latency", (d.latency_ms ?? "—") + " ms");

  if (d.boxes?.length) {
    let t = "<table><tr><th>#</th><th>type</th><th>size (mm)</th><th>points</th><th>score</th></tr>";
    d.boxes.forEach((b, i) => {
      const conf = b.type_conf != null ? ` <span style="color:var(--mut)">${Math.round(b.type_conf * 100)}%</span>` : "";
      t += `<tr><td>${i + 1}</td><td>${b.type ? b.type + conf : "—"}</td>` +
        `<td>${b.size_mm ?? "—"}</td><td class="p${b.points || 0}">${b.points ?? "—"}</td>` +
        `<td>${Number(b.score).toFixed(2)}</td></tr>`;
    });
    $("defects").innerHTML = t + "</table>";
  } else $("defects").innerHTML = '<div class="hint">No defects flagged in this frame.</div>';
}
