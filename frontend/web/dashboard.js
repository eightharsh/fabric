/* Operator dashboard — BATCH mode: inspect many frames at once, grade together. */
"use strict";
const $ = (id) => document.getElementById(id);
let frames = [];          // { file, el, img, boxLayer, badge, meta, result }
let done = 0, total = 0;
let view = "boxes";       // "boxes" (original + boxes) | "heatmap"

// A defect type is shown as "uncertain" when the backend abstained (type
// "unknown") or the type-probe confidence is low. Keeps a shaky guess from
// reading as authoritative on the operator's grade sheet.
const LOWCONF = 0.6;
const isUncertain = (b) => b && (b.type === "unknown" || (b.type_conf || 0) < LOWCONF);
const typeText = (b) => (!b || !b.type ? "defect" : b.type === "unknown" ? "uncertain" : b.type);

/* ── endpoint / health / controls (shared with the console) ──────────────── */
(function initEndpoint() {
  const saved = localStorage.getItem("fd_endpoint");
  if (saved) $("endpoint").value = saved;
  else if (location.hostname) $("endpoint").value = `${location.protocol}//${location.hostname}:8000/predict`;
})();
$("endpoint").addEventListener("change", () => { localStorage.setItem("fd_endpoint", $("endpoint").value.trim()); checkHealth(); });
$("ppm").addEventListener("change", () => localStorage.setItem("fd_ppm", $("ppm").value));
function pillOnline() {
  if ($("status").classList.contains("ok")) $("statusText").textContent = `online · ${$("category").value}`;
}
$("category").addEventListener("change", () => {
  localStorage.setItem("fd_category", $("category").value);
  pillOnline();                       // reflect the selected category in the status pill
});

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
    populateCategories(d.available, d.category);
    pillOnline();                       // show the currently-selected category
    if (!$("ppm").dataset.filled) { $("ppm").value = localStorage.getItem("fd_ppm") || d.pixels_per_mm || 5; $("ppm").dataset.filled = "1"; }
  } catch { $("status").className = "pill down"; $("statusText").textContent = "backend offline"; }
}
checkHealth();

/* ── input: multi-upload + drop many ─────────────────────────────────────── */
$("upload").addEventListener("change", (e) => addFiles(e.target.files));
$("clearBtn").addEventListener("click", clearAll);
$("reportBtn").addEventListener("click", downloadReport);
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
    if (b.type) {
      const unc = isUncertain(b);
      bx.innerHTML = `<span class="lbl${unc ? " lbl-unc" : ""}">${typeText(b)}</span>`;
    }
    fr.boxLayer.appendChild(bx);
  });
  applyView(fr);
  const badge = document.createElement("span");
  badge.className = "badge " + (defective ? "bad" : "ok");
  badge.textContent = defective ? "DEFECT" : "PASS";
  fr.el.querySelector(".thumb").appendChild(badge);
  const top = (d.boxes || []).slice().sort((a, b) => b.area - a.area)[0];
  fr.ty.className = "ty " + (defective ? "bad" : "ok");
  fr.ty.textContent = defective ? typeText(top) : "pass";
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

/* ── Downloadable inspection report — formal QA certificate, print-ready ─── */
function isBad(d) { return d.is_defective === null ? d.num_defects > 0 : d.is_defective; }
function severity(p) { return p >= 3 ? "Major" : "Minor"; }
function esc(s) { return String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }

function downloadReport() {
  const results = frames.map((f) => ({ name: f.file.name, d: f.result })).filter((x) => x.d);
  if (!results.length) return;

  const defective = results.filter(({ d }) => isBad(d)).length;
  const passN = results.length - defective;
  const points = results.reduce((s, { d }) => s + (d.defect_points || 0), 0);
  const size = results[0].d.image_size || 224;
  const ppm = parseFloat($("ppm").value) || 5;
  const g = batchGrade(points, results.length, size, ppm);
  const cat = $("category").value;
  const TOL = 40;
  const accepted = g.grade === "FIRST";

  // 4-point calculation components (line-scan proxy: frames stacked along roll)
  const frameMm = size / ppm;
  const lengthYd = (results.length * frameMm) / 914.4;
  const widthIn = frameMm / 25.4;
  const areaSqYd = lengthYd * (widthIn / 36);

  const dt = new Date();
  const pad = (x) => String(x).padStart(2, "0");
  const reportNo = `FI-${dt.getFullYear()}${pad(dt.getMonth() + 1)}${pad(dt.getDate())}-${pad(dt.getHours())}${pad(dt.getMinutes())}`;
  const when = dt.toLocaleString();

  // flatten every detected defect into a log line
  const log = [];
  results.forEach(({ name, d }) => {
    if (!isBad(d)) return;
    const boxes = d.boxes || [];
    if (!boxes.length) {
      log.push({ frame: name, type: "unlocalized", size: "—", pts: d.defect_points || 0, note: true });
    } else {
      boxes.forEach((b) => log.push({ frame: name, type: typeText(b), size: b.size_mm ?? "—", pts: b.points ?? 0 }));
    }
  });
  const byType = {};
  log.forEach((x) => { byType[x.type] = byType[x.type] || { n: 0, p: 0 }; byType[x.type].n++; byType[x.type].p += Number(x.pts) || 0; });

  const field = (v) => (v || v === 0 ? `<b>${esc(v)}</b>` : `<span class="fill"></span>`);
  const paramRows = `
    <tr><td>Material / category</td><td>${field(cat)}</td><td>Report no.</td><td><b>${reportNo}</b></td></tr>
    <tr><td>Inspection standard</td><td><b>ASTM D5430</b></td><td>Date / time</td><td><b>${esc(when)}</b></td></tr>
    <tr><td>Grading system</td><td><b>4-Point</b></td><td>Roll / lot no.</td><td>${field()}</td></tr>
    <tr><td>Detection model</td><td><b>DINOv3 ViT-L/16 + PatchCore</b></td><td>Line / machine</td><td>${field()}</td></tr>
    <tr><td>Calibration</td><td><b>${ppm} px/mm</b></td><td>Inspector</td><td>${field()}</td></tr>`;

  const calcRows = `
    <tr><td>Frames inspected</td><td class="n">${results.length}</td></tr>
    <tr><td>Effective length (proxy)</td><td class="n">${lengthYd.toFixed(2)} yd</td></tr>
    <tr><td>Effective width (proxy)</td><td class="n">${widthIn.toFixed(2)} in</td></tr>
    <tr><td>Inspected area</td><td class="n">${areaSqYd.toFixed(3)} yd²</td></tr>
    <tr><td>Total penalty points</td><td class="n">${points}</td></tr>
    <tr class="hl"><td>Points per 100 yd²</td><td class="n">${g.p100}</td></tr>
    <tr><td>Acceptance tolerance</td><td class="n">${TOL} / 100 yd²</td></tr>`;

  const typeRows = Object.entries(byType).sort((a, b) => b[1].p - a[1].p)
    .map(([t, v]) => `<tr><td>${esc(t)}</td><td class="n">${v.n}</td><td class="n">${v.p}</td></tr>`).join("")
    || `<tr><td colspan="3" class="muted">No defects detected.</td></tr>`;

  const logRows = log.length
    ? log.map((x, i) => `<tr><td class="n">${i + 1}</td><td>${esc(x.frame)}</td><td>${esc(x.type)}${x.note ? ' <span class="tag">not localized</span>' : ""}</td><td class="n">${x.size === "—" ? "—" : esc(x.size)}</td><td>${severity(x.pts)}</td><td class="n">${x.pts}</td></tr>`).join("")
    : `<tr><td colspan="6" class="muted">No defects logged — all frames within acceptance.</td></tr>`;

  const evidence = results.filter(({ d }) => isBad(d)).map(({ name, d }) => {
    const top = (d.boxes || []).slice().sort((a, b) => b.area - a.area)[0];
    const thumb = d.overlay_png || d.heatmap_png || d.original_png || "";
    return `<figure><div class="ph">${thumb ? `<img src="${thumb}">` : ""}</div><figcaption>${esc(name)}<span>${esc(typeText(top))} · ${d.defect_points ?? 0} pt</span></figcaption></figure>`;
  }).join("");

  const logo = '<svg viewBox="0 0 24 24" fill="none"><rect x="2.5" y="2.5" width="19" height="19" rx="4" stroke="currentColor" stroke-width="1.6"/><rect x="8" y="8" width="8" height="8" rx="1.5" fill="currentColor"/></svg>';
  const html = `<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Inspection report ${reportNo}</title><style>
 :root{--ink:#161a1d;--mut:#5b6570;--line:#c4ccd3;--line2:#e3e8ec;--head:#eef1f4;--nav:#1e3a5f;--ok:#15803d;--okb:#e7f6ec;--bad:#b91c1c;--badb:#fbeaea}
 *{box-sizing:border-box}
 html,body{margin:0;background:#e9edf0;color:var(--ink);font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;font-size:12.5px;line-height:1.5}
 .doc{max-width:820px;margin:22px auto;background:#fff;padding:34px 40px 30px;box-shadow:0 4px 24px rgba(0,0,0,.10);border-top:4px solid var(--nav)}
 .hd{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;border-bottom:2px solid var(--ink);padding-bottom:14px}
 .brand{display:flex;align-items:center;gap:9px;font-weight:700;font-size:15px}.brand svg{width:22px;height:22px;color:var(--nav)}
 .brand small{display:block;font-weight:500;font-size:10px;color:var(--mut);letter-spacing:.02em}
 .hd .t{text-align:right}
 .hd h1{margin:0;font-size:19px;letter-spacing:.02em;text-transform:uppercase}
 .hd .sub{color:var(--mut);font-size:11px;margin-top:3px;letter-spacing:.14em}
 .disp{display:flex;align-items:center;gap:16px;border:1.5px solid;border-radius:6px;padding:14px 20px;margin:18px 0 6px}
 .disp.ok{border-color:var(--ok);background:var(--okb)}.disp.bad{border-color:var(--bad);background:var(--badb)}
 .disp .stamp{font-size:22px;font-weight:800;letter-spacing:.06em}.disp.ok .stamp{color:var(--ok)}.disp.bad .stamp{color:var(--bad)}
 .disp .d{font-size:12px;color:var(--mut)}.disp .d b{color:var(--ink)}
 .disp .r{margin-left:auto;text-align:right;font-family:ui-monospace,monospace;font-size:12px;color:var(--mut)}
 .disp .r b{font-size:18px;color:var(--ink)}
 h2{font-size:11px;letter-spacing:.09em;text-transform:uppercase;color:var(--nav);border-bottom:1px solid var(--line);padding-bottom:5px;margin:24px 0 10px}
 table{width:100%;border-collapse:collapse;font-size:12px}
 .fields td{padding:6px 10px;border:1px solid var(--line2);vertical-align:top}
 .fields td:nth-child(odd){background:var(--head);color:var(--mut);width:20%;font-size:11px;text-transform:uppercase;letter-spacing:.03em}
 .fill{display:inline-block;min-width:120px;border-bottom:1px solid var(--line);height:14px}
 .two{display:grid;grid-template-columns:1.15fr 1fr;gap:24px;align-items:start}
 .grid-t td{padding:6px 10px;border-bottom:1px solid var(--line2)}
 .grid-t td.n{text-align:right;font-family:ui-monospace,monospace}
 .grid-t tr.hl td{background:var(--head);font-weight:700}
 .data{border:1px solid var(--line)}
 .data th{background:var(--head);text-align:left;padding:7px 10px;font-size:10.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut);border-bottom:1px solid var(--line)}
 .data th.n,.data td.n{text-align:right;font-family:ui-monospace,monospace}
 .data td{padding:6px 10px;border-bottom:1px solid var(--line2)}
 .data tr:last-child td{border-bottom:none}.data tbody tr:nth-child(even) td{background:#fafbfc}
 .tag{font-size:9.5px;color:var(--bad);border:1px solid var(--bad);border-radius:3px;padding:0 4px;letter-spacing:.03em;vertical-align:1px}
 .muted{color:var(--mut);text-align:center;padding:12px}
 .evi{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:12px}
 .evi figure{margin:0;border:1px solid var(--line);border-radius:4px;overflow:hidden}
 .evi .ph{aspect-ratio:1;background:#0e0f12}.evi img{width:100%;height:100%;object-fit:cover;display:block}
 .evi figcaption{font-size:10px;padding:5px 7px;color:var(--ink)}.evi figcaption span{display:block;color:var(--mut);font-family:ui-monospace,monospace}
 .sign{display:grid;grid-template-columns:1fr 1fr;gap:40px;margin-top:34px}
 .sign .l{border-top:1px solid var(--ink);padding-top:6px;font-size:11px;color:var(--mut)}.sign .l b{display:block;color:var(--ink);font-size:12px;margin-bottom:22px}
 .ft{margin-top:26px;border-top:1px solid var(--line);padding-top:10px;font-size:10px;color:var(--mut);display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap}
 tr{break-inside:avoid}
 @media print{@page{size:A4;margin:14mm}html,body{background:#fff}.doc{box-shadow:none;margin:0;max-width:none;padding:0}}
</style></head><body><div class="doc">
 <div class="hd">
  <div class="brand">${logo}<span>Fabric Inspect<small>Automated fabric QA · unsupervised</small></span></div>
  <div class="t"><h1>Fabric Inspection Report</h1><div class="sub">CERTIFICATE OF INSPECTION</div></div>
 </div>

 <div class="disp ${accepted ? "ok" : "bad"}">
  <span class="stamp">${accepted ? "ACCEPTED" : "REJECTED"}</span>
  <span class="d">Disposition — <b>${accepted ? "First Quality" : "Second Quality"}</b><br>${accepted ? "Within" : "Exceeds"} acceptance tolerance of ${TOL} points / 100 yd².</span>
  <span class="r"><b>${g.p100}</b> pts / 100 yd²<br>${defective} of ${results.length} frames defective</span>
 </div>

 <h2>Inspection details</h2>
 <table class="fields"><tbody>${paramRows}</tbody></table>

 <div class="two">
  <div>
   <h2>Defect summary by type</h2>
   <table class="data"><thead><tr><th>Defect type</th><th class="n">Count</th><th class="n">Points</th></tr></thead><tbody>${typeRows}</tbody></table>
  </div>
  <div>
   <h2>Grade calculation (4-Point)</h2>
   <table class="grid-t"><tbody>${calcRows}</tbody></table>
  </div>
 </div>

 <h2>Defect log</h2>
 <table class="data"><thead><tr><th class="n">#</th><th>Frame</th><th>Defect type</th><th class="n">Size (mm)</th><th>Severity</th><th class="n">Points</th></tr></thead><tbody>${logRows}</tbody></table>

 ${evidence ? `<h2>Defect evidence</h2><div class="evi">${evidence}</div>` : ""}

 <div class="sign">
  <div class="l"><b></b>Inspected by (name / signature / date)</div>
  <div class="l"><b></b>Approved by (name / signature / date)</div>
 </div>

 <div class="ft"><span>${reportNo} · Generated ${esc(when)} · Fabric Inspect (DINOv3 + PatchCore)</span><span>Machine-assisted inspection — calibration ${ppm} px/mm is an operator input; verify against a physical scale. Not a substitute for accredited testing.</span></div>
</div></body></html>`;

  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([html], { type: "text/html" }));
  a.download = `${reportNo}_${cat}.html`;
  a.click();
  URL.revokeObjectURL(a.href);
}
