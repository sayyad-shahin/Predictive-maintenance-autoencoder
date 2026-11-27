(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const PAD_L = 52;
  let U = null, cur = 0, timer = null;

  async function api(url, opts) {
    const r = await fetch(url, opts);
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || "Request failed");
    return d;
  }
  const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
  const num = (v, nd = 2) => (v === null || v === undefined ? "-" : Number(v).toFixed(nd));

  // ---- a small line chart on a canvas -------------------------------------------------
  function chart(canvas, o) {
    const dpr = window.devicePixelRatio || 1, W = canvas.clientWidth || 600, H = canvas.clientHeight || 200;
    canvas.width = W * dpr; canvas.height = H * dpr;
    const g = canvas.getContext("2d");
    if (!g) return;
    g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, W, H);
    const R = 10, T = 8, B = 24, pw = W - PAD_L - R, ph = H - T - B;
    const f = o.log ? (v) => Math.log10(Math.max(v, 1e-3)) : (v) => v;
    const vals = o.y.filter((v) => v !== null).concat(o.hline !== undefined ? [o.hline] : []).map(f);
    let lo = Math.min(...vals), hi = Math.max(...vals);
    if (hi - lo < 1e-9) hi = lo + 1;
    const pad = (hi - lo) * 0.06; lo -= pad; hi += pad;
    const x0 = o.x[0], x1 = o.x[o.x.length - 1] || x0 + 1;
    const X = (x) => PAD_L + ((x - x0) / (x1 - x0 || 1)) * pw, Y = (v) => T + ph - ((f(v) - lo) / (hi - lo)) * ph;
    const cs = getComputedStyle(document.body);
    const ink = cs.color, line = cs.getPropertyValue("--line").trim() || "#ccc", muted = cs.getPropertyValue("--muted").trim() || "#777";
    g.font = "11px sans-serif"; g.lineWidth = 1; g.strokeStyle = line; g.fillStyle = muted;
    const yt = [];
    if (o.log) { for (let k = Math.ceil(lo); k <= Math.floor(hi); k++) yt.push(Math.pow(10, k)); }
    else { for (let i = 0; i <= 3; i++) yt.push(Math.pow(10, 0) * (lo + ((hi - lo) * i) / 3)); }
    yt.forEach((v) => { const yy = o.log ? Y(v) : T + ph - ((v - lo) / (hi - lo)) * ph; g.beginPath(); g.moveTo(PAD_L, yy); g.lineTo(W - R, yy); g.stroke(); g.textAlign = "right"; g.fillText(String(Number(v.toPrecision(3))), PAD_L - 6, yy + 4); });
    g.textAlign = "center";
    for (let i = 0; i <= 5; i++) { const xv = x0 + ((x1 - x0) * i) / 5; g.fillText(String(Math.round(xv)), X(xv), H - 8); }
    if (o.hline !== undefined) { g.strokeStyle = "#d93a3a"; g.setLineDash([6, 4]); g.lineWidth = 1.5; g.beginPath(); g.moveTo(PAD_L, Y(o.hline)); g.lineTo(W - R, Y(o.hline)); g.stroke(); g.setLineDash([]); }
    (o.vlines || []).forEach((v) => { if (!v.show) return; g.strokeStyle = v.color === "ink" ? ink : v.color; g.lineWidth = 2; g.beginPath(); g.moveTo(X(v.x), T); g.lineTo(X(v.x), T + ph); g.stroke(); });
    g.strokeStyle = o.color || "#3b5bdb"; g.lineWidth = 2; g.beginPath();
    let started = false;
    for (let i = 0; i <= o.upto && i < o.x.length; i++) { if (o.y[i] === null) continue; const px = X(o.x[i]), py = Y(o.y[i]); if (started) g.lineTo(px, py); else { g.moveTo(px, py); started = true; } }
    g.stroke();
    const cx = X(o.x[Math.min(o.upto, o.x.length - 1)]);
    g.strokeStyle = muted; g.lineWidth = 1; g.setLineDash([2, 3]); g.beginPath(); g.moveTo(cx, T); g.lineTo(cx, T + ph); g.stroke(); g.setLineDash([]);
  }

  // ---- drawing the selected machine at the current cycle -------------------------------
  function render() {
    if (!U) return;
    const n = U.cycles.length;
    $("slider").value = cur;
    $("cycleNum").textContent = U.cycles[cur];
    $("cycleOf").textContent = `of ${n}`;
    chart($("chartHealth"), { x: U.cycles, y: U.health, upto: cur, hline: 1, log: true, vlines: [
      { x: U.alarm_cycle, color: "#e8590c", show: U.alarm_cycle !== null && U.cycles[cur] >= U.alarm_cycle },
      { x: U.failure_cycle, color: "ink", show: cur >= n - 1 }] });
    chart($("chartSensor"), { x: U.cycles, y: U.sensors[$("sensor").value], upto: cur, color: "#12806f" });

    const status = U.status[cur];
    const b = $("badge"); b.textContent = status; b.className = "badge " + status;
    $("advice").textContent = U.advice[status];
    $("hVal").textContent = `${num(U.health[cur])} × alarm level`;
    const rul = U.rul_estimate[cur];
    $("rulVal").textContent = rul === null ? "-" : `about ${rul} cycles`;
    $("trueVal").textContent = `${U.true_rul[cur]} cycles`;
    $("truth").textContent = U.truth ? `Simulation ground truth: fault = ${U.truth.fault.replace("_", " ")}, wear began at cycle ${U.truth.degrade_start}.` : "";

    const rows = Object.keys(U.sensor_ratio).map((s) => [s, U.sensor_ratio[s][cur]]).sort((a, b2) => b2[1] - a[1]).slice(0, 5);
    const scale = Math.max(3, rows[0][1]);
    const box = $("bars"); box.innerHTML = "";
    rows.forEach(([s, v]) => {
      const row = el("div", "barrow"), track = el("div", "track"), fill = el("div", "fill" + (v >= 3 ? " hot" : ""));
      fill.style.width = Math.min(100, (v / scale) * 100) + "%"; track.appendChild(fill);
      row.append(el("span", "", s.replace("_", " ")), track, el("span", "num", num(v, 1) + "×")); box.appendChild(row);
    });
  }

  // ---- replay ---------------------------------------------------------------------------------
  function stop() { if (timer) { clearInterval(timer); timer = null; } $("playBtn").textContent = "▶ Play"; }
  function play() {
    if (timer) { stop(); return; }
    if (!U) return;
    if (cur >= U.cycles.length - 1) cur = 0;
    $("playBtn").textContent = "❚❚ Pause";
    timer = setInterval(() => {
      cur = Math.min(U.cycles.length - 1, cur + Number($("speed").value));
      render();
      if (cur >= U.cycles.length - 1) stop();
    }, 70);
  }

  async function loadUnit(u) {
    stop();
    U = await api("/api/unit/" + u);
    $("slider").max = U.cycles.length - 1;
    const sel = $("sensor"), keep = sel.value;
    if (!sel.options.length) Object.keys(U.sensors).forEach((s) => { const o = el("option", "", s.replace("_", " ")); o.value = s; sel.appendChild(o); });
    if (keep) sel.value = keep;
    cur = U.cycles.length - 1;
    render();
  }

  // ---- upload ----------------------------------------------------------------------------------
  async function score() {
    const f = $("file").files[0], msg = $("uploadMsg");
    if (!f) { msg.textContent = "Choose a CSV file first."; return; }
    $("scoreBtn").disabled = true; msg.textContent = "Scoring...";
    try {
      const fd = new FormData(); fd.append("file", f);
      const d = await api("/api/score", { method: "POST", body: fd });
      const body = $("uploadTable").querySelector("tbody"); body.innerHTML = "";
      d.units.forEach((u) => {
        const tr = el("tr"), tag = el("td"), st = el("span", "tag " + u.final_status, u.final_status); tag.appendChild(st);
        const top = u.final_status === "normal" ? "-" : u.top_sensors.map((t) => `${t.sensor.replace("_", " ")} (${t.times_normal}×)`).join(", ");
        tr.append(el("td", "", "Machine " + u.unit), el("td", "num", u.cycles), tag, el("td", "num", num(u.health_now)), el("td", "", u.alarm_cycle === null ? "none" : "cycle " + u.alarm_cycle), el("td", "", top));
        body.appendChild(tr);
      });
      $("uploadTable").classList.remove("hidden");
      const a = $("dl"); a.href = URL.createObjectURL(new Blob([d.csv], { type: "text/csv" })); a.classList.remove("hidden");
      msg.textContent = `Scored ${d.units.length} machine(s).`;
    } catch (e) { msg.textContent = e.message; $("uploadTable").classList.add("hidden"); $("dl").classList.add("hidden"); }
    finally { $("scoreBtn").disabled = false; }
  }

  // ---- start ------------------------------------------------------------------------------------
  async function init() {
    const [model, units] = await Promise.all([api("/api/model"), api("/api/units")]);
    const c = model.config;
    $("modelBadge").textContent = `Autoencoder · ${c.features.length} sensors · ${c.window}-cycle window · alarm after ${c.patience} cycles above threshold`;
    $("cols").textContent = c.features.join(", ");
    const tb = $("metrics").querySelector("tbody"), ae = model.autoencoder, pca = model.pca_baseline;
    if (ae) {
      const pct = (v) => (v === null || v === undefined ? "-" : (v * 100).toFixed(0) + "%");
      [["Failures caught", "detection_rate", pct], ["Machines with false alarm", "false_alarm_machines", pct],
       ["Mean warning time", "mean_lead_time_cycles", (v) => (v === null || v === undefined ? "-" : v + " cycles")],
       ["AUROC (last 50 cycles)", "auroc_last_50_cycles", (v) => num(v, 3)]].forEach(([label, key, fmt]) => {
        const tr = el("tr"); tr.append(el("td", "", label), el("td", "num", fmt(ae[key])), el("td", "num", pca ? fmt(pca[key]) : "-")); tb.appendChild(tr);
      });
      const note = el("tr"); const td = el("td", "muted small", `${ae.machines_tested} machines not used for training`); td.colSpan = 3; note.appendChild(td); tb.appendChild(note);
    }
    const sel = $("unit");
    units.forEach((u) => { const o = el("option", "", `Machine ${u.unit} · ${u.cycles} cycles${u.is_test ? " · unseen test machine" : ""}`); o.value = u.unit; sel.appendChild(o); });
    sel.addEventListener("change", () => loadUnit(sel.value));
    $("playBtn").addEventListener("click", play);
    $("endBtn").addEventListener("click", () => { stop(); if (U) { cur = U.cycles.length - 1; render(); } });
    $("slider").addEventListener("input", () => { stop(); cur = Number($("slider").value); render(); });
    $("sensor").addEventListener("change", render);
    $("scoreBtn").addEventListener("click", score);
    ["chartHealth", "chartSensor"].forEach((id) => $(id).addEventListener("click", (e) => {
      if (!U) return;
      const r = e.target.getBoundingClientRect(), frac = (e.clientX - r.left - PAD_L) / (r.width - PAD_L - 10);
      stop(); cur = Math.max(0, Math.min(U.cycles.length - 1, Math.round(frac * (U.cycles.length - 1)))); render();
    }));
    window.addEventListener("resize", render);
    await loadUnit(units[0].unit);
  }
  init().catch((e) => { document.body.insertAdjacentHTML("afterbegin", `<p style="padding:20px;color:#d93a3a">Could not start: ${e.message}</p>`); });
})();
