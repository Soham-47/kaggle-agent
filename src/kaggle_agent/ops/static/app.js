const extraLines = [];

const FLOW = [
  ["gateway", "Gateway", "/run"],
  ["memory", "Memory", "pack"],
  ["research", "Research", "loop"],
  ["plan", "Plan", "loop"],
  ["code", "Code", "loop"],
  ["smoke", "Smoke", "local"],
  ["kernel", "Kernel", "train"],
  ["submit", "Submit", "LB"],
  ["heal", "Heal", "next"],
  ["ops", "LLM Ops", "trace"],
];

function $(id) {
  return document.getElementById(id);
}

function el(tag, attrs, ...kids) {
  const node = document.createElement(tag);
  Object.entries(attrs || {}).forEach(([k, v]) => {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  });
  kids.forEach((c) => node.append(c));
  return node;
}

function kv(label, value) {
  const box = el("div", {});
  box.append(el("dt", { text: label }), el("dd", { text: String(value) }));
  return box;
}

function paintArch(active) {
  const wrap = $("arch");
  wrap.replaceChildren();
  const idx = FLOW.findIndex((item) => item[0] === active);
  FLOW.forEach((item, i) => {
    if (i) wrap.append(el("span", { class: "arrow", text: "→" }));
    const box = el("div", { class: "node" });
    if (item[0] === active) box.classList.add("active");
    else if (idx > 0 && i < idx) box.classList.add("done");
    box.append(el("span", { class: "lab", text: item[1] }));
    box.append(el("span", { class: "hint", text: item[2] }));
    wrap.append(box);
  });
}

function paintTerm(lines) {
  const term = $("term");
  const atBottom = term.scrollHeight - term.scrollTop - term.clientHeight < 40;
  term.replaceChildren();
  (lines || []).forEach((ln) => {
    const row = el("div", { class: ln.level || "info" });
    const ts = String(ln.ts || "");
    const short = ts.length > 8 ? ts.slice(11, 19) : ts;
    if (short) row.append(el("span", { class: "ts", text: short + "  " }));
    row.append(document.createTextNode(ln.text || ""));
    term.append(row);
  });
  if (!lines || !lines.length) {
    term.append(el("div", { class: "info", text: "Idle. Press Run live or type /run." }));
  }
  if (atBottom) term.scrollTop = term.scrollHeight;
}

function paint(data) {
  const st = data.state || {};
  const running = !!data.running;
  $("status-bar").replaceChildren(
    kv("phase", st.phase || "IDLE"),
    kv("score", st.public_best || "—"),
    kv("result", st.last_result || "—"),
    kv("gate", data.evals && data.evals.passed ? "open" : "closed"),
  );
  $("term-state").textContent = running ? "running" : "idle";
  $("run-live").disabled = running;
  $("run-dry").disabled = running;
  paintArch(data.active_node || "");
  const ev = data.evals || {};
  $("evals").replaceChildren(
    ...(ev.checks || []).map((c) =>
      el("span", { class: "pill " + (c.ok ? "ok" : "bad"), text: c.id.replaceAll("_", " ") }),
    ),
  );
  const u = data.usage || {};
  const loop = data.loop || {};
  const invalid = loop.counts ? loop.counts["research:invalid_json"] || 0 : 0;
  $("ops").textContent =
    `${u.calls || 0} calls · ${u.tokens_in || 0} in / ${u.tokens_out || 0} out` +
    (invalid ? ` · ${invalid} bad JSON` : "");
  paintTerm([...(data.terminal || []), ...extraLines]);
}

async function load() {
  const err = $("load-error");
  try {
    const res = await fetch("/api/snapshot");
    if (!res.ok) throw new Error("snapshot " + res.status);
    paint(await res.json());
    err.hidden = true;
  } catch (e) {
    err.hidden = false;
    err.textContent = "Could not load: " + e.message;
  }
}

async function send(text) {
  extraLines.push({ level: "info", text: "> " + text, ts: "" });
  const res = await fetch("/api/command", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  const body = await res.json();
  extraLines.push({
    level: body.ok ? "ok" : "error",
    text: body.reply || String(res.status),
    ts: "",
  });
  load();
}

$("run-live").addEventListener("click", () => send("/run"));
$("run-dry").addEventListener("click", () => send("/run dry"));
$("cmd").addEventListener("submit", (e) => {
  e.preventDefault();
  const input = $("cmd-in");
  const text = (input.value || "").trim();
  if (!text) return;
  input.value = "";
  send(text.startsWith("/") ? text : "/" + text);
});

load();
setInterval(load, 2000);
