from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from agent_race.config import load_settings
from agent_race.memory import AgentRaceStore
from agent_race.scheduler import AgentRaceScheduler


settings = load_settings()
store = AgentRaceStore(settings.db_path, settings.workspace_dir)
scheduler = AgentRaceScheduler(settings, store)
base = settings.base_path


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.scheduler_enabled:
        scheduler.start()
    yield
    await scheduler.stop()


app = FastAPI(title="Agent Race", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get(base)
async def dashboard_redirect() -> RedirectResponse:
    return RedirectResponse(f"{base}/")


@app.get(f"{base}/", response_class=HTMLResponse)
async def dashboard() -> str:
    return DASHBOARD_HTML.replace("__BASE_PATH__", base)


@app.get(f"{base}/api/status")
async def status() -> JSONResponse:
    payload = store.overview()
    payload["runtime"] = scheduler.runtime_status()
    return JSONResponse(payload)


@app.post(f"{base}/api/tick")
async def trigger_tick() -> dict[str, bool]:
    accepted = await scheduler.trigger_once()
    return {"accepted": accepted}


DASHBOARD_HTML = """<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Agent Race</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101214;
      --panel: #181c20;
      --panel-2: #20262b;
      --text: #edf2f4;
      --muted: #aab4bd;
      --line: #33404a;
      --accent: #8cc63f;
      --warn: #f6b44b;
      --bad: #ff6b6b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
      letter-spacing: 0;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
      padding: 18px 22px;
      border-bottom: 1px solid var(--line);
      background: #121619;
      position: sticky;
      top: 0;
      z-index: 2;
    }
    h1 { font-size: 22px; margin: 0; }
    button {
      background: var(--accent);
      color: #071006;
      border: 0;
      border-radius: 6px;
      padding: 9px 12px;
      font-weight: 700;
      cursor: pointer;
    }
    main { padding: 20px; display: grid; gap: 18px; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      min-width: 0;
    }
    .metric { font-size: 28px; font-weight: 800; margin-top: 6px; }
    .muted { color: var(--muted); }
    .agents { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; }
    .agent { background: var(--panel-2); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }
    .agent h3 { margin: 0 0 8px; font-size: 16px; overflow-wrap: anywhere; }
    .row { display: flex; justify-content: space-between; gap: 12px; margin: 7px 0; }
    .summary { line-height: 1.55; white-space: pre-wrap; }
    table { width: 100%; border-collapse: collapse; }
    th, td { text-align: left; padding: 9px; border-bottom: 1px solid var(--line); vertical-align: top; }
    th { color: var(--muted); font-size: 13px; }
    td { font-size: 14px; }
    code { color: #c9e98f; overflow-wrap: anywhere; }
    .ok { color: var(--accent); }
    .fallback, .warn { color: var(--warn); }
    .error, .rate_limited { color: var(--bad); }
    @media (max-width: 900px) {
      header { align-items: flex-start; flex-direction: column; }
      .grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 560px) {
      main { padding: 12px; }
      .grid { grid-template-columns: 1fr; }
      th, td { font-size: 12px; padding: 7px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Agent Race</h1>
      <div class="muted">Crypto arbitrage LLM arena</div>
    </div>
    <button id="tick">Run Tick</button>
  </header>
  <main>
    <section class="grid">
      <div class="panel"><div class="muted">Agents</div><div class="metric" id="agent-count">-</div></div>
      <div class="panel"><div class="muted">Cycle</div><div class="metric" id="tick-count">-</div></div>
      <div class="panel"><div class="muted">Interval</div><div class="metric" id="interval">-</div></div>
      <div class="panel"><div class="muted">Live Trading</div><div class="metric" id="trading">-</div></div>
    </section>
    <section class="panel">
      <h2>LLM Monitor Summary</h2>
      <div class="summary" id="arena-summary">Loading...</div>
    </section>
    <section>
      <h2>Agents</h2>
      <div class="agents" id="agents"></div>
    </section>
    <section class="panel">
      <h2>Strategy Candidates</h2>
      <table><thead><tr><th>Time</th><th>Agent</th><th>Title</th><th>Edge bps</th><th>Risk</th></tr></thead><tbody id="strategies"></tbody></table>
    </section>
    <section class="panel">
      <h2>Recent Events</h2>
      <table><thead><tr><th>Time</th><th>Agent</th><th>Kind</th><th>Message</th></tr></thead><tbody id="events"></tbody></table>
    </section>
  </main>
  <script>
    const BASE = "__BASE_PATH__";
    const text = (value) => value === null || value === undefined || value === "" ? "-" : String(value);
    async function refresh() {
      const res = await fetch(`${BASE}/api/status`, { cache: "no-store" });
      const data = await res.json();
      document.getElementById("agent-count").textContent = text(data.runtime.agent_count);
      document.getElementById("tick-count").textContent = text(data.runtime.tick_count);
      document.getElementById("interval").textContent = `${data.runtime.tick_seconds}s`;
      document.getElementById("trading").textContent = data.runtime.live_trading_enabled ? "ON" : "OFF";
      const summary = data.arena_summary || {};
      document.getElementById("arena-summary").textContent = summary.summary || "No summary yet.";

      document.getElementById("agents").innerHTML = (data.agents || []).map(agent => {
        const payload = agent.payload_json || {};
        const decision = payload.decision || {};
        return `<article class="agent">
          <h3>${text(agent.name)}</h3>
          <div class="row"><span class="muted">Model</span><code>${text(agent.model)}</code></div>
          <div class="row"><span class="muted">Status</span><strong class="${text(agent.status)}">${text(agent.status)}</strong></div>
          <div class="row"><span class="muted">Score</span><strong>${Number(agent.score || 0).toFixed(3)}</strong></div>
          <div class="row"><span class="muted">Last Tick</span><span>${text(agent.last_tick_at)}</span></div>
          <p class="summary">${text(agent.last_summary || decision.summary)}</p>
        </article>`;
      }).join("");

      document.getElementById("strategies").innerHTML = (data.strategies || []).map(item =>
        `<tr><td>${text(item.ts)}</td><td><code>${text(item.agent_id)}</code></td><td>${text(item.title)}</td><td>${text(item.expected_edge_bps)}</td><td>${text(item.risk_score)}</td></tr>`
      ).join("");

      document.getElementById("events").innerHTML = (data.events || []).map(item =>
        `<tr><td>${text(item.ts)}</td><td><code>${text(item.agent_id)}</code></td><td>${text(item.kind)}</td><td>${text(item.message)}</td></tr>`
      ).join("");
    }
    document.getElementById("tick").addEventListener("click", async () => {
      await fetch(`${BASE}/api/tick`, { method: "POST" });
      setTimeout(refresh, 1000);
    });
    refresh();
    setInterval(refresh, 15000);
  </script>
</body>
</html>"""
