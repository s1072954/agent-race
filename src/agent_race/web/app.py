from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from agent_race.config import load_settings
from agent_race.memory import AgentRaceStore
from agent_race.scheduler import AgentRaceScheduler


settings = load_settings()
store = AgentRaceStore(settings.db_path, settings.workspace_dir)
scheduler = AgentRaceScheduler(settings, store)
base = settings.base_path


class SchedulerConfigRequest(BaseModel):
    tick_seconds: int = Field(ge=60, le=86_400)
    fallback_tick_seconds: int = Field(ge=60, le=86_400)


class RootChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


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
    runtime = scheduler.runtime_status()
    active_models = set(runtime["models"])
    payload["agents"] = [agent for agent in payload.get("agents", []) if agent.get("model") in active_models]
    payload["runtime"] = runtime
    return JSONResponse(payload)


@app.post(f"{base}/api/tick")
async def trigger_tick() -> dict[str, bool]:
    accepted = await scheduler.trigger_once()
    return {"accepted": accepted}


@app.post(f"{base}/api/config")
async def update_config(config: SchedulerConfigRequest) -> dict[str, object]:
    updated = scheduler.update_config(config.tick_seconds, config.fallback_tick_seconds)
    return {"ok": True, "scheduler_config": updated}


@app.post(f"{base}/api/root-chat")
async def root_chat(request: RootChatRequest) -> dict[str, object]:
    return await scheduler.ask_root_agent(request.message)


DASHBOARD_HTML = """<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Agent 競賽</title>
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
    button:disabled { cursor: progress; opacity: 0.65; }
    input, textarea {
      width: 100%;
      min-height: 38px;
      background: #0f1316;
      color: var(--text);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      font: inherit;
    }
    textarea { min-height: 96px; resize: vertical; line-height: 1.5; }
    label { display: grid; gap: 6px; color: var(--muted); font-size: 13px; }
    main { padding: 20px; display: grid; gap: 18px; }
    .grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
    .control-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; align-items: end; }
    .chat-grid { display: grid; grid-template-columns: minmax(0, 1fr) 180px; gap: 12px; align-items: end; }
    .chat-answer { margin-top: 12px; background: #0f1316; border: 1px solid var(--line); border-radius: 8px; padding: 12px; min-height: 72px; }
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
    .markdown {
      line-height: 1.65;
      overflow-x: auto;
      white-space: normal;
    }
    .markdown h1, .markdown h2, .markdown h3, .markdown h4 {
      margin: 18px 0 8px;
      line-height: 1.3;
    }
    .markdown h1 { font-size: 22px; }
    .markdown h2 { font-size: 19px; }
    .markdown h3 { font-size: 17px; }
    .markdown p { margin: 10px 0; }
    .markdown ul, .markdown ol { margin: 10px 0 10px 22px; padding: 0; }
    .markdown li { margin: 5px 0; }
    .markdown table {
      min-width: 760px;
      margin: 12px 0;
      border: 1px solid var(--line);
      background: #121619;
    }
    .markdown th {
      color: var(--text);
      background: #20262b;
      font-size: 13px;
      white-space: nowrap;
    }
    .markdown td {
      line-height: 1.55;
    }
    .markdown strong { color: #ffffff; }
    .limit-alert {
      margin-top: 14px;
      border: 1px solid rgba(255, 107, 107, 0.45);
      background: rgba(255, 107, 107, 0.08);
      border-radius: 8px;
      padding: 12px;
    }
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
      .control-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .chat-grid { grid-template-columns: 1fr; }
    }
    @media (max-width: 560px) {
      main { padding: 12px; }
      .grid { grid-template-columns: 1fr; }
      .control-grid { grid-template-columns: 1fr; }
      th, td { font-size: 12px; padding: 7px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Agent 競賽</h1>
      <div class="muted">加密貨幣套利 LLM 競技場</div>
    </div>
    <button id="tick">手動執行一輪</button>
  </header>
  <main>
    <section class="grid">
      <div class="panel"><div class="muted">Agent 數量</div><div class="metric" id="agent-count">-</div></div>
      <div class="panel"><div class="muted">已完成輪數</div><div class="metric" id="tick-count">-</div></div>
      <div class="panel"><div class="muted">執行間隔</div><div class="metric" id="interval">-</div></div>
      <div class="panel"><div class="muted">實盤交易</div><div class="metric" id="trading">-</div></div>
    </section>
    <section class="panel">
      <h2>排程控制</h2>
      <div class="control-grid">
        <label>
          Agent 執行間隔（秒）
          <input id="interval-input" type="number" min="60" max="86400" step="60" />
        </label>
        <label>
          超額 fallback 間隔（秒）
          <input id="fallback-input" type="number" min="60" max="86400" step="60" />
        </label>
        <button id="save-config">儲存頻率</button>
        <div>
          <div class="muted">Fallback 狀態</div>
          <strong id="fallback-state">-</strong>
        </div>
      </div>
      <div class="muted" id="config-status" style="margin-top:10px;">時間顯示：台灣時間 UTC+8</div>
      <div class="limit-alert" id="limit-alert" hidden>
        <strong class="error">偵測到使用額度或速率限制</strong>
        <div class="row"><span class="muted">發生時間</span><span id="limit-time">-</span></div>
        <div class="row"><span class="muted">模型</span><code id="limit-model">-</code></div>
        <div class="row"><span class="muted">建議等待時間</span><span id="limit-retry">-</span></div>
        <div class="row"><span class="muted">超額保護間隔</span><span id="limit-fallback">-</span></div>
        <p class="summary" id="limit-message"></p>
      </div>
    </section>
    <section class="panel">
      <h2>LLM 監控摘要</h2>
      <div class="summary markdown" id="arena-summary">載入中...</div>
    </section>
    <section class="panel">
      <h2>詢問 Root Agent</h2>
      <div class="chat-grid">
        <label>
          問題
          <textarea id="root-question" maxlength="2000" placeholder="例如：你目前最看好的策略是什麼？還卡在哪些驗證？下一步要做什麼？"></textarea>
        </label>
        <button id="ask-root">送出問題</button>
      </div>
      <div class="summary markdown chat-answer" id="root-answer">尚未提問。</div>
    </section>
    <section class="panel">
      <h2>流程瓶頸</h2>
      <div class="grid">
        <div class="panel"><div class="muted">紙上可追蹤</div><div class="metric" id="ready-count">-</div></div>
        <div class="panel"><div class="muted">僅研究</div><div class="metric" id="research-count">-</div></div>
        <div class="panel"><div class="muted">阻擋</div><div class="metric" id="blocked-count">-</div></div>
        <div class="panel"><div class="muted">借幣資料源</div><div class="metric" id="borrow-source">-</div></div>
      </div>
      <table style="margin-top:14px;"><thead><tr><th>主要阻擋原因</th><th>次數</th></tr></thead><tbody id="blockers"></tbody></table>
    </section>
    <section class="panel">
      <h2>模型可靠度</h2>
      <table><thead><tr><th>模型</th><th>呼叫</th><th>成功率</th><th>Timeout</th><th>格式修復</th><th>格式失敗</th><th>平均延遲</th><th>平均 Prompt</th></tr></thead><tbody id="llm-diagnostics"></tbody></table>
    </section>
    <section>
      <h2>Agent 狀態</h2>
      <div class="agents" id="agents"></div>
    </section>
    <section class="panel">
      <h2>市場掃描候選</h2>
      <table><thead><tr><th>時間</th><th>類型</th><th>標的</th><th>候選機會</th><th>毛邊際 bps</th><th>估計成本 bps</th><th>淨邊際 bps</th><th>狀態</th></tr></thead><tbody id="opportunities"></tbody></table>
    </section>
    <section class="panel">
      <h2>紙上交易驗證</h2>
      <table><thead><tr><th>時間</th><th>類型</th><th>標的</th><th>驗證項目</th><th>本金 USDT</th><th>毛邊際 bps</th><th>估計成本 bps</th><th>可成交淨 bps</th><th>狀態</th><th>阻擋原因</th></tr></thead><tbody id="paper-signals"></tbody></table>
    </section>
    <section class="panel">
      <h2>候選策略</h2>
      <table><thead><tr><th>時間</th><th>Agent</th><th>標題</th><th>預估優勢 bps</th><th>風險</th></tr></thead><tbody id="strategies"></tbody></table>
    </section>
    <section class="panel">
      <h2>近期事件</h2>
      <table><thead><tr><th>時間</th><th>Agent</th><th>類型</th><th>訊息</th></tr></thead><tbody id="events"></tbody></table>
    </section>
  </main>
  <script>
    const BASE = "__BASE_PATH__";
    const text = (value) => value === null || value === undefined || value === "" ? "-" : String(value);
    const html = (value) => text(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
    const twFormatter = new Intl.DateTimeFormat("zh-TW", {
      timeZone: "Asia/Taipei",
      year: "numeric",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false
    });
    function twTime(value) {
      if (!value) return "-";
      const date = new Date(value);
      if (Number.isNaN(date.getTime())) return text(value);
      return twFormatter.format(date);
    }
    function statusClass(value) {
      return text(value).toLowerCase().replace(/[^a-z0-9_-]/g, "");
    }
    function agentStatusLabel(value) {
      const labels = {
        ok: "正常",
        idle: "待命",
        fallback: "Fallback 保護",
        error: "錯誤",
        rate_limited: "速率限制"
      };
      return labels[text(value).toLowerCase()] || text(value);
    }
    function eventKindLabel(value) {
      const labels = {
        cycle_started: "循環開始",
        cycle_completed: "循環完成",
        scheduler_error: "排程錯誤",
        scheduler_config_updated: "排程設定更新",
        tick_started: "Agent 開始執行",
        tick_completed: "Agent 執行完成",
        root_agent_error: "Root Agent 錯誤",
        subagent_completed: "Sub-agent 完成",
        subagent_error: "Sub-agent 錯誤",
        llm_limit_fallback: "LLM 超額保護"
      };
      return labels[text(value)] || text(value);
    }
    function opportunityKindLabel(value) {
      const labels = {
        spot_spread: "現貨跨所價差",
        funding_rate: "資金費率",
      };
      return labels[text(value)] || text(value);
    }
    function opportunityStatusLabel(value) {
      const labels = {
        actionable_research: "可研究",
        watch: "觀察",
        ignore: "忽略",
      };
      return labels[text(value)] || text(value);
    }
    function paperSignalStatusLabel(value) {
      const labels = {
        paper_trade_ready: "紙上可追蹤",
        research_only: "僅研究",
        watch: "觀察",
        blocked: "阻擋",
      };
      return labels[text(value)] || text(value);
    }
    function blockersText(value) {
      if (Array.isArray(value)) return value.join("；");
      if (!value) return "-";
      return text(value);
    }
    function setInputUnlessFocused(id, value) {
      const input = document.getElementById(id);
      if (document.activeElement !== input) input.value = value ?? "";
    }
    function inlineMarkdown(value) {
      return html(value)
        .replace(/&lt;br\\s*\\/?&gt;/gi, "<br>")
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\\*\\*([^*]+)\\*\\*/g, "<strong>$1</strong>");
    }
    function splitMarkdownRow(line) {
      return line.trim().replace(/^\\|/, "").replace(/\\|$/, "").split("|").map(cell => cell.trim());
    }
    function isTableSeparator(line) {
      return /^\\|?\\s*:?-{3,}:?\\s*(\\|\\s*:?-{3,}:?\\s*)+\\|?$/.test(line.trim());
    }
    function markdownToHtml(markdown) {
      const lines = text(markdown).split(/\\r?\\n/);
      const out = [];
      for (let i = 0; i < lines.length; i += 1) {
        const line = lines[i];
        const trimmed = line.trim();
        if (!trimmed) continue;
        if (trimmed.startsWith("|") && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
          const headers = splitMarkdownRow(trimmed);
          i += 2;
          const rows = [];
          while (i < lines.length && lines[i].trim().startsWith("|")) {
            rows.push(splitMarkdownRow(lines[i]));
            i += 1;
          }
          i -= 1;
          out.push("<table><thead><tr>" + headers.map(cell => `<th>${inlineMarkdown(cell)}</th>`).join("") + "</tr></thead><tbody>" + rows.map(row => "<tr>" + row.map(cell => `<td>${inlineMarkdown(cell)}</td>`).join("") + "</tr>").join("") + "</tbody></table>");
          continue;
        }
        const heading = /^(#{1,4})\\s+(.+)$/.exec(trimmed);
        if (heading) {
          const level = heading[1].length;
          out.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
          continue;
        }
        if (/^[-*]\\s+/.test(trimmed)) {
          const items = [];
          while (i < lines.length && /^[-*]\\s+/.test(lines[i].trim())) {
            items.push(lines[i].trim().replace(/^[-*]\\s+/, ""));
            i += 1;
          }
          i -= 1;
          out.push("<ul>" + items.map(item => `<li>${inlineMarkdown(item)}</li>`).join("") + "</ul>");
          continue;
        }
        if (/^\\d+\\.\\s+/.test(trimmed)) {
          const items = [];
          while (i < lines.length && /^\\d+\\.\\s+/.test(lines[i].trim())) {
            items.push(lines[i].trim().replace(/^\\d+\\.\\s+/, ""));
            i += 1;
          }
          i -= 1;
          out.push("<ol>" + items.map(item => `<li>${inlineMarkdown(item)}</li>`).join("") + "</ol>");
          continue;
        }
        out.push(`<p>${inlineMarkdown(trimmed)}</p>`);
      }
      return out.join("");
    }
    async function refresh() {
      const res = await fetch(`${BASE}/api/status`, { cache: "no-store" });
      const data = await res.json();
      const runtime = data.runtime || {};
      const config = runtime.scheduler_config || {};
      document.getElementById("agent-count").textContent = text(data.runtime.agent_count);
      document.getElementById("tick-count").textContent = text(runtime.tick_count);
      document.getElementById("interval").textContent = `${runtime.tick_seconds}s`;
      document.getElementById("trading").textContent = runtime.live_trading_enabled ? "開啟" : "關閉";
      document.getElementById("fallback-state").textContent = runtime.fallback_active ? "作用中" : "正常";
      document.getElementById("fallback-state").className = runtime.fallback_active ? "warn" : "ok";
      setInputUnlessFocused("interval-input", config.tick_seconds ?? runtime.tick_seconds);
      setInputUnlessFocused("fallback-input", config.fallback_tick_seconds ?? runtime.fallback_tick_seconds);
      const summary = data.arena_summary || {};
      document.getElementById("arena-summary").textContent = summary.summary
        ? ""
        : "尚無摘要。";
      if (summary.summary) {
        document.getElementById("arena-summary").innerHTML = `${markdownToHtml(summary.summary)}<p class="muted">更新時間：${twTime(summary.ts)}</p>`;
      }

      const limit = data.last_limit_event || {};
      const alert = document.getElementById("limit-alert");
      if (limit.ts) {
        alert.hidden = false;
        document.getElementById("limit-time").textContent = twTime(limit.ts);
        document.getElementById("limit-model").textContent = text(limit.model);
        document.getElementById("limit-retry").textContent = limit.retry_after_seconds ? `${limit.retry_after_seconds}s` : "-";
        document.getElementById("limit-fallback").textContent = `${limit.fallback_tick_seconds || runtime.fallback_tick_seconds}s`;
        document.getElementById("limit-message").textContent = text(limit.message);
      } else {
        alert.hidden = true;
      }

      const paperDiagnostics = data.paper_diagnostics || {};
      const statusCounts = Object.fromEntries((paperDiagnostics.statuses || []).map(item => [item.status, item.count]));
      document.getElementById("ready-count").textContent = text(statusCounts.paper_trade_ready || 0);
      document.getElementById("research-count").textContent = text(statusCounts.research_only || 0);
      document.getElementById("blocked-count").textContent = text(statusCounts.blocked || 0);
      const borrowSnapshot = (data.last_market_snapshot || {}).borrow_snapshot || {};
      document.getElementById("borrow-source").textContent = borrowSnapshot.configured ? "已連線" : "未設定";
      document.getElementById("borrow-source").className = borrowSnapshot.configured ? "metric ok" : "metric warn";
      document.getElementById("blockers").innerHTML = (paperDiagnostics.top_blockers || []).map(item =>
        `<tr><td>${html(item.blocker)}</td><td>${html(item.count)}</td></tr>`
      ).join("");

      document.getElementById("llm-diagnostics").innerHTML = (data.llm_diagnostics || []).map(item => {
        const repairText = `${item.format_repairs_completed || 0}/${item.format_repairs_started || 0}`;
        return `<tr><td><code>${html(item.model || item.agent_id)}</code></td><td>${html(item.calls)}</td><td>${((item.ok_rate || 0) * 100).toFixed(1)}%</td><td>${html(item.timeouts || 0)}</td><td>${html(repairText)}</td><td>${html(item.format_repairs_failed || 0)}</td><td>${Number(item.avg_latency_ms || 0).toFixed(0)} ms</td><td>${Number(item.avg_prompt_tokens || 0).toFixed(0)}</td></tr>`;
      }).join("");

      document.getElementById("agents").innerHTML = (data.agents || []).map(agent => {
        const payload = agent.payload_json || {};
        const decision = payload.decision || {};
        return `<article class="agent">
          <h3>${html(agent.name)}</h3>
          <div class="row"><span class="muted">模型</span><code>${html(agent.model)}</code></div>
          <div class="row"><span class="muted">狀態</span><strong class="${statusClass(agent.status)}">${html(agentStatusLabel(agent.status))}</strong></div>
          <div class="row"><span class="muted">分數</span><strong>${Number(agent.score || 0).toFixed(3)}</strong></div>
          <div class="row"><span class="muted">上次執行</span><span>${twTime(agent.last_tick_at)}</span></div>
          <p class="summary">${html(agent.last_summary || decision.summary)}</p>
        </article>`;
      }).join("");

      document.getElementById("opportunities").innerHTML = (data.opportunities || []).map(item =>
        `<tr><td>${twTime(item.ts)}</td><td>${html(opportunityKindLabel(item.kind))}</td><td><code>${html(item.symbol)}</code></td><td>${html(item.title)}</td><td>${Number(item.gross_edge_bps || 0).toFixed(3)}</td><td>${Number(item.estimated_cost_bps || 0).toFixed(3)}</td><td>${Number(item.net_edge_bps || 0).toFixed(3)}</td><td>${html(opportunityStatusLabel(item.status))}</td></tr>`
      ).join("");

      document.getElementById("paper-signals").innerHTML = (data.paper_signals || []).map(item =>
        `<tr><td>${twTime(item.ts)}</td><td>${html(opportunityKindLabel(item.kind))}</td><td><code>${html(item.symbol)}</code></td><td>${html(item.title)}</td><td>${Number(item.notional_usdt || 0).toFixed(2)}</td><td>${Number(item.gross_edge_bps || 0).toFixed(3)}</td><td>${Number(item.estimated_cost_bps || 0).toFixed(3)}</td><td>${Number(item.net_edge_bps || 0).toFixed(3)}</td><td>${html(paperSignalStatusLabel(item.status))}</td><td>${html(blockersText(item.blockers_json))}</td></tr>`
      ).join("");

      document.getElementById("strategies").innerHTML = (data.strategies || []).map(item =>
        `<tr><td>${twTime(item.ts)}</td><td><code>${html(item.agent_id)}</code></td><td>${html(item.title)}</td><td>${html(item.expected_edge_bps)}</td><td>${html(item.risk_score)}</td></tr>`
      ).join("");

      document.getElementById("events").innerHTML = (data.events || []).map(item =>
        `<tr><td>${twTime(item.ts)}</td><td><code>${html(item.agent_id)}</code></td><td>${html(eventKindLabel(item.kind))}</td><td>${html(item.message)}</td></tr>`
      ).join("");
    }
    document.getElementById("tick").addEventListener("click", async () => {
      await fetch(`${BASE}/api/tick`, { method: "POST" });
      setTimeout(refresh, 1000);
    });
    document.getElementById("ask-root").addEventListener("click", async () => {
      const button = document.getElementById("ask-root");
      const answer = document.getElementById("root-answer");
      const message = document.getElementById("root-question").value.trim();
      if (!message) {
        answer.textContent = "請先輸入問題。";
        return;
      }
      button.disabled = true;
      answer.textContent = "Root Agent 思考中...";
      try {
        const res = await fetch(`${BASE}/api/root-chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message })
        });
        const data = await res.json();
        answer.innerHTML = `${markdownToHtml(data.answer || "沒有收到回覆。")}<p class="muted">回覆時間：${twTime(data.ts)}${data.model ? `，模型：${html(data.model)}` : ""}</p>`;
        refresh();
      } catch (error) {
        answer.textContent = `Root Agent 暫時無法回覆：${error}`;
      } finally {
        button.disabled = false;
      }
    });
    document.getElementById("save-config").addEventListener("click", async () => {
      const tickSeconds = Number(document.getElementById("interval-input").value);
      const fallbackSeconds = Number(document.getElementById("fallback-input").value);
      const status = document.getElementById("config-status");
      status.textContent = "儲存中...";
      const res = await fetch(`${BASE}/api/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tick_seconds: tickSeconds, fallback_tick_seconds: fallbackSeconds })
      });
      if (!res.ok) {
        status.textContent = "儲存失敗。數值必須介於 60 到 86400 秒。";
        status.className = "error";
        return;
      }
      const data = await res.json();
      status.className = "ok";
      status.textContent = `已儲存。執行間隔 ${data.scheduler_config.tick_seconds}s，fallback ${data.scheduler_config.fallback_tick_seconds}s。`;
      refresh();
    });
    refresh();
    setInterval(refresh, 15000);
  </script>
</body>
</html>"""
