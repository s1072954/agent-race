from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from typing import Any

from agent_race.agents import AgentSpec, RootAgent
from agent_race.config import Settings
from agent_race.llm import AsyncSlidingWindowLimiter, NvidiaChatClient
from agent_race.memory import AgentRaceStore
from agent_race.memory.store import utc_now
from agent_race.tools import fetch_borrow_snapshot, fetch_market_snapshot, validate_opportunities


class AgentRaceScheduler:
    def __init__(self, settings: Settings, store: AgentRaceStore) -> None:
        self.settings = settings
        self.store = store
        self.limiter = AsyncSlidingWindowLimiter(settings.nvidia_global_rpm, settings.nvidia_model_rpm)
        self.llm = NvidiaChatClient(settings, self.limiter, store)
        self.agents = [
            RootAgent(AgentSpec.from_model(model), settings, store, self.llm)
            for model in settings.nvidia_models
        ]
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()
        self._schedule_changed = asyncio.Event()
        self._tick_count = 0
        self._running_once = asyncio.Lock()

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task

    async def trigger_once(self) -> bool:
        if self._running_once.locked():
            return False
        asyncio.create_task(self.run_once())
        return True

    def update_config(self, tick_seconds: int, fallback_tick_seconds: int) -> dict[str, Any]:
        config = self.store.update_scheduler_config(
            tick_seconds=tick_seconds,
            fallback_tick_seconds=fallback_tick_seconds,
            updated_by="dashboard",
            fallback_active=False,
        )
        self._schedule_changed.set()
        return config

    async def run_once(self) -> None:
        async with self._running_once:
            self._tick_count += 1
            self.store.record_event("cycle_started", "Agent race cycle started")
            market_snapshot = await fetch_market_snapshot()
            borrow_snapshot = await fetch_borrow_snapshot(self.settings, market_snapshot.get("opportunities", []))
            paper_signals = await validate_opportunities(
                market_snapshot.get("opportunities", []),
                borrow_snapshot=borrow_snapshot,
            )
            market_snapshot["borrow_snapshot"] = borrow_snapshot
            market_snapshot["paper_signals"] = paper_signals
            self.store.set_state("last_market_snapshot", market_snapshot)
            self.store.record_opportunities(market_snapshot.get("opportunities", []))
            self.store.record_paper_signals(paper_signals)
            semaphore = asyncio.Semaphore(self.settings.max_parallel_llm_calls)
            allow_subagents = self._tick_count % self.settings.subagent_every_ticks == 0

            async def run_agent(agent: RootAgent) -> None:
                async with semaphore:
                    await agent.run_tick(market_snapshot, allow_subagents=allow_subagents)

            await asyncio.gather(*(run_agent(agent) for agent in self.agents))
            if self._tick_count % self.settings.summary_every_ticks == 0:
                await self._summarize_arena()
            self.store.record_event("cycle_completed", "Agent race cycle completed")

    async def _loop(self) -> None:
        next_run_at = asyncio.get_running_loop().time()
        while not self._stop.is_set():
            wait_seconds = max(0.0, next_run_at - asyncio.get_running_loop().time())
            wait_result = await self._wait_for_schedule(wait_seconds)
            if wait_result == "stop":
                break
            if wait_result == "config":
                interval = self._scheduler_config()["tick_seconds"]
                next_run_at = asyncio.get_running_loop().time() + interval
                continue
            try:
                await self.run_once()
            except Exception as exc:  # noqa: BLE001
                self.store.record_event("scheduler_error", str(exc))
            next_run_at = asyncio.get_running_loop().time() + self._scheduler_config()["tick_seconds"]

    async def _wait_for_schedule(self, wait_seconds: float) -> str:
        if wait_seconds <= 0:
            return "timeout"
        stop_task = asyncio.create_task(self._stop.wait())
        changed_task = asyncio.create_task(self._schedule_changed.wait())
        done, pending = await asyncio.wait(
            {stop_task, changed_task},
            timeout=wait_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        with suppress(asyncio.CancelledError):
            await asyncio.gather(*pending)
        if stop_task in done:
            return "stop"
        if changed_task in done:
            self._schedule_changed.clear()
            return "config"
        return "timeout"

    async def _summarize_arena(self) -> None:
        overview = self._compact_overview()
        if not self.settings.can_call_llm:
            self.store.set_state(
                "arena_summary",
                {
                    "ts": utc_now(),
                    "summary": "NVIDIA_API_KEY 未設定；dashboard 目前只顯示排程與市場資料狀態。",
                    "model": None,
                },
            )
            return
        prompt = json.dumps(
            {
                "instruction": (
                    "請用繁體中文摘要目前的加密貨幣套利 LLM 競賽狀態。"
                    "所有標題、狀態、風險、下一步都必須使用繁體中文；"
                    "交易對、模型名稱、交易所名稱、bps、API 等專有名詞可以保留英文。"
                    "不要輸出英文段落。請保持精簡、具體、風險意識清楚。"
                    "內容需涵蓋：Agent 狀態、主要策略候選、是否可執行、主要阻礙、下一個操作重點。"
                ),
                "overview": overview,
            },
            ensure_ascii=False,
        )
        try:
            result = await self.llm.chat(
                agent_id="arena-summary",
                model=self.settings.nvidia_summary_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是加密貨幣套利 LLM 競賽場的監控摘要員。"
                            "你必須只用繁體中文輸出；不得使用英文句子。"
                            "可保留交易對、模型名、交易所名與技術縮寫。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=700,
                temperature=0.15,
                retries=0,
            )
            summary = result.content.strip()
        except Exception as exc:  # noqa: BLE001
            summary = f"Summary LLM unavailable: {exc}"
        self.store.set_state(
            "arena_summary",
            {
                "ts": utc_now(),
                "summary": summary,
                "model": self.settings.nvidia_summary_model,
            },
        )

    async def ask_root_agent(self, question: str) -> dict[str, Any]:
        cleaned = question.strip()
        now = utc_now()
        if not cleaned:
            return {"ok": False, "ts": now, "answer": "請先輸入問題。", "model": None}
        if not self.agents:
            return {"ok": False, "ts": now, "answer": "目前沒有可用的 Root Agent。", "model": None}

        agent = self.agents[0]
        if self._running_once.locked():
            answer = (
                "Root Agent 目前正在執行本輪分析，為了避免額外 LLM 呼叫造成 timeout 或限流，"
                "這次先不插隊呼叫模型。你可以先查看監控摘要、事件與策略候選；本輪完成後再發問。"
            )
            return {"ok": False, "ts": now, "answer": answer, "model": agent.spec.model, "busy": True}
        if not self.settings.can_call_llm:
            return {"ok": False, "ts": now, "answer": "NVIDIA_API_KEY 未設定，無法直接詢問 Root Agent。", "model": None}

        overview = self._compact_overview()
        memory_path = agent.workspace / "memory.md"
        memory_note = memory_path.read_text(encoding="utf-8")[-5000:] if memory_path.exists() else ""
        prompt = json.dumps(
            {
                "instruction": (
                    "你是目前這個加密貨幣套利系統的 Root Agent。"
                    "請用繁體中文回答使用者問題，務必根據提供的狀態、記憶與市場資料。"
                    "不可建議實盤下單，不可聲稱已經下單。"
                    "如果策略不可執行，請明確說出阻礙與下一步驗證。"
                ),
                "user_question": cleaned,
                "utc_now": now,
                "agent": agent.spec.__dict__,
                "runtime": self.runtime_status(),
                "overview": overview,
                "memory_note": memory_note,
            },
            ensure_ascii=False,
        )
        try:
            result = await self.llm.chat(
                agent_id=agent.spec.id,
                model=agent.spec.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是 Root Agent 的對話介面。只用繁體中文回答。"
                            "保持務實、具體、風險優先；不要輸出 JSON；不要建議實盤下單。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=700,
                temperature=0.2,
                retries=0,
            )
            answer = result.content.strip()
            payload = {"question": cleaned, "answer": answer, "model": agent.spec.model, "ts": now}
            self.store.set_state("last_root_chat", payload)
            self.store.record_event("root_chat_completed", cleaned[:160], agent_id=agent.spec.id, payload=payload)
            return {"ok": True, **payload}
        except Exception as exc:  # noqa: BLE001
            answer = f"Root Agent 暫時無法回覆：{exc}"
            payload = {"question": cleaned, "answer": answer, "model": agent.spec.model, "ts": now}
            self.store.set_state("last_root_chat", payload)
            self.store.record_event("root_chat_error", str(exc), agent_id=agent.spec.id, payload=payload)
            return {"ok": False, **payload}

    def _compact_overview(self) -> dict[str, Any]:
        overview = self.store.overview()
        market = overview.get("last_market_snapshot", {})
        return {
            "agents": [
                {
                    "id": item["id"],
                    "name": item["name"],
                    "model": item["model"],
                    "status": item["status"],
                    "score": round(float(item["score"]), 4),
                    "last_tick_at": item["last_tick_at"],
                    "last_summary": (item["last_summary"] or "")[:500],
                }
                for item in overview["agents"]
            ],
            "top_market_opportunities": overview.get("opportunities", [])[:12],
            "top_paper_signals": overview.get("paper_signals", [])[:12],
            "recent_strategy_candidates": [
                {
                    "ts": item["ts"],
                    "agent_id": item["agent_id"],
                    "title": item["title"],
                    "expected_edge_bps": item["expected_edge_bps"],
                    "risk_score": item["risk_score"],
                }
                for item in overview.get("strategies", [])[:10]
            ],
            "market_snapshot": {
                "ts": market.get("ts"),
                "data_quality": market.get("data_quality", []),
                "notes": market.get("notes", [])[:8],
                "top_spreads": market.get("spreads", [])[:10],
                "top_funding_rates": market.get("funding_rates", [])[:10],
                "paper_signals": market.get("paper_signals", [])[:10],
                "borrow_snapshot": market.get("borrow_snapshot", {}),
            },
            "llm_usage_totals": overview["llm_usage"]["totals"],
            "paper_diagnostics": overview.get("paper_diagnostics", {}),
            "llm_diagnostics": overview.get("llm_diagnostics", [])[:8],
            "recent_errors": [
                {
                    "ts": item["ts"],
                    "agent_id": item["agent_id"],
                    "kind": item["kind"],
                    "message": item["message"][:300],
                }
                for item in overview["events"]
                if "error" in item["kind"] or "fallback" in item["kind"]
            ][:8],
        }

    def runtime_status(self) -> dict[str, Any]:
        config = self._scheduler_config()
        return {
            "scheduler_enabled": self.settings.scheduler_enabled,
            "running": self._task is not None and not self._task.done(),
            "tick_seconds": config["tick_seconds"],
            "fallback_tick_seconds": config["fallback_tick_seconds"],
            "fallback_active": config["fallback_active"],
            "scheduler_config": config,
            "tick_count": self._tick_count,
            "agent_count": len(self.agents),
            "models": [agent.spec.model for agent in self.agents],
            "live_trading_enabled": self.settings.live_trading_enabled,
            "shell_tools_enabled": self.settings.shell_tools_enabled,
            "binance_margin_read_enabled": self.settings.can_call_binance_margin_read,
        }

    def _scheduler_config(self) -> dict[str, Any]:
        return self.store.scheduler_config(
            self.settings.tick_seconds,
            self.settings.fallback_tick_seconds,
        )
