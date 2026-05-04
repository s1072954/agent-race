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
from agent_race.tools import fetch_market_snapshot


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
            self.store.set_state("last_market_snapshot", market_snapshot)
            self.store.record_opportunities(market_snapshot.get("opportunities", []))
            semaphore = asyncio.Semaphore(self.settings.max_parallel_llm_calls)

            async def run_agent(agent: RootAgent) -> None:
                async with semaphore:
                    await agent.run_tick(market_snapshot)

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
                    "summary": "NVIDIA_API_KEY is not configured; dashboard is showing deterministic scheduler state.",
                    "model": None,
                },
            )
            return
        prompt = json.dumps(
            {
                "instruction": "Summarize the current LLM agent race in concise Traditional Chinese. Focus on each agent status, best strategy candidates, risk, and next operational concern.",
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
                        "content": "You are the monitor for a crypto arbitrage LLM agent arena. Be concise, factual, and risk-aware.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=700,
                temperature=0.15,
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
            },
            "llm_usage_totals": overview["llm_usage"]["totals"],
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
        }

    def _scheduler_config(self) -> dict[str, Any]:
        return self.store.scheduler_config(
            self.settings.tick_seconds,
            self.settings.fallback_tick_seconds,
        )
