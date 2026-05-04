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

    async def run_once(self) -> None:
        async with self._running_once:
            self._tick_count += 1
            self.store.record_event("cycle_started", "Agent race cycle started")
            market_snapshot = await fetch_market_snapshot()
            self.store.set_state("last_market_snapshot", market_snapshot)
            semaphore = asyncio.Semaphore(self.settings.max_parallel_llm_calls)

            async def run_agent(agent: RootAgent) -> None:
                async with semaphore:
                    await agent.run_tick(market_snapshot)

            await asyncio.gather(*(run_agent(agent) for agent in self.agents))
            if self._tick_count % self.settings.summary_every_ticks == 0:
                await self._summarize_arena()
            self.store.record_event("cycle_completed", "Agent race cycle completed")

    async def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                await self.run_once()
            except Exception as exc:  # noqa: BLE001
                self.store.record_event("scheduler_error", str(exc))
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.settings.tick_seconds)
            except asyncio.TimeoutError:
                continue

    async def _summarize_arena(self) -> None:
        overview = self.store.overview()
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

    def runtime_status(self) -> dict[str, Any]:
        return {
            "scheduler_enabled": self.settings.scheduler_enabled,
            "running": self._task is not None and not self._task.done(),
            "tick_seconds": self.settings.tick_seconds,
            "tick_count": self._tick_count,
            "agent_count": len(self.agents),
            "models": [agent.spec.model for agent in self.agents],
            "live_trading_enabled": self.settings.live_trading_enabled,
            "shell_tools_enabled": self.settings.shell_tools_enabled,
        }
