from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

from agent_race.agents.protocol import (
    RootAgentDecision,
    SubAgentResult,
    fallback_decision,
    parse_json_model,
)
from agent_race.config import Settings
from agent_race.llm.client import NvidiaChatClient
from agent_race.memory.store import AgentRaceStore, utc_now


T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class AgentSpec:
    id: str
    name: str
    model: str

    @classmethod
    def from_model(cls, model: str) -> "AgentSpec":
        slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
        return cls(id=slug, name=model.split("/")[-1], model=model)


class RootAgent:
    def __init__(
        self,
        spec: AgentSpec,
        settings: Settings,
        store: AgentRaceStore,
        llm: NvidiaChatClient,
    ) -> None:
        self.spec = spec
        self.settings = settings
        self.store = store
        self.llm = llm
        self.workspace = self.store.upsert_agent(spec.id, spec.name, spec.model)

    async def run_tick(self, market_snapshot: dict[str, Any]) -> RootAgentDecision:
        self.store.record_event("tick_started", "Root agent tick started", agent_id=self.spec.id)
        root_prompt = _read_prompt(self.settings.prompt_dir / "root_agent.md")
        recent_events = self.store.recent_events(self.spec.id, limit=8)
        memory_note = self._read_memory_note()
        agent_market_snapshot = _compact_market_snapshot(market_snapshot)
        user_prompt = json.dumps(
            {
                "agent": self.spec.__dict__,
                "utc_now": utc_now(),
                "market_snapshot": agent_market_snapshot,
                "recent_events": recent_events,
                "memory_note": memory_note[-4000:],
                "required_json_schema": RootAgentDecision.model_json_schema(),
            },
            ensure_ascii=False,
        )

        try:
            result = await self.llm.chat(
                agent_id=self.spec.id,
                model=self.spec.model,
                messages=[
                    {"role": "system", "content": root_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1200,
                temperature=0.25,
            )
            decision = await self._parse_or_repair(
                content=result.content,
                model_type=RootAgentDecision,
                context="root_agent",
                max_tokens=1200,
            )
            status = "ok"
        except Exception as exc:  # noqa: BLE001
            decision = fallback_decision(str(exc))
            status = "fallback"
            self.store.record_event("root_agent_error", str(exc), agent_id=self.spec.id)

        subagent_results = await self._run_subagents(decision, market_snapshot)
        payload = {
            "decision": decision.model_dump(),
            "subagent_results": [item.model_dump() for item in subagent_results],
            "market_snapshot_ts": market_snapshot.get("ts"),
        }
        score_delta = self._score_delta(decision, subagent_results, market_snapshot)
        self.store.update_agent_status(
            agent_id=self.spec.id,
            status=status,
            summary=decision.summary,
            score_delta=score_delta,
            payload=payload,
        )
        for strategy in decision.strategy_candidates:
            self.store.record_strategy(
                agent_id=self.spec.id,
                title=strategy.title,
                hypothesis=strategy.hypothesis,
                expected_edge_bps=strategy.expected_edge_bps,
                risk_score=strategy.risk_score,
                payload=strategy.model_dump(),
            )
        self._write_memory_note(decision, subagent_results)
        self.store.record_event(
            "tick_completed",
            f"Tick completed with {len(decision.strategy_candidates)} strategies",
            agent_id=self.spec.id,
            payload={"score_delta": score_delta, "status": status},
        )
        return decision

    async def _run_subagents(
        self, decision: RootAgentDecision, market_snapshot: dict[str, Any]
    ) -> list[SubAgentResult]:
        if self.settings.max_subagent_tasks <= 0:
            return []
        results: list[SubAgentResult] = []
        for task in decision.subagent_tasks[: self.settings.max_subagent_tasks]:
            prompt_file = self.settings.prompt_dir / "subagents" / f"{task.role}.md"
            system_prompt = _read_prompt(prompt_file, default=_read_prompt(self.settings.prompt_dir / "subagents/researcher.md"))
            user_prompt = json.dumps(
                {
                    "agent": self.spec.__dict__,
                    "task": task.model_dump(),
                    "market_snapshot": _compact_market_snapshot(market_snapshot),
                    "required_json_schema": SubAgentResult.model_json_schema(),
                },
                ensure_ascii=False,
            )
            try:
                result = await self.llm.chat(
                    agent_id=self.spec.id,
                    model=self.spec.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=700,
                    temperature=0.2,
                )
                parsed = await self._parse_or_repair(
                    content=result.content,
                    model_type=SubAgentResult,
                    context="subagent",
                    max_tokens=700,
                )
            except Exception as exc:  # noqa: BLE001
                parsed = SubAgentResult(
                    role=task.role,
                    summary=f"Sub-agent fallback: {exc}",
                    findings=[],
                    artifacts=[],
                    confidence=0.1,
                )
                self.store.record_event("subagent_error", str(exc), agent_id=self.spec.id)
            results.append(parsed)
            self.store.record_event(
                "subagent_completed",
                parsed.summary,
                agent_id=self.spec.id,
                payload=parsed.model_dump(),
            )
        return results

    async def _parse_or_repair(
        self,
        *,
        content: str,
        model_type: type[T],
        context: str,
        max_tokens: int,
    ) -> T:
        try:
            return parse_json_model(content, model_type)
        except Exception as first_exc:  # noqa: BLE001
            self.store.record_event(
                f"{context}_format_repair_started",
                str(first_exc),
                agent_id=self.spec.id,
                payload={"response_preview": content[:1000]},
            )

        repair_prompt = json.dumps(
            {
                "instruction": (
                    "Repair the invalid model output into exactly one valid JSON object. "
                    "Preserve the intent when possible. If a field is missing, use an empty list, "
                    "empty string, or conservative numeric value that matches the schema. "
                    "Return JSON only."
                ),
                "required_json_schema": model_type.model_json_schema(),
                "invalid_output": content[:6000],
            },
            ensure_ascii=False,
        )
        try:
            repaired = await self.llm.chat(
                agent_id=self.spec.id,
                model=self.spec.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict JSON repair tool. Return exactly one valid JSON object. "
                            "No Markdown, no code fence, no prose."
                        ),
                    },
                    {"role": "user", "content": repair_prompt},
                ],
                max_tokens=max_tokens,
                temperature=0,
                retries=0,
            )
            parsed = parse_json_model(repaired.content, model_type)
            self.store.record_event(
                f"{context}_format_repair_completed",
                "Invalid JSON output was repaired successfully",
                agent_id=self.spec.id,
            )
            return parsed
        except Exception as repair_exc:  # noqa: BLE001
            self.store.record_event(
                f"{context}_format_repair_failed",
                str(repair_exc),
                agent_id=self.spec.id,
            )
            raise ValueError(f"JSON parse failed and repair failed: {repair_exc}") from repair_exc

    def _score_delta(
        self,
        decision: RootAgentDecision,
        subagent_results: list[SubAgentResult],
        market_snapshot: dict[str, Any],
    ) -> float:
        paper_signals = market_snapshot.get("paper_signals", [])
        ready_edges = [
            float(item.get("net_edge_bps") or 0)
            for item in paper_signals
            if item.get("status") == "paper_trade_ready"
        ]
        watch_edges = [
            float(item.get("net_edge_bps") or 0)
            for item in paper_signals
            if item.get("status") == "watch"
        ]
        has_executable_evidence = bool(ready_edges)
        has_near_executable_evidence = bool(watch_edges)
        if not decision.strategy_candidates:
            no_trade_bonus = 0.035 if not has_executable_evidence else 0.005
            return round(no_trade_bonus * decision.confidence, 4)
        avg_edge = sum(item.expected_edge_bps for item in decision.strategy_candidates) / len(decision.strategy_candidates)
        avg_risk = sum(item.risk_score for item in decision.strategy_candidates) / len(decision.strategy_candidates)
        subagent_bonus = sum(item.confidence for item in subagent_results) * 0.03
        if has_executable_evidence:
            evidence_bonus = max(ready_edges) * 0.04
        elif has_near_executable_evidence:
            evidence_bonus = max(watch_edges) * 0.01
        else:
            evidence_bonus = 0.0
        score = avg_edge / 150 - avg_risk * 0.04 + decision.confidence * 0.06 + subagent_bonus + evidence_bonus
        if not has_executable_evidence:
            score *= 0.25 if has_near_executable_evidence else 0.15
        return round(max(0.0, score), 4)

    def _read_memory_note(self) -> str:
        path = self.workspace / "memory.md"
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def _write_memory_note(self, decision: RootAgentDecision, subagent_results: list[SubAgentResult]) -> None:
        lines = [
            f"# {self.spec.name}",
            "",
            f"Last updated: {utc_now()}",
            "",
            "## Latest Summary",
            decision.summary,
            "",
            "## Next Actions",
            *[f"- {item}" for item in decision.next_actions],
            "",
            "## Risk Notes",
            *[f"- {item}" for item in decision.risk_notes],
            "",
            "## Sub-Agent Results",
            *[f"- {item.role}: {item.summary}" for item in subagent_results],
            "",
        ]
        (self.workspace / "memory.md").write_text("\n".join(lines), encoding="utf-8")


def _read_prompt(path: Path, default: str = "") -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return default


def _compact_market_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "ts": snapshot.get("ts"),
        "data_quality": snapshot.get("data_quality", [])[:6],
        "notes": snapshot.get("notes", [])[:8],
        "opportunities": [_compact_item(item) for item in snapshot.get("opportunities", [])[:12]],
        "paper_signals": [_compact_item(item) for item in snapshot.get("paper_signals", [])[:12]],
        "spreads": [_compact_item(item) for item in snapshot.get("spreads", [])[:12]],
        "funding_rates": [_compact_item(item) for item in snapshot.get("funding_rates", [])[:12]],
        "borrow_snapshot": _compact_borrow_snapshot(snapshot.get("borrow_snapshot", {})),
    }


def _compact_item(item: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "kind",
        "symbol",
        "title",
        "base",
        "exchange",
        "left_exchange",
        "right_exchange",
        "lower_exchange",
        "higher_exchange",
        "lower_price",
        "higher_price",
        "spread_bps",
        "quote_mismatch",
        "min_quote_volume_usd",
        "gross_edge_bps",
        "estimated_cost_bps",
        "net_edge_bps",
        "confidence",
        "status",
        "next_validation",
        "notional_usdt",
        "validation",
        "blockers",
        "last_funding_rate",
        "funding_bps",
        "annualized_percent",
        "next_funding_time",
    }
    compact = {key: value for key, value in item.items() if key in allowed}
    evidence = item.get("evidence")
    if isinstance(evidence, dict):
        compact["evidence"] = {key: value for key, value in evidence.items() if key in allowed}
    return compact


def _compact_borrow_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    if not snapshot:
        return {}
    assets = snapshot.get("assets") or {}
    return {
        "ts": snapshot.get("ts"),
        "provider": snapshot.get("provider"),
        "configured": snapshot.get("configured", False),
        "assets_requested": snapshot.get("assets_requested", []),
        "notes": snapshot.get("notes", [])[:4],
        "errors": snapshot.get("errors", [])[:4],
        "assets": {
            asset: {
                "status": item.get("status"),
                "available_inventory_amount": item.get("available_inventory_amount"),
                "daily_interest_bps": item.get("daily_interest_bps"),
                "borrow_cost_bps_per_funding": item.get("borrow_cost_bps_per_funding"),
                "max_borrowable_amount": item.get("max_borrowable_amount"),
            }
            for asset, item in list(assets.items())[:12]
            if isinstance(item, dict)
        },
    }
