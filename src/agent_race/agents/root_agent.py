from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_race.agents.protocol import (
    RootAgentDecision,
    SubAgentResult,
    fallback_decision,
    parse_json_model,
)
from agent_race.config import Settings
from agent_race.llm.client import LLMError, NvidiaChatClient
from agent_race.memory.store import AgentRaceStore, utc_now


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
        user_prompt = json.dumps(
            {
                "agent": self.spec.__dict__,
                "utc_now": utc_now(),
                "market_snapshot": market_snapshot,
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
            decision = parse_json_model(result.content, RootAgentDecision)
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
                    "market_snapshot": market_snapshot,
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
                parsed = parse_json_model(result.content, SubAgentResult)
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

    def _score_delta(
        self,
        decision: RootAgentDecision,
        subagent_results: list[SubAgentResult],
        market_snapshot: dict[str, Any],
    ) -> float:
        opportunities = market_snapshot.get("opportunities", [])
        best_net_edge = max((float(item.get("net_edge_bps") or 0) for item in opportunities), default=0.0)
        has_actionable_evidence = any(item.get("status") == "actionable_research" for item in opportunities)
        if not decision.strategy_candidates:
            no_trade_bonus = 0.035 if not has_actionable_evidence else 0.01
            return round(no_trade_bonus * decision.confidence, 4)
        avg_edge = sum(item.expected_edge_bps for item in decision.strategy_candidates) / len(decision.strategy_candidates)
        avg_risk = sum(item.risk_score for item in decision.strategy_candidates) / len(decision.strategy_candidates)
        subagent_bonus = sum(item.confidence for item in subagent_results) * 0.03
        evidence_bonus = max(0.0, best_net_edge) * 0.015
        score = avg_edge / 120 - avg_risk * 0.03 + decision.confidence * 0.08 + subagent_bonus + evidence_bonus
        if not has_actionable_evidence:
            score *= 0.45
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
