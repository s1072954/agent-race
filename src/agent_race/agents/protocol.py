from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, Field


T = TypeVar("T", bound=BaseModel)


class SubAgentTask(BaseModel):
    role: str = Field(description="researcher, strategist, backtester, or risk_officer")
    objective: str
    expected_output: str


class StrategyCandidate(BaseModel):
    title: str
    hypothesis: str
    market: str = "crypto"
    expected_edge_bps: float = 0
    risk_score: float = Field(default=5, ge=0, le=10)
    validation_plan: str = ""


class RootAgentDecision(BaseModel):
    summary: str
    observations: list[str] = Field(default_factory=list)
    subagent_tasks: list[SubAgentTask] = Field(default_factory=list)
    strategy_candidates: list[StrategyCandidate] = Field(default_factory=list)
    risk_notes: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)
    next_tick_seconds: int = Field(default=900, ge=60)


class SubAgentResult(BaseModel):
    role: str
    summary: str
    findings: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0, le=1)


def parse_json_model(text: str, model_type: type[T]) -> T:
    try:
        return model_type.model_validate_json(text)
    except Exception:
        pass
    extracted = extract_json_object(text)
    return model_type.model_validate(json.loads(extracted))


def extract_json_object(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response")
    return text[start : end + 1]


def fallback_decision(reason: str) -> RootAgentDecision:
    return RootAgentDecision(
        summary=f"Fallback cycle used because the model response could not be used: {reason}",
        observations=["No valid model output was available for this cycle."],
        subagent_tasks=[
            SubAgentTask(
                role="researcher",
                objective="Inspect current market spreads and identify whether data quality is sufficient.",
                expected_output="Short data-quality note with any obvious arbitrage candidates.",
            )
        ],
        strategy_candidates=[],
        risk_notes=["No strategy should be promoted from an invalid or missing response."],
        next_actions=["Retry on the next scheduled tick with the same model budget."],
        confidence=0.1,
    )
