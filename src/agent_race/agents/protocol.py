from __future__ import annotations

import ast
import json
import re
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
    errors: list[str] = []
    for candidate in _candidate_json_objects(text):
        try:
            return model_type.model_validate_json(candidate)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

        parsed = _loads_jsonish(candidate)
        if parsed is None:
            continue
        try:
            return model_type.model_validate(parsed)
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))

    if not errors:
        raise ValueError("No JSON object found in model response")
    raise ValueError(f"Could not parse JSON object: {errors[-1]}")


def extract_json_object(text: str) -> str:
    for candidate in _candidate_json_objects(text):
        return candidate
    raise ValueError("No JSON object found in model response")


def _candidate_json_objects(text: str) -> list[str]:
    candidates: list[str] = []
    for source in _json_sources(text):
        candidates.extend(_balanced_json_objects(source))
    return candidates


def _json_sources(text: str) -> list[str]:
    stripped = text.strip()
    sources = [match.group(1).strip() for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)]
    if stripped:
        sources.append(stripped)
    return sources


def _balanced_json_objects(text: str) -> list[str]:
    results: list[str] = []
    start: int | None = None
    depth = 0
    in_string = False
    escape = False

    for index, char in enumerate(text):
        if start is None:
            if char == "{":
                start = index
                depth = 1
                in_string = False
                escape = False
            continue

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                results.append(text[start : index + 1])
                start = None

    return results


def _loads_jsonish(text: str) -> Any | None:
    attempts = [
        text,
        _remove_trailing_commas(text),
        _remove_trailing_commas(text).translate(str.maketrans({"“": '"', "”": '"'})),
    ]
    for candidate in dict.fromkeys(attempts):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        try:
            parsed = ast.literal_eval(candidate)
        except (SyntaxError, ValueError):
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _remove_trailing_commas(text: str) -> str:
    return re.sub(r",\s*([}\]])", r"\1", text)


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
