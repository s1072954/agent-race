import pytest

from agent_race.agents.protocol import RootAgentDecision, SubAgentResult, parse_json_model
from agent_race.agents.root_agent import _compact_market_snapshot, _merge_memory_backlog


def test_parse_json_model_accepts_markdown_fenced_json() -> None:
    parsed = parse_json_model(
        """```json
{"role":"researcher","summary":"ok","findings":["a"],"artifacts":[],"confidence":0.6}
```""",
        SubAgentResult,
    )

    assert parsed.role == "researcher"
    assert parsed.confidence == 0.6


def test_parse_json_model_repairs_trailing_commas() -> None:
    parsed = parse_json_model(
        '{"summary":"no trade","observations":["fees too high",],"strategy_candidates":[],"risk_notes":[],"next_actions":[],"confidence":0.4,}',
        RootAgentDecision,
    )

    assert parsed.summary == "no trade"
    assert parsed.observations == ["fees too high"]


def test_parse_json_model_closes_truncated_object() -> None:
    parsed = parse_json_model(
        '{"role":"researcher","summary":"borrow data missing","findings":["need API key',
        SubAgentResult,
    )

    assert parsed.role == "researcher"
    assert parsed.summary == "borrow data missing"
    assert parsed.findings == ["need API key"]


def test_parse_json_model_rejects_plain_text_without_json() -> None:
    with pytest.raises(ValueError, match="No JSON object"):
        parse_json_model("I cannot provide JSON for this answer.", SubAgentResult)


def test_compact_market_snapshot_removes_large_sources() -> None:
    snapshot = {
        "ts": "2026-05-05T00:00:00+00:00",
        "sources": {"binance": {f"COIN{i}USDT": {"last": i} for i in range(100)}},
        "opportunities": [{"symbol": f"COIN{i}USDT", "net_edge_bps": i, "evidence": {"funding_bps": i, "unused": "drop"}} for i in range(20)],
        "paper_signals": [],
        "spreads": [],
        "funding_rates": [],
    }

    compact = _compact_market_snapshot(snapshot)

    assert "sources" not in compact
    assert len(compact["opportunities"]) == 12
    assert "unused" not in compact["opportunities"][0]["evidence"]


def test_merge_memory_backlog_preserves_repeated_ideas() -> None:
    decision = RootAgentDecision(
        summary="no trade",
        strategy_candidates=[
            {
                "title": "Stablecoin basis monitor",
                "hypothesis": "Track USDT/USD route dislocations across venues.",
                "expected_edge_bps": 4,
                "risk_score": 3,
                "validation_plan": "Fetch executable bid/ask and transfer constraints.",
            }
        ],
        next_actions=["驗證 OKX 與 Bybit 的借幣庫存"],
    )

    first = _merge_memory_backlog([], decision, "2026-05-07T00:00:00+00:00")
    second = _merge_memory_backlog(first, decision, "2026-05-07T00:03:00+00:00")

    assert len(second) == 2
    assert second[0]["last_seen"] == "2026-05-07T00:03:00+00:00"
    assert any(item["title"] == "Stablecoin basis monitor" and item["sightings"] == 2 for item in second)
    assert any(item["type"] == "next_action" for item in second)


def test_merge_memory_backlog_skips_generic_and_negative_edge_items() -> None:
    decision = RootAgentDecision(
        summary="no trade",
        strategy_candidates=[
            {
                "title": "Negative spot spread",
                "hypothesis": "Fees exceed the edge.",
                "expected_edge_bps": -2,
                "risk_score": 6,
                "validation_plan": "Keep as observation only.",
            }
        ],
        next_actions=["驗證", "檢查 stablecoin basis 的 bid/ask 深度與手續費。"],
    )

    backlog = _merge_memory_backlog([], decision, "2026-05-08T00:00:00+00:00")

    assert all(item["type"] != "strategy" for item in backlog)
    assert len(backlog) == 1
    assert backlog[0]["type"] == "next_action"
