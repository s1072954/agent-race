from agent_race.memory.store import AgentRaceStore


def test_limit_event_applies_fallback_interval(tmp_path) -> None:
    store = AgentRaceStore(tmp_path / "race.sqlite", tmp_path / "agents")
    store.update_scheduler_config(tick_seconds=120, fallback_tick_seconds=900, updated_by="test")

    event = store.record_limit_fallback(
        agent_id="agent-a",
        model="model-a",
        message="rate limit",
        retry_after_seconds=30,
        default_tick_seconds=120,
        default_fallback_tick_seconds=900,
    )

    config = store.scheduler_config(default_tick_seconds=120, default_fallback_tick_seconds=900)
    assert config["tick_seconds"] == 900
    assert config["fallback_active"] is True
    assert event["fallback_tick_seconds"] == 900
    assert store.get_state("last_limit_event")["model"] == "model-a"


def test_records_market_opportunities(tmp_path) -> None:
    store = AgentRaceStore(tmp_path / "race.sqlite", tmp_path / "agents")
    store.record_opportunities(
        [
            {
                "kind": "funding_rate",
                "symbol": "BTCUSDT",
                "title": "BTC funding candidate",
                "gross_edge_bps": 8,
                "estimated_cost_bps": 2.5,
                "net_edge_bps": 5.5,
                "confidence": 0.42,
                "status": "actionable_research",
            }
        ]
    )

    opportunities = store.recent_opportunities()
    assert len(opportunities) == 1
    assert opportunities[0]["symbol"] == "BTCUSDT"
    assert opportunities[0]["payload_json"]["status"] == "actionable_research"


def test_records_paper_signals(tmp_path) -> None:
    store = AgentRaceStore(tmp_path / "race.sqlite", tmp_path / "agents")
    store.record_paper_signals(
        [
            {
                "kind": "spot_spread",
                "symbol": "ETHUSDT",
                "title": "ETH spread",
                "notional_usdt": 100,
                "gross_edge_bps": 12,
                "estimated_cost_bps": 8,
                "net_edge_bps": 4,
                "status": "watch",
                "blockers": ["small edge"],
            }
        ]
    )

    signals = store.recent_paper_signals()
    assert len(signals) == 1
    assert signals[0]["symbol"] == "ETHUSDT"
    assert signals[0]["blockers_json"] == ["small edge"]
