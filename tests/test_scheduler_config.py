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
