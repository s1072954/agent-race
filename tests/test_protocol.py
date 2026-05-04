from agent_race.agents.protocol import RootAgentDecision, parse_json_model


def test_parse_json_model_extracts_wrapped_json() -> None:
    parsed = parse_json_model(
        'noise {"summary":"ok","confidence":0.7,"next_tick_seconds":900} trailing',
        RootAgentDecision,
    )
    assert parsed.summary == "ok"
    assert parsed.confidence == 0.7
