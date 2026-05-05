from agent_race.tools.execution import simulate_buy_with_quote, simulate_sell_base


def test_simulate_buy_with_quote_walks_multiple_levels() -> None:
    result = simulate_buy_with_quote([[10, 5], [11, 10]], 105)

    assert result["filled"] is True
    assert round(result["base_qty"], 6) == round(5 + 55 / 11, 6)
    assert result["quote_amount"] == 105


def test_simulate_sell_base_detects_insufficient_depth() -> None:
    result = simulate_sell_base([[10, 1]], 2)

    assert result["filled"] is False
    assert result["base_qty"] == 1
    assert result["quote_amount"] == 10
