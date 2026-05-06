import asyncio

from agent_race.tools import execution
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


def test_negative_funding_can_be_paper_ready_with_borrow_data(monkeypatch) -> None:
    async def fake_spot_book(client, exchange, symbol):
        return {"bids": [[100, 10]], "asks": [[100, 10]]}

    async def fake_perp_book(client, symbol):
        return {"bids": [[100, 10]], "asks": [[100, 10]]}

    monkeypatch.setattr(execution, "_fetch_spot_book", fake_spot_book)
    monkeypatch.setattr(execution, "_fetch_binance_perp_book", fake_perp_book)

    signals = asyncio.run(
        execution.validate_opportunities(
            [
                {
                    "kind": "funding_rate",
                    "symbol": "ABCUSDT",
                    "title": "ABC funding",
                    "evidence": {"base": "ABC", "funding_bps": -30},
                }
            ],
            borrow_snapshot={
                "configured": True,
                "assets": {
                    "ABC": {
                        "status": "ok",
                        "available_inventory_amount": 5,
                        "borrow_cost_bps_per_funding": 2,
                    }
                },
            },
        )
    )

    assert signals[0]["status"] == "paper_trade_ready"
    assert signals[0]["estimated_cost_bps"] == 17
    assert signals[0]["validation"]["borrow"]["status"] == "ok"
