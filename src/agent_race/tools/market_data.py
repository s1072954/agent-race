from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import httpx


SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


async def fetch_market_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "ts": datetime.now(UTC).isoformat(timespec="seconds"),
        "sources": {},
        "notes": [],
    }
    async with httpx.AsyncClient(timeout=12) as client:
        await _fetch_binance(client, snapshot)
        await _fetch_coinbase(client, snapshot)
    snapshot["spreads"] = _estimate_spreads(snapshot)
    return snapshot


async def _fetch_binance(client: httpx.AsyncClient, snapshot: dict[str, Any]) -> None:
    try:
        response = await client.get(
            "https://api.binance.com/api/v3/ticker/price",
            params={"symbols": '["BTCUSDT","ETHUSDT","SOLUSDT"]'},
        )
        response.raise_for_status()
        snapshot["sources"]["binance"] = {
            item["symbol"]: float(item["price"]) for item in response.json()
        }
    except Exception as exc:  # noqa: BLE001
        snapshot["sources"]["binance"] = {}
        snapshot["notes"].append(f"binance fetch failed: {type(exc).__name__}")


async def _fetch_coinbase(client: httpx.AsyncClient, snapshot: dict[str, Any]) -> None:
    products = {"BTC-USDT": "BTCUSDT", "ETH-USDT": "ETHUSDT", "SOL-USDT": "SOLUSDT"}
    prices: dict[str, float] = {}
    for product, symbol in products.items():
        try:
            response = await client.get(f"https://api.exchange.coinbase.com/products/{product}/ticker")
            response.raise_for_status()
            prices[symbol] = float(response.json()["price"])
        except Exception as exc:  # noqa: BLE001
            snapshot["notes"].append(f"coinbase {product} fetch failed: {type(exc).__name__}")
    snapshot["sources"]["coinbase"] = prices


def _estimate_spreads(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    binance = snapshot["sources"].get("binance") or {}
    coinbase = snapshot["sources"].get("coinbase") or {}
    spreads: list[dict[str, Any]] = []
    for symbol in SYMBOLS:
        left = binance.get(symbol)
        right = coinbase.get(symbol)
        if not left or not right:
            continue
        mid = (left + right) / 2
        spread_bps = abs(left - right) / mid * 10_000
        cheaper = "binance" if left < right else "coinbase"
        richer = "coinbase" if cheaper == "binance" else "binance"
        spreads.append(
            {
                "symbol": symbol,
                "binance": left,
                "coinbase": right,
                "spread_bps": round(spread_bps, 3),
                "cheaper": cheaper,
                "richer": richer,
            }
        )
    return spreads
