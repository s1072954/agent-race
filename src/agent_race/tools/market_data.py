from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx


CORE_BASES = {
    "BTC",
    "ETH",
    "SOL",
    "XRP",
    "DOGE",
    "ADA",
    "AVAX",
    "LINK",
    "LTC",
    "BCH",
    "DOT",
    "NEAR",
    "ARB",
    "OP",
    "SUI",
    "APT",
    "UNI",
    "AAVE",
}
MIN_QUOTE_VOLUME_USD = 5_000_000


async def fetch_market_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sources": {},
        "notes": [],
        "data_quality": [
            "Spot spreads use last-traded prices, not executable bid/ask depth.",
            "USD and USDT comparisons include stablecoin basis and transfer risk.",
            "Funding candidates are research leads, not trade signals.",
        ],
    }
    async with httpx.AsyncClient(timeout=14) as client:
        await asyncio.gather(
            _fetch_binance_spot(client, snapshot),
            _fetch_okx_spot(client, snapshot),
            _fetch_coinbase_spot(client, snapshot),
            _fetch_binance_funding(client, snapshot),
        )
    snapshot["spreads"] = _estimate_spreads(snapshot)
    snapshot["funding_rates"] = _rank_funding(snapshot)
    snapshot["opportunities"] = _build_opportunities(snapshot)
    return snapshot


async def _fetch_binance_spot(client: httpx.AsyncClient, snapshot: dict[str, Any]) -> None:
    try:
        response = await client.get("https://api.binance.com/api/v3/ticker/24hr")
        response.raise_for_status()
        prices: dict[str, dict[str, Any]] = {}
        for item in response.json():
            symbol = item.get("symbol", "")
            if not symbol.endswith("USDT"):
                continue
            base = symbol[:-4]
            quote_volume = _float(item.get("quoteVolume"))
            last = _float(item.get("lastPrice"))
            if not last or quote_volume < MIN_QUOTE_VOLUME_USD:
                continue
            prices[symbol] = {
                "base": base,
                "quote": "USDT",
                "last": last,
                "quote_volume_usd": quote_volume,
            }
        snapshot["sources"]["binance"] = prices
    except Exception as exc:  # noqa: BLE001
        snapshot["sources"]["binance"] = {}
        snapshot["notes"].append(f"binance spot fetch failed: {type(exc).__name__}: {exc}")


async def _fetch_okx_spot(client: httpx.AsyncClient, snapshot: dict[str, Any]) -> None:
    try:
        response = await client.get("https://www.okx.com/api/v5/market/tickers", params={"instType": "SPOT"})
        response.raise_for_status()
        prices: dict[str, dict[str, Any]] = {}
        for item in response.json().get("data", []):
            inst_id = item.get("instId", "")
            if not inst_id.endswith("-USDT"):
                continue
            base = inst_id.split("-")[0]
            last = _float(item.get("last"))
            volume = _float(item.get("volCcy24h"))
            if not last or volume < MIN_QUOTE_VOLUME_USD:
                continue
            prices[f"{base}USDT"] = {
                "base": base,
                "quote": "USDT",
                "last": last,
                "quote_volume_usd": volume,
            }
        snapshot["sources"]["okx"] = prices
    except Exception as exc:  # noqa: BLE001
        snapshot["sources"]["okx"] = {}
        snapshot["notes"].append(f"okx spot fetch failed: {type(exc).__name__}: {exc}")


async def _fetch_coinbase_spot(client: httpx.AsyncClient, snapshot: dict[str, Any]) -> None:
    prices: dict[str, dict[str, Any]] = {}
    products = [f"{base}-USD" for base in sorted(CORE_BASES)]

    async def fetch_product(product: str) -> None:
        base = product.split("-")[0]
        try:
            response = await client.get(f"https://api.exchange.coinbase.com/products/{product}/ticker")
            if response.status_code == 404:
                return
            response.raise_for_status()
            data = response.json()
            last = _float(data.get("price"))
            volume = _float(data.get("volume")) * last
            if not last or volume < MIN_QUOTE_VOLUME_USD:
                return
            prices[f"{base}USD"] = {
                "base": base,
                "quote": "USD",
                "last": last,
                "quote_volume_usd": volume,
            }
        except Exception as exc:  # noqa: BLE001
            snapshot["notes"].append(f"coinbase {product} fetch failed: {type(exc).__name__}")

    await asyncio.gather(*(fetch_product(product) for product in products))
    snapshot["sources"]["coinbase"] = prices


async def _fetch_binance_funding(client: httpx.AsyncClient, snapshot: dict[str, Any]) -> None:
    try:
        response = await client.get("https://fapi.binance.com/fapi/v1/premiumIndex")
        response.raise_for_status()
        rates: dict[str, dict[str, Any]] = {}
        for item in response.json():
            symbol = item.get("symbol", "")
            if not symbol.endswith("USDT"):
                continue
            base = symbol[:-4]
            rate = _float(item.get("lastFundingRate"))
            if base not in CORE_BASES and abs(rate) < 0.0003:
                continue
            rates[symbol] = {
                "exchange": "binance_futures",
                "base": base,
                "quote": "USDT",
                "last_funding_rate": rate,
                "funding_bps": round(rate * 10_000, 4),
                "annualized_percent": round(rate * 3 * 365 * 100, 2),
                "next_funding_time": item.get("nextFundingTime"),
            }
        snapshot["sources"]["binance_funding"] = rates
    except Exception as exc:  # noqa: BLE001
        snapshot["sources"]["binance_funding"] = {}
        snapshot["notes"].append(f"binance funding fetch failed: {type(exc).__name__}: {exc}")


def _estimate_spreads(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    comparable: dict[str, list[dict[str, Any]]] = {}
    for exchange, prices in snapshot["sources"].items():
        if exchange.endswith("funding"):
            continue
        for symbol, payload in prices.items():
            base = payload.get("base")
            quote = payload.get("quote")
            if not base or quote not in {"USDT", "USD"}:
                continue
            comparable.setdefault(base, []).append({"exchange": exchange, "symbol": symbol, **payload})

    for base, venues in comparable.items():
        for left_index, left in enumerate(venues):
            for right in venues[left_index + 1 :]:
                if not left["last"] or not right["last"]:
                    continue
                mid = (left["last"] + right["last"]) / 2
                spread_bps = abs(left["last"] - right["last"]) / mid * 10_000
                lower, higher = (left, right) if left["last"] < right["last"] else (right, left)
                rows.append(
                    {
                        "kind": "spot_spread",
                        "symbol": f"{base}{left['quote']}/{right['quote']}",
                        "base": base,
                        "left_exchange": left["exchange"],
                        "right_exchange": right["exchange"],
                        "lower_exchange": lower["exchange"],
                        "higher_exchange": higher["exchange"],
                        "lower_price": lower["last"],
                        "higher_price": higher["last"],
                        "spread_bps": round(spread_bps, 4),
                        "quote_mismatch": left["quote"] != right["quote"],
                        "min_quote_volume_usd": round(min(left["quote_volume_usd"], right["quote_volume_usd"]), 2),
                    }
                )
    return sorted(rows, key=lambda item: item["spread_bps"], reverse=True)[:30]


def _rank_funding(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    rates = list((snapshot["sources"].get("binance_funding") or {}).values())
    return sorted(rates, key=lambda item: abs(item["funding_bps"]), reverse=True)[:20]


def _build_opportunities(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    opportunities: list[dict[str, Any]] = []
    spot_symbols = set()
    for exchange in ("binance", "okx"):
        spot_symbols.update((snapshot["sources"].get(exchange) or {}).keys())

    for spread in snapshot.get("spreads", [])[:20]:
        estimated_cost_bps = 30.0 if spread["quote_mismatch"] else 18.0
        net_edge_bps = round(spread["spread_bps"] - estimated_cost_bps, 4)
        if net_edge_bps > 5:
            status = "actionable_research"
            confidence = 0.62
        elif spread["spread_bps"] > estimated_cost_bps * 0.55:
            status = "watch"
            confidence = 0.38
        else:
            continue
        opportunities.append(
            {
                "kind": "spot_spread",
                "symbol": spread["symbol"],
                "title": f"{spread['base']} spot spread: buy {spread['lower_exchange']}, sell {spread['higher_exchange']}",
                "gross_edge_bps": spread["spread_bps"],
                "estimated_cost_bps": estimated_cost_bps,
                "net_edge_bps": net_edge_bps,
                "confidence": confidence,
                "status": status,
                "evidence": spread,
                "next_validation": "Fetch bid/ask order books on both venues and recompute executable edge after fees.",
            }
        )

    for funding in snapshot.get("funding_rates", [])[:12]:
        spot_symbol = f"{funding['base']}USDT"
        if spot_symbol not in spot_symbols:
            continue
        funding_bps = funding["funding_bps"]
        gross_edge_bps = abs(funding_bps)
        estimated_cost_bps = 2.5
        net_edge_bps = round(gross_edge_bps - estimated_cost_bps, 4)
        if gross_edge_bps < 3.0:
            continue
        side = "short perp / long spot" if funding_bps > 0 else "long perp / short spot"
        opportunities.append(
            {
                "kind": "funding_rate",
                "symbol": spot_symbol,
                "title": f"{funding['base']} funding candidate: {side}",
                "gross_edge_bps": round(gross_edge_bps, 4),
                "estimated_cost_bps": estimated_cost_bps,
                "net_edge_bps": net_edge_bps,
                "confidence": 0.42 if net_edge_bps > 0 else 0.25,
                "status": "actionable_research" if net_edge_bps > 1.0 else "watch",
                "evidence": funding,
                "next_validation": "Check spot/perp borrow, inventory, liquidation buffer, and historical funding persistence.",
            }
        )

    return sorted(opportunities, key=lambda item: (item["status"] != "actionable_research", -item["net_edge_bps"]))[:20]


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
