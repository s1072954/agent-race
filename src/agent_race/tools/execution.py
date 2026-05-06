from __future__ import annotations

import asyncio
from typing import Any

import httpx


PAPER_NOTIONAL_USDT = 100.0
MAX_VALIDATIONS_PER_CYCLE = 8
SPOT_TAKER_FEE_BPS = {
    "binance": 10.0,
    "okx": 10.0,
}
BINANCE_PERP_TAKER_FEE_BPS = 5.0


async def validate_opportunities(
    opportunities: list[dict[str, Any]],
    *,
    notional_usdt: float = PAPER_NOTIONAL_USDT,
    max_items: int = MAX_VALIDATIONS_PER_CYCLE,
    borrow_snapshot: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    selected = opportunities[:max_items]
    if not selected:
        return []
    semaphore = asyncio.Semaphore(3)
    async with httpx.AsyncClient(timeout=10) as client:
        async def validate(item: dict[str, Any]) -> dict[str, Any]:
            async with semaphore:
                try:
                    if item.get("kind") == "spot_spread":
                        return await _validate_spot_spread(client, item, notional_usdt)
                    if item.get("kind") == "funding_rate":
                        return await _validate_funding_rate(client, item, notional_usdt, borrow_snapshot)
                    return _blocked_signal(item, notional_usdt, ["unsupported opportunity kind"])
                except Exception as exc:  # noqa: BLE001
                    return _blocked_signal(item, notional_usdt, [f"validation failed: {type(exc).__name__}: {exc}"])

        return await asyncio.gather(*(validate(item) for item in selected))


async def _validate_spot_spread(
    client: httpx.AsyncClient,
    opportunity: dict[str, Any],
    notional_usdt: float,
) -> dict[str, Any]:
    evidence = opportunity.get("evidence") or {}
    lower_exchange = evidence.get("lower_exchange")
    higher_exchange = evidence.get("higher_exchange")
    base = evidence.get("base")
    blockers: list[str] = []
    if evidence.get("quote_mismatch"):
        blockers.append("USD/USDT quote mismatch requires stablecoin basis validation")
    if lower_exchange not in SPOT_TAKER_FEE_BPS or higher_exchange not in SPOT_TAKER_FEE_BPS:
        blockers.append("unsupported spot venue for order book validation")
    if not base:
        blockers.append("missing base asset")
    if blockers:
        return _blocked_signal(opportunity, notional_usdt, blockers)

    buy_book, sell_book = await asyncio.gather(
        _fetch_spot_book(client, lower_exchange, f"{base}USDT"),
        _fetch_spot_book(client, higher_exchange, f"{base}USDT"),
    )
    buy = simulate_buy_with_quote(buy_book["asks"], notional_usdt)
    sell = simulate_sell_base(sell_book["bids"], buy["base_qty"])
    if not buy["filled"]:
        blockers.append(f"insufficient ask depth on {lower_exchange}")
    if not sell["filled"]:
        blockers.append(f"insufficient bid depth on {higher_exchange}")
    gross_edge_bps = (sell["quote_amount"] / buy["quote_amount"] - 1) * 10_000 if buy["quote_amount"] else 0.0
    estimated_cost_bps = SPOT_TAKER_FEE_BPS[lower_exchange] + SPOT_TAKER_FEE_BPS[higher_exchange]
    net_edge_bps = gross_edge_bps - estimated_cost_bps
    if net_edge_bps <= 0:
        blockers.append("net executable edge is not positive after taker fees")
    status = _status_from_blockers_and_edge(blockers, net_edge_bps)
    return {
        "kind": "spot_spread",
        "symbol": opportunity.get("symbol", f"{base}USDT"),
        "title": opportunity.get("title", ""),
        "notional_usdt": notional_usdt,
        "gross_edge_bps": round(gross_edge_bps, 4),
        "estimated_cost_bps": round(estimated_cost_bps, 4),
        "net_edge_bps": round(net_edge_bps, 4),
        "status": status,
        "blockers": blockers,
        "validation": {
            "buy_exchange": lower_exchange,
            "sell_exchange": higher_exchange,
            "buy_avg_price": buy["avg_price"],
            "sell_avg_price": sell["avg_price"],
            "base_qty": buy["base_qty"],
            "sell_quote_amount": sell["quote_amount"],
        },
        "source_opportunity": opportunity,
    }


async def _validate_funding_rate(
    client: httpx.AsyncClient,
    opportunity: dict[str, Any],
    notional_usdt: float,
    borrow_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    evidence = opportunity.get("evidence") or {}
    symbol = opportunity.get("symbol") or f"{evidence.get('base', '')}USDT"
    funding_bps = float(evidence.get("funding_bps") or 0)
    blockers: list[str] = []
    if not symbol.endswith("USDT"):
        blockers.append("unsupported funding symbol")
    spot_book, perp_book = await asyncio.gather(
        _fetch_spot_book(client, "binance", symbol),
        _fetch_binance_perp_book(client, symbol),
    )

    if funding_bps > 0:
        # Receive funding by shorting perp and holding spot.
        spot_buy = simulate_buy_with_quote(spot_book["asks"], notional_usdt)
        perp_sell = simulate_sell_base(perp_book["bids"], spot_buy["base_qty"])
        execution_basis_bps = (
            (perp_sell["quote_amount"] / spot_buy["quote_amount"] - 1) * 10_000
            if spot_buy["quote_amount"]
            else 0.0
        )
        borrow_required = False
        borrow_cost_bps = 0.0
        validation = {
            "structure": "short perp / long spot",
            "spot_avg_price": spot_buy["avg_price"],
            "perp_avg_price": perp_sell["avg_price"],
            "base_qty": spot_buy["base_qty"],
        }
        if not spot_buy["filled"]:
            blockers.append("insufficient Binance spot ask depth")
        if not perp_sell["filled"]:
            blockers.append("insufficient Binance perp bid depth")
    else:
        # Receive negative funding by longing perp and shorting spot; real execution needs borrow.
        base_qty = notional_usdt / best_price(spot_book["bids"])
        spot_sell = simulate_sell_base(spot_book["bids"], base_qty)
        perp_buy = simulate_buy_base(perp_book["asks"], base_qty)
        execution_basis_bps = (
            (spot_sell["quote_amount"] / perp_buy["quote_amount"] - 1) * 10_000
            if perp_buy["quote_amount"]
            else 0.0
        )
        borrow_required = True
        borrow_cost_bps = 0.0
        validation = {
            "structure": "long perp / short spot",
            "spot_avg_price": spot_sell["avg_price"],
            "perp_avg_price": perp_buy["avg_price"],
            "base_qty": base_qty,
        }
        borrow_info = _borrow_info(symbol[:-4], borrow_snapshot)
        validation["borrow"] = borrow_info
        if not borrow_info["configured"]:
            blockers.append("Binance margin read-only API is not configured for borrow validation")
        elif borrow_info["status"] != "ok":
            blockers.append(f"Binance margin borrow data is not usable: {borrow_info['status']}")
        else:
            borrow_cost_bps = float(borrow_info.get("borrow_cost_bps_per_funding") or 0)
            required_base = base_qty
            available_inventory = borrow_info.get("available_inventory_amount")
            max_borrowable = borrow_info.get("max_borrowable_amount")
            if available_inventory is not None and available_inventory < required_base:
                blockers.append("Binance margin inventory is below paper notional requirement")
            if max_borrowable is not None and max_borrowable < required_base:
                blockers.append("Binance account max borrowable is below paper notional requirement")
        if not spot_sell["filled"]:
            blockers.append("insufficient Binance spot bid depth")
        if not perp_buy["filled"]:
            blockers.append("insufficient Binance perp ask depth")

    estimated_cost_bps = SPOT_TAKER_FEE_BPS["binance"] + BINANCE_PERP_TAKER_FEE_BPS + borrow_cost_bps
    gross_edge_bps = abs(funding_bps) + execution_basis_bps
    net_edge_bps = gross_edge_bps - estimated_cost_bps
    if net_edge_bps <= 0:
        blockers.append("net funding edge is not positive after entry basis and taker fees")
    status = _status_from_blockers_and_edge(blockers, net_edge_bps)
    if borrow_required and blockers and net_edge_bps > 0:
        status = "research_only"
    return {
        "kind": "funding_rate",
        "symbol": symbol,
        "title": opportunity.get("title", ""),
        "notional_usdt": notional_usdt,
        "gross_edge_bps": round(gross_edge_bps, 4),
        "estimated_cost_bps": round(estimated_cost_bps, 4),
        "net_edge_bps": round(net_edge_bps, 4),
        "status": status,
        "blockers": blockers,
        "validation": {
            **validation,
            "funding_bps": funding_bps,
            "execution_basis_bps": round(execution_basis_bps, 4),
            "borrow_required": borrow_required,
            "borrow_cost_bps_per_funding": round(borrow_cost_bps, 4),
        },
        "source_opportunity": opportunity,
    }


async def _fetch_spot_book(client: httpx.AsyncClient, exchange: str, symbol: str) -> dict[str, list[list[float]]]:
    if exchange == "binance":
        response = await client.get("https://api.binance.com/api/v3/depth", params={"symbol": symbol, "limit": 50})
        response.raise_for_status()
        data = response.json()
        return {"bids": _levels(data.get("bids", [])), "asks": _levels(data.get("asks", []))}
    if exchange == "okx":
        inst_id = symbol[:-4] + "-USDT"
        response = await client.get("https://www.okx.com/api/v5/market/books", params={"instId": inst_id, "sz": 50})
        response.raise_for_status()
        data = response.json().get("data", [{}])[0]
        return {"bids": _levels(data.get("bids", [])), "asks": _levels(data.get("asks", []))}
    raise ValueError(f"unsupported spot exchange: {exchange}")


async def _fetch_binance_perp_book(client: httpx.AsyncClient, symbol: str) -> dict[str, list[list[float]]]:
    response = await client.get("https://fapi.binance.com/fapi/v1/depth", params={"symbol": symbol, "limit": 50})
    response.raise_for_status()
    data = response.json()
    return {"bids": _levels(data.get("bids", [])), "asks": _levels(data.get("asks", []))}


def _levels(raw_levels: list[list[Any]]) -> list[list[float]]:
    return [[float(level[0]), float(level[1])] for level in raw_levels if len(level) >= 2]


def simulate_buy_with_quote(asks: list[list[float]], quote_amount: float) -> dict[str, Any]:
    remaining = quote_amount
    base_qty = 0.0
    spent = 0.0
    for price, qty in asks:
        quote_available = price * qty
        use_quote = min(remaining, quote_available)
        if use_quote <= 0:
            break
        base_qty += use_quote / price
        spent += use_quote
        remaining -= use_quote
        if remaining <= 1e-9:
            break
    return {
        "filled": remaining <= 1e-6,
        "base_qty": base_qty,
        "quote_amount": spent,
        "avg_price": spent / base_qty if base_qty else 0.0,
    }


def simulate_buy_base(asks: list[list[float]], base_qty: float) -> dict[str, Any]:
    return _walk_base(asks, base_qty)


def simulate_sell_base(bids: list[list[float]], base_qty: float) -> dict[str, Any]:
    return _walk_base(bids, base_qty)


def _walk_base(levels: list[list[float]], base_qty: float) -> dict[str, Any]:
    remaining = base_qty
    filled_base = 0.0
    quote_amount = 0.0
    for price, qty in levels:
        use_base = min(remaining, qty)
        if use_base <= 0:
            break
        filled_base += use_base
        quote_amount += use_base * price
        remaining -= use_base
        if remaining <= 1e-12:
            break
    return {
        "filled": remaining <= 1e-9,
        "base_qty": filled_base,
        "quote_amount": quote_amount,
        "avg_price": quote_amount / filled_base if filled_base else 0.0,
    }


def best_price(levels: list[list[float]]) -> float:
    if not levels:
        raise ValueError("empty order book")
    return levels[0][0]


def _borrow_info(base: str, borrow_snapshot: dict[str, Any] | None) -> dict[str, Any]:
    if not borrow_snapshot:
        return {"asset": base, "configured": False, "status": "missing_snapshot"}
    asset = (borrow_snapshot.get("assets") or {}).get(base) or {}
    if not borrow_snapshot.get("configured"):
        return {"asset": base, "configured": False, "status": "not_configured"}
    if not asset:
        return {"asset": base, "configured": True, "status": "asset_not_returned"}
    return {
        "asset": base,
        "configured": True,
        "status": asset.get("status", "unknown"),
        "available_inventory_amount": asset.get("available_inventory_amount"),
        "daily_interest_bps": asset.get("daily_interest_bps"),
        "borrow_cost_bps_per_funding": asset.get("borrow_cost_bps_per_funding"),
        "max_borrowable_amount": asset.get("max_borrowable_amount"),
        "borrow_limit_amount": asset.get("borrow_limit_amount"),
        "notes": asset.get("notes", []),
    }


def _status_from_blockers_and_edge(blockers: list[str], net_edge_bps: float) -> str:
    if blockers:
        return "blocked"
    if net_edge_bps >= 5:
        return "paper_trade_ready"
    if net_edge_bps > 0:
        return "watch"
    return "blocked"


def _blocked_signal(opportunity: dict[str, Any], notional_usdt: float, blockers: list[str]) -> dict[str, Any]:
    return {
        "kind": opportunity.get("kind", "unknown"),
        "symbol": opportunity.get("symbol", "unknown"),
        "title": opportunity.get("title", ""),
        "notional_usdt": notional_usdt,
        "gross_edge_bps": 0.0,
        "estimated_cost_bps": 0.0,
        "net_edge_bps": 0.0,
        "status": "blocked",
        "blockers": blockers,
        "validation": {},
        "source_opportunity": opportunity,
    }
