from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode

import httpx

from agent_race.config import Settings


BINANCE_BASE_URL = "https://api.binance.com"


async def fetch_borrow_snapshot(settings: Settings, opportunities: list[dict[str, Any]]) -> dict[str, Any]:
    assets = sorted(_borrow_assets_from_opportunities(opportunities))
    snapshot: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider": "binance_margin",
        "configured": settings.can_call_binance_margin_read,
        "assets_requested": assets,
        "assets": {},
        "notes": [],
        "errors": [],
    }
    if not assets:
        snapshot["notes"].append("No negative-funding candidates require spot borrow validation this cycle.")
        return snapshot
    if not settings.can_call_binance_margin_read:
        snapshot["notes"].append(
            "BINANCE_API_KEY, BINANCE_API_SECRET, and BINANCE_MARGIN_READ_ENABLED are required for borrow validation."
        )
        return snapshot

    async with httpx.AsyncClient(timeout=10) as client:
        inventory = await _fetch_available_inventory(client, settings, snapshot)

        async def fetch_asset(asset: str) -> None:
            asset_info: dict[str, Any] = {
                "asset": asset,
                "status": "unknown",
                "available_inventory_amount": _float(inventory.get(asset)),
                "daily_interest_rate": None,
                "daily_interest_bps": None,
                "borrow_cost_bps_per_funding": None,
                "max_borrowable_amount": None,
                "borrow_limit_amount": None,
                "notes": [],
            }
            await _attach_interest_rate(client, settings, asset_info)
            if settings.binance_query_max_borrowable:
                await _attach_max_borrowable(client, settings, asset_info)
            asset_info["status"] = _asset_status(asset_info, settings.binance_query_max_borrowable)
            snapshot["assets"][asset] = asset_info

        await asyncio.gather(*(fetch_asset(asset) for asset in assets))
    return snapshot


def _borrow_assets_from_opportunities(opportunities: list[dict[str, Any]]) -> set[str]:
    assets: set[str] = set()
    for item in opportunities:
        evidence = item.get("evidence") or {}
        funding_bps = float(evidence.get("funding_bps") or 0)
        if item.get("kind") != "funding_rate" or funding_bps >= 0:
            continue
        base = evidence.get("base") or str(item.get("symbol", "")).removesuffix("USDT")
        if base:
            assets.add(base.upper())
    return assets


async def _fetch_available_inventory(
    client: httpx.AsyncClient,
    settings: Settings,
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    try:
        data = await _signed_get(client, settings, "/sapi/v1/margin/available-inventory", {"type": "MARGIN"})
        assets = data.get("assets") if isinstance(data, dict) else None
        if isinstance(assets, dict):
            snapshot["inventory_update_time"] = data.get("updateTime")
            return assets
        snapshot["errors"].append("Binance available-inventory response did not include assets.")
    except Exception as exc:  # noqa: BLE001
        snapshot["errors"].append(f"available-inventory failed: {type(exc).__name__}: {exc}")
    return {}


async def _attach_interest_rate(
    client: httpx.AsyncClient,
    settings: Settings,
    asset_info: dict[str, Any],
) -> None:
    asset = asset_info["asset"]
    try:
        data = await _signed_get(client, settings, "/sapi/v1/margin/interestRateHistory", {"asset": asset})
        rows = data if isinstance(data, list) else []
        if not rows:
            asset_info["notes"].append("No Binance margin interest rate history returned.")
            return
        latest = max(rows, key=lambda row: int(row.get("timestamp") or 0))
        daily_rate = _float(latest.get("dailyInterestRate"))
        if daily_rate is None:
            asset_info["notes"].append("Latest daily interest rate was empty.")
            return
        asset_info["daily_interest_rate"] = daily_rate
        asset_info["daily_interest_bps"] = round(daily_rate * 10_000, 4)
        asset_info["borrow_cost_bps_per_funding"] = round(daily_rate * 10_000 / 3, 4)
        asset_info["interest_rate_ts"] = latest.get("timestamp")
        asset_info["vip_level"] = latest.get("vipLevel")
    except Exception as exc:  # noqa: BLE001
        asset_info["notes"].append(f"interestRateHistory failed: {type(exc).__name__}: {exc}")


async def _attach_max_borrowable(
    client: httpx.AsyncClient,
    settings: Settings,
    asset_info: dict[str, Any],
) -> None:
    asset = asset_info["asset"]
    try:
        data = await _signed_get(client, settings, "/sapi/v1/margin/maxBorrowable", {"asset": asset})
        if not isinstance(data, dict):
            asset_info["notes"].append("maxBorrowable response was not an object.")
            return
        asset_info["max_borrowable_amount"] = _float(data.get("amount"))
        asset_info["borrow_limit_amount"] = _float(data.get("borrowLimit"))
    except Exception as exc:  # noqa: BLE001
        asset_info["notes"].append(f"maxBorrowable failed: {type(exc).__name__}: {exc}")


async def _signed_get(
    client: httpx.AsyncClient,
    settings: Settings,
    path: str,
    params: dict[str, Any],
) -> Any:
    if not settings.binance_api_key or not settings.binance_api_secret:
        raise ValueError("Binance API credentials are not configured")
    signed = {
        **params,
        "recvWindow": 5000,
        "timestamp": int(time.time() * 1000),
    }
    query = urlencode(signed)
    signature = hmac.new(settings.binance_api_secret.encode("utf-8"), query.encode("utf-8"), hashlib.sha256).hexdigest()
    response = await client.get(
        f"{BINANCE_BASE_URL}{path}",
        params={**signed, "signature": signature},
        headers={"X-MBX-APIKEY": settings.binance_api_key},
    )
    response.raise_for_status()
    return response.json()


def _asset_status(asset_info: dict[str, Any], max_borrowable_checked: bool) -> str:
    if asset_info.get("daily_interest_rate") is None:
        return "missing_interest_rate"
    inventory = asset_info.get("available_inventory_amount")
    if inventory is None or inventory <= 0:
        return "missing_inventory"
    if max_borrowable_checked and (asset_info.get("max_borrowable_amount") is None or asset_info["max_borrowable_amount"] <= 0):
        return "missing_account_borrow_limit"
    return "ok"


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
