from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import httpx

from agent_race.config import Settings
from agent_race.llm.rate_limiter import AsyncSlidingWindowLimiter
from agent_race.memory.store import AgentRaceStore


class LLMError(RuntimeError):
    pass


class LLMRateLimitError(LLMError):
    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class LLMUsageLimitError(LLMRateLimitError):
    pass


@dataclass
class ChatResult:
    content: str
    model: str
    latency_ms: int
    usage: dict[str, Any]


class NvidiaChatClient:
    def __init__(
        self,
        settings: Settings,
        limiter: AsyncSlidingWindowLimiter,
        store: AgentRaceStore | None = None,
    ) -> None:
        self.settings = settings
        self.limiter = limiter
        self.store = store

    async def chat(
        self,
        *,
        agent_id: str | None,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 900,
        retries: int = 2,
    ) -> ChatResult:
        if not self.settings.nvidia_api_key:
            raise LLMError("NVIDIA_API_KEY is not configured")

        await self.limiter.acquire(model)
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {
            "Authorization": f"Bearer {self.settings.nvidia_api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.settings.nvidia_base_url}/chat/completions"
        attempt = 0
        while True:
            started = time.perf_counter()
            try:
                async with httpx.AsyncClient(timeout=self.settings.nvidia_request_timeout_seconds) as client:
                    response = await client.post(url, headers=headers, json=payload)
                latency_ms = int((time.perf_counter() - started) * 1000)
                if response.status_code == 429:
                    retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                    raise LLMRateLimitError("NVIDIA API rate limited this request", retry_after)
                if _is_usage_limit_response(response.status_code, response.text):
                    retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                    raise LLMUsageLimitError(
                        f"NVIDIA API usage limit returned HTTP {response.status_code}: {response.text[:300]}",
                        retry_after,
                    )
                if response.status_code >= 400:
                    raise LLMError(f"NVIDIA API returned HTTP {response.status_code}: {response.text[:300]}")
                data = response.json()
                content = data["choices"][0]["message"].get("content") or ""
                usage = data.get("usage") or {}
                if self.store:
                    self.store.record_llm_call(
                        agent_id=agent_id,
                        model=model,
                        status="ok",
                        latency_ms=latency_ms,
                        usage=usage,
                        error=None,
                    )
                return ChatResult(content=content, model=model, latency_ms=latency_ms, usage=usage)
            except LLMRateLimitError as exc:
                latency_ms = int((time.perf_counter() - started) * 1000)
                if self.store:
                    self.store.record_llm_call(agent_id, model, "rate_limited", latency_ms, {}, str(exc))
                    self.store.record_limit_fallback(
                        agent_id=agent_id,
                        model=model,
                        message=str(exc),
                        retry_after_seconds=exc.retry_after,
                        default_tick_seconds=self.settings.tick_seconds,
                        default_fallback_tick_seconds=self.settings.fallback_tick_seconds,
                    )
                if attempt >= retries:
                    raise
                await asyncio.sleep(exc.retry_after or min(30, 2**attempt * 5))
            except httpx.TimeoutException as exc:
                latency_ms = int((time.perf_counter() - started) * 1000)
                message = f"NVIDIA API request timed out after {self.settings.nvidia_request_timeout_seconds}s"
                if self.store:
                    self.store.record_llm_call(agent_id, model, "error", latency_ms, {}, message)
                if attempt >= retries:
                    raise LLMError(message) from exc
                await asyncio.sleep(min(20, 2**attempt * 3))
            except (httpx.HTTPError, LLMError) as exc:
                latency_ms = int((time.perf_counter() - started) * 1000)
                if self.store:
                    self.store.record_llm_call(agent_id, model, "error", latency_ms, {}, str(exc))
                if attempt >= retries:
                    raise LLMError(str(exc)) from exc
                await asyncio.sleep(min(20, 2**attempt * 3))
            attempt += 1


def _parse_retry_after(raw: str | None) -> float | None:
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def _is_usage_limit_response(status_code: int, body: str) -> bool:
    if status_code not in {402, 403}:
        return False
    lowered = body.lower()
    return any(
        marker in lowered
        for marker in (
            "quota",
            "rate limit",
            "rate_limit",
            "usage limit",
            "credit",
            "credits",
            "exceeded",
            "too many requests",
        )
    )
