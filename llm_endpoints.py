"""Central LLM inference endpoints and usage helpers.

This file consolidates the OpenAI-compatible cloud LLM endpoints found in:
- podcast/llm_api.py
- coin_trader/ai_summary/llm_api.py
- coin_trader/ai_summary/ai_news.py
- openclaw/nv_test.py
- openclaw/nvidia_compat_test.py

Usage:
    python llm_endpoints.py --list
    python llm_endpoints.py --provider openrouter --prompt "用繁體中文說明你是誰"
    python llm_endpoints.py --provider nvidia --prompt "請只回答：正常"

In code:
    from llm_endpoints import get_llm

    llm = get_llm("openrouter")
    text = llm.chat("請用三點摘要今天的 AI 新聞")
    print(text)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from openai import OpenAI


ROOT = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = ROOT / ".env"


@dataclass(frozen=True)
class LLMEndpoint:
    name: str
    provider: str
    base_url_env: str
    default_base_url: str
    model_env: str
    default_model: str
    api_key_envs: tuple[str, ...]
    endpoint_path: str = "/chat/completions"
    notes: str = ""

    @property
    def default_chat_completions_url(self) -> str:
        return self.default_base_url.rstrip("/") + self.endpoint_path


ENDPOINTS: dict[str, LLMEndpoint] = {
    "openrouter": LLMEndpoint(
        name="OpenRouter",
        provider="openrouter",
        base_url_env="OPENROUTER_BASE_URL",
        default_base_url="https://openrouter.ai/api/v1",
        model_env="OPENROUTER_MODEL",
        default_model="google/gemma-4-31b-it:free",
        api_key_envs=("OPENROUTER_API_KEY", "openrouter_api_key"),
        notes=(
            "Used by podcast/llm_api.py, coin_trader/ai_summary/llm_api.py, "
            "and coin_trader AI news fallback."
        ),
    ),
    "nvidia": LLMEndpoint(
        name="NVIDIA NIM",
        provider="nvidia",
        base_url_env="NVIDIA_BASE_URL",
        default_base_url="https://integrate.api.nvidia.com/v1",
        model_env="NVIDIA_MODEL",
        default_model="nvidia/nemotron-3-super-120b-a12b",
        api_key_envs=("NVIDIA_API_KEY",),
        notes=(
            "Used by openclaw/nv_test.py, openclaw/nvidia_compat_test.py, "
            "coin_trader/ai_summary/nv_test.py, and coin_trader AI news primary path."
        ),
    ),
}


def load_env_file(env_path: str | Path = DEFAULT_ENV_PATH) -> None:
    path = Path(env_path)
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def first_env_value(names: Iterable[str]) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return None


class OpenAICompatibleLLM:
    def __init__(
        self,
        endpoint: LLMEndpoint,
        *,
        env_path: str | Path = DEFAULT_ENV_PATH,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        load_env_file(env_path)

        self.endpoint = endpoint
        self.base_url = base_url or os.getenv(endpoint.base_url_env) or endpoint.default_base_url
        self.model = model or os.getenv(endpoint.model_env) or endpoint.default_model
        self.api_key = api_key or first_env_value(endpoint.api_key_envs)
        if not self.api_key:
            keys = ", ".join(endpoint.api_key_envs)
            raise RuntimeError(f"Missing API key for {endpoint.name}. Set one of: {keys}")

        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        top_p: float | None = None,
        max_tokens: int = 1024,
        stream: bool = False,
        extra_body: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs,
        }
        if top_p is not None:
            payload["top_p"] = top_p
        if extra_body is not None:
            payload["extra_body"] = extra_body

        return self.client.chat.completions.create(**payload)

    def chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        disable_reasoning: bool = True,
        **kwargs: Any,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        extra_body = kwargs.pop("extra_body", None)
        if disable_reasoning:
            extra_body = merge_disable_reasoning(self.endpoint.provider, extra_body)

        response = self.create_chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
            **kwargs,
        )
        return response.choices[0].message.content or ""

    def stream_chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        disable_reasoning: bool = True,
        **kwargs: Any,
    ) -> Iterable[str]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        extra_body = kwargs.pop("extra_body", None)
        if disable_reasoning:
            extra_body = merge_disable_reasoning(self.endpoint.provider, extra_body)

        stream = self.create_chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            extra_body=extra_body,
            **kwargs,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            content = chunk.choices[0].delta.content
            if content:
                yield content


def merge_disable_reasoning(provider: str, extra_body: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(extra_body or {})
    if provider == "nvidia":
        merged.setdefault("chat_template_kwargs", {"enable_thinking": False})
    elif provider == "openrouter":
        merged.setdefault("reasoning", {"enabled": False})
    return merged


def get_llm(
    provider: str = "openrouter",
    *,
    env_path: str | Path = DEFAULT_ENV_PATH,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> OpenAICompatibleLLM:
    key = provider.lower().strip()
    if key not in ENDPOINTS:
        available = ", ".join(sorted(ENDPOINTS))
        raise ValueError(f"Unknown provider '{provider}'. Available providers: {available}")
    return OpenAICompatibleLLM(
        ENDPOINTS[key],
        env_path=env_path,
        api_key=api_key,
        base_url=base_url,
        model=model,
    )


def list_endpoints() -> None:
    load_env_file()
    for key, endpoint in ENDPOINTS.items():
        configured_url = os.getenv(endpoint.base_url_env) or endpoint.default_base_url
        configured_model = os.getenv(endpoint.model_env) or endpoint.default_model
        has_key = bool(first_env_value(endpoint.api_key_envs))
        print(f"{key}: {endpoint.name}")
        print(f"  base_url env: {endpoint.base_url_env}")
        print(f"  base_url: {configured_url}")
        print(f"  chat endpoint: {configured_url.rstrip('/')}{endpoint.endpoint_path}")
        print(f"  model env: {endpoint.model_env}")
        print(f"  model: {configured_model}")
        print(f"  key envs: {', '.join(endpoint.api_key_envs)}")
        print(f"  key loaded: {'yes' if has_key else 'no'}")
        print(f"  notes: {endpoint.notes}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Central OpenAI-compatible LLM endpoint helper.")
    parser.add_argument("--list", action="store_true", help="List configured LLM endpoints.")
    parser.add_argument("--provider", default="openrouter", choices=sorted(ENDPOINTS), help="Provider to call.")
    parser.add_argument("--prompt", help="Prompt to send to the selected provider.")
    parser.add_argument("--system", help="Optional system prompt.")
    parser.add_argument("--model", help="Override model for this call.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--stream", action="store_true", help="Stream response text.")
    args = parser.parse_args()

    if args.list or not args.prompt:
        list_endpoints()
        if not args.prompt:
            return 0

    llm = get_llm(args.provider, model=args.model)
    if args.stream:
        for part in llm.stream_chat(
            args.prompt,
            system=args.system,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ):
            print(part, end="", flush=True)
        print()
    else:
        print(
            llm.chat(
                args.prompt,
                system=args.system,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
