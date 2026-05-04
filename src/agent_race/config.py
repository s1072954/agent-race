from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = ROOT / ".env"


def load_env_file(path: str | Path = DEFAULT_ENV_PATH) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    nvidia_api_key: str | None
    nvidia_base_url: str
    nvidia_default_model: str
    nvidia_models: list[str]
    nvidia_summary_model: str
    nvidia_global_rpm: int
    nvidia_model_rpm: int
    max_parallel_llm_calls: int
    max_subagent_tasks: int
    base_path: str
    tick_seconds: int
    summary_every_ticks: int
    db_path: Path
    workspace_dir: Path
    scheduler_enabled: bool
    live_trading_enabled: bool
    shell_tools_enabled: bool
    prompt_dir: Path

    @property
    def can_call_llm(self) -> bool:
        return bool(self.nvidia_api_key)


def load_settings(env_path: str | Path = DEFAULT_ENV_PATH) -> Settings:
    load_env_file(env_path)
    default_models = [
        "qwen/qwen3-coder-480b-a35b-instruct",
        "nvidia/nemotron-3-super-120b-a12b",
        "minimaxai/minimax-m2.7",
        "z-ai/glm4.7",
    ]
    base_path = os.getenv("AGENT_RACE_BASE_PATH", "/agent-race").strip() or "/agent-race"
    if not base_path.startswith("/"):
        base_path = f"/{base_path}"
    return Settings(
        nvidia_api_key=os.getenv("NVIDIA_API_KEY") or None,
        nvidia_base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/"),
        nvidia_default_model=os.getenv("NVIDIA_MODEL", "nvidia/nemotron-3-super-120b-a12b"),
        nvidia_models=env_list("NVIDIA_MODELS", default_models),
        nvidia_summary_model=os.getenv("NVIDIA_SUMMARY_MODEL", "nvidia/nemotron-3-nano-30b-a3b"),
        nvidia_global_rpm=env_int("NVIDIA_GLOBAL_RPM", 8),
        nvidia_model_rpm=env_int("NVIDIA_MODEL_RPM", 2),
        max_parallel_llm_calls=max(1, env_int("AGENT_RACE_MAX_PARALLEL_LLM_CALLS", 1)),
        max_subagent_tasks=max(0, env_int("AGENT_RACE_MAX_SUBAGENT_TASKS", 1)),
        base_path=base_path.rstrip("/"),
        tick_seconds=max(60, env_int("AGENT_RACE_TICK_SECONDS", 900)),
        summary_every_ticks=max(1, env_int("AGENT_RACE_SUMMARY_EVERY_TICKS", 1)),
        db_path=Path(os.getenv("AGENT_RACE_DB_PATH", "data/agent_race.sqlite")),
        workspace_dir=Path(os.getenv("AGENT_RACE_WORKSPACE_DIR", "data/agents")),
        scheduler_enabled=env_bool("AGENT_RACE_SCHEDULER_ENABLED", True),
        live_trading_enabled=env_bool("LIVE_TRADING_ENABLED", False),
        shell_tools_enabled=env_bool("AGENT_RACE_ENABLE_SHELL_TOOLS", False),
        prompt_dir=Path(os.getenv("AGENT_RACE_PROMPT_DIR", "prompts")),
    )
