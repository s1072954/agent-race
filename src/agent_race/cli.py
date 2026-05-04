from __future__ import annotations

import argparse
import asyncio

from agent_race.config import load_settings
from agent_race.memory import AgentRaceStore
from agent_race.scheduler import AgentRaceScheduler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Agent Race control CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init-db", help="Initialize SQLite tables and agent rows")
    subparsers.add_parser("tick", help="Run one scheduler cycle")
    subparsers.add_parser("status", help="Print a compact status snapshot")
    return parser


async def run_tick() -> None:
    settings = load_settings()
    store = AgentRaceStore(settings.db_path, settings.workspace_dir)
    scheduler = AgentRaceScheduler(settings, store)
    await scheduler.run_once()


def print_status() -> None:
    settings = load_settings()
    store = AgentRaceStore(settings.db_path, settings.workspace_dir)
    overview = store.overview()
    print(f"agents: {len(overview['agents'])}")
    print(f"summary: {overview['arena_summary'].get('summary', '')[:300]}")
    for agent in overview["agents"]:
        print(f"- {agent['id']} status={agent['status']} score={agent['score']:.3f}")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    settings = load_settings()
    store = AgentRaceStore(settings.db_path, settings.workspace_dir)
    if args.command == "init-db":
        for model in settings.nvidia_models:
            from agent_race.agents import AgentSpec

            spec = AgentSpec.from_model(model)
            store.upsert_agent(spec.id, spec.name, spec.model)
        print(f"initialized {settings.db_path}")
        return 0
    if args.command == "tick":
        asyncio.run(run_tick())
        return 0
    if args.command == "status":
        print_status()
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
