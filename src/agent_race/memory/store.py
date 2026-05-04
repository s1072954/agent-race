from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class AgentRaceStore:
    def __init__(self, db_path: Path, workspace_dir: Path) -> None:
        self.db_path = db_path
        self.workspace_dir = workspace_dir
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self.init_db()

    def init_db(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                create table if not exists agents (
                    id text primary key,
                    name text not null,
                    model text not null,
                    status text not null default 'idle',
                    score real not null default 0,
                    last_tick_at text,
                    last_summary text not null default '',
                    workspace_path text not null,
                    payload_json text not null default '{}'
                );

                create table if not exists events (
                    id integer primary key autoincrement,
                    ts text not null,
                    agent_id text,
                    kind text not null,
                    message text not null,
                    payload_json text not null default '{}'
                );

                create table if not exists strategies (
                    id integer primary key autoincrement,
                    ts text not null,
                    agent_id text not null,
                    title text not null,
                    hypothesis text not null,
                    expected_edge_bps real not null default 0,
                    risk_score real not null default 5,
                    payload_json text not null default '{}'
                );

                create table if not exists llm_calls (
                    id integer primary key autoincrement,
                    ts text not null,
                    agent_id text,
                    model text not null,
                    status text not null,
                    latency_ms integer not null,
                    prompt_tokens integer,
                    completion_tokens integer,
                    total_tokens integer,
                    error text
                );

                create table if not exists arena_state (
                    key text primary key,
                    value_json text not null,
                    updated_at text not null
                );
                """
            )

    def upsert_agent(self, agent_id: str, name: str, model: str) -> Path:
        workspace_path = self.workspace_dir / agent_id
        workspace_path.mkdir(parents=True, exist_ok=True)
        with self._lock, self._conn:
            self._conn.execute(
                """
                insert into agents (id, name, model, workspace_path)
                values (?, ?, ?, ?)
                on conflict(id) do update set
                    name=excluded.name,
                    model=excluded.model,
                    workspace_path=excluded.workspace_path
                """,
                (agent_id, name, model, str(workspace_path)),
            )
        return workspace_path

    def update_agent_status(
        self,
        *,
        agent_id: str,
        status: str,
        summary: str,
        score_delta: float,
        payload: dict[str, Any],
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                update agents
                set status=?,
                    score=score + ?,
                    last_tick_at=?,
                    last_summary=?,
                    payload_json=?
                where id=?
                """,
                (status, score_delta, utc_now(), summary, json.dumps(payload, ensure_ascii=False), agent_id),
            )

    def record_event(
        self,
        kind: str,
        message: str,
        *,
        agent_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "insert into events (ts, agent_id, kind, message, payload_json) values (?, ?, ?, ?, ?)",
                (utc_now(), agent_id, kind, message, json.dumps(payload or {}, ensure_ascii=False)),
            )

    def record_strategy(
        self,
        *,
        agent_id: str,
        title: str,
        hypothesis: str,
        expected_edge_bps: float,
        risk_score: float,
        payload: dict[str, Any],
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                insert into strategies
                    (ts, agent_id, title, hypothesis, expected_edge_bps, risk_score, payload_json)
                values (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_now(),
                    agent_id,
                    title,
                    hypothesis,
                    expected_edge_bps,
                    risk_score,
                    json.dumps(payload, ensure_ascii=False),
                ),
            )

    def record_llm_call(
        self,
        agent_id: str | None,
        model: str,
        status: str,
        latency_ms: int,
        usage: dict[str, Any],
        error: str | None,
    ) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                insert into llm_calls
                    (ts, agent_id, model, status, latency_ms, prompt_tokens, completion_tokens, total_tokens, error)
                values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    utc_now(),
                    agent_id,
                    model,
                    status,
                    latency_ms,
                    usage.get("prompt_tokens"),
                    usage.get("completion_tokens"),
                    usage.get("total_tokens"),
                    error,
                ),
            )

    def set_state(self, key: str, value: dict[str, Any]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                insert into arena_state (key, value_json, updated_at)
                values (?, ?, ?)
                on conflict(key) do update set
                    value_json=excluded.value_json,
                    updated_at=excluded.updated_at
                """,
                (key, json.dumps(value, ensure_ascii=False), utc_now()),
            )

    def get_state(self, key: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            row = self._conn.execute("select value_json from arena_state where key=?", (key,)).fetchone()
        if not row:
            return default or {}
        return json.loads(row["value_json"])

    def list_agents(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("select * from agents order by score desc, id asc").fetchall()
        return [self._row_to_dict(row) for row in rows]

    def recent_events(self, agent_id: str | None = None, limit: int = 40) -> list[dict[str, Any]]:
        with self._lock:
            if agent_id:
                rows = self._conn.execute(
                    "select * from events where agent_id=? order by id desc limit ?", (agent_id, limit)
                ).fetchall()
            else:
                rows = self._conn.execute("select * from events order by id desc limit ?", (limit,)).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def recent_strategies(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("select * from strategies order by id desc limit ?", (limit,)).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def llm_usage(self, limit: int = 100) -> dict[str, Any]:
        with self._lock:
            totals = self._conn.execute(
                """
                select status, count(*) as calls, coalesce(sum(total_tokens), 0) as total_tokens
                from llm_calls
                group by status
                """
            ).fetchall()
            recent = self._conn.execute(
                "select * from llm_calls order by id desc limit ?", (limit,)
            ).fetchall()
        return {
            "totals": [self._row_to_dict(row) for row in totals],
            "recent": [self._row_to_dict(row) for row in recent],
        }

    def overview(self) -> dict[str, Any]:
        return {
            "agents": self.list_agents(),
            "events": self.recent_events(limit=25),
            "strategies": self.recent_strategies(limit=20),
            "llm_usage": self.llm_usage(limit=25),
            "arena_summary": self.get_state("arena_summary", {}),
            "last_market_snapshot": self.get_state("last_market_snapshot", {}),
        }

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        for key in ("payload_json", "value_json"):
            if key in data:
                try:
                    data[key] = json.loads(data[key])
                except json.JSONDecodeError:
                    data[key] = {}
        return data
