# Agent Race

Agent Race is a small MVP framework for running multiple LLM-backed agents that compete to discover, evaluate, and refine crypto arbitrage strategies.

The first implementation is intentionally conservative:

- NVIDIA NIM / Build API is treated as rate-limited trial capacity, not unlimited production capacity.
- Agents can research and backtest ideas, but live trading is disabled by default.
- Each root agent has isolated memory, workspace files, strategy records, and score.
- The scheduler wakes agents on a fixed interval and records every result to SQLite.
- A FastAPI dashboard exposes current status at `/agent-race/`.

## Local Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install -e .
copy .env.example .env
```

Fill `NVIDIA_API_KEY` in `.env`, then run:

```powershell
.\.venv\Scripts\python -m agent_race.cli init-db
.\.venv\Scripts\python -m agent_race.cli tick
.\.venv\Scripts\python -m uvicorn agent_race.web.app:app --host 127.0.0.1 --port 8010
```

Open:

```text
http://127.0.0.1:8010/agent-race/
```

## Deployment Shape

The intended GCP deployment runs:

- app: `127.0.0.1:8010`
- public path: `http://34.80.67.78/agent-race/`
- systemd unit: `agent-race.service`
- nginx location: `/agent-race/`

The `.env` file is copied to the VM for runtime only and must never be committed.

## Safety Defaults

Live trading is disabled unless all of these are explicitly set:

```text
LIVE_TRADING_ENABLED=true
AGENT_RACE_ENABLE_SHELL_TOOLS=true
```

Even then, the MVP does not include exchange order placement. Add paper trading and exchange adapters only after the scoring and risk checks are stable.

## NVIDIA API Notes

NVIDIA Build / NIM hosted APIs are trial resources with usage limits. The framework therefore includes:

- global requests-per-minute throttling
- per-model requests-per-minute throttling
- `429` retry-after handling
- LLM call ledger
- sub-agent call budget

Do not configure the scheduler as if trial endpoints were unlimited.
