"""AgentRadar FastAPI application."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm

from agent_radar.api.auth import Token, authenticate_user, create_access_token, require_viewer
from agent_radar.schema import (
    AgentEvent,
    AgentMetricsSummary,
    AgentPlatform,
    AgentRegistration,
    Anomaly,
    PolicyViolation,
)
from agent_radar.storage.models import create_engine, create_session_factory, init_db

_engine = None
_session_factory = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _session_factory
    db_url = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./agent_radar.db")
    _engine = create_engine(db_url)
    _session_factory = create_session_factory(_engine)
    await init_db(_engine)
    yield
    if _engine:
        await _engine.dispose()


app = FastAPI(
    title="AgentRadar",
    description=(
        "Open-source observability for enterprise AI agent fleets. "
        "Vendor-neutral, OpenTelemetry-native."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory stores for the default/dev mode (replace with DB queries in production)
_events: list[AgentEvent] = []
_agents: dict[str, AgentRegistration] = {}
_anomalies: list[Anomaly] = []
_violations: list[PolicyViolation] = []


@app.post("/auth/token", response_model=Token, tags=["auth"])
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token({"sub": user["username"], "role": user["role"]})
    return Token(access_token=token, token_type="bearer", role=user["role"])


@app.get("/health", tags=["system"])
async def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}


# ── Agents ──────────────────────────────────────────────────────────────────


@app.get("/agents", response_model=list[AgentRegistration], tags=["agents"])
async def list_agents(
    platform: AgentPlatform | None = None,
    active_only: bool = True,
    _user=Depends(require_viewer),
) -> list[AgentRegistration]:
    agents = list(_agents.values())
    if platform:
        agents = [a for a in agents if a.platform == platform]
    if active_only:
        agents = [a for a in agents if a.is_active]
    return agents


@app.post("/agents", response_model=AgentRegistration, tags=["agents"])
async def register_agent(
    agent: AgentRegistration,
    _user=Depends(require_viewer),
) -> AgentRegistration:
    _agents[agent.agent_id] = agent
    return agent


@app.get("/agents/{agent_id}", response_model=AgentRegistration, tags=["agents"])
async def get_agent(agent_id: str, _user=Depends(require_viewer)) -> AgentRegistration:
    if agent_id not in _agents:
        raise HTTPException(status_code=404, detail="Agent not found")
    return _agents[agent_id]


# ── Events / Metrics ─────────────────────────────────────────────────────────


@app.post("/events", response_model=AgentEvent, status_code=201, tags=["events"])
async def ingest_event(event: AgentEvent, _user=Depends(require_viewer)) -> AgentEvent:
    _events.append(event)
    # Auto-register agent if not known
    if event.agent_id not in _agents:
        _agents[event.agent_id] = AgentRegistration(
            agent_id=event.agent_id,
            agent_name=event.agent_name,
            platform=event.platform,
        )
    return event


@app.get("/events", response_model=list[AgentEvent], tags=["events"])
async def list_events(
    agent_id: str | None = None,
    platform: AgentPlatform | None = None,
    limit: int = 100,
    _user=Depends(require_viewer),
) -> list[AgentEvent]:
    evs = _events
    if agent_id:
        evs = [e for e in evs if e.agent_id == agent_id]
    if platform:
        evs = [e for e in evs if e.platform == platform]
    return evs[-limit:]


@app.get("/metrics", response_model=list[AgentMetricsSummary], tags=["metrics"])
async def get_metrics(
    agent_id: str | None = None,
    _user=Depends(require_viewer),
) -> list[AgentMetricsSummary]:
    import statistics
    from datetime import datetime, timedelta, timezone

    grouped: dict[str, list[AgentEvent]] = {}
    for e in _events:
        grouped.setdefault(e.agent_id, []).append(e)

    now = datetime.now(timezone.utc)
    summaries = []
    for aid, evs in grouped.items():
        if agent_id and aid != agent_id:
            continue
        n = len(evs)
        succeeded = sum(1 for e in evs if e.success)
        latencies = [e.latency_ms for e in evs if e.latency_ms is not None]
        summaries.append(
            AgentMetricsSummary(
                agent_id=aid,
                agent_name=evs[0].agent_name,
                platform=evs[0].platform,
                window_start=now - timedelta(hours=24),
                window_end=now,
                total_queries=n,
                successful_queries=succeeded,
                failed_queries=n - succeeded,
                success_rate=succeeded / n if n else 0.0,
                avg_latency_ms=statistics.mean(latencies) if latencies else None,
                p95_latency_ms=(
                    sorted(latencies)[int(len(latencies) * 0.95)] if latencies else None
                ),
                total_tokens_input=sum(e.tokens_input or 0 for e in evs),
                total_tokens_output=sum(e.tokens_output or 0 for e in evs),
                total_cost_usd=sum(e.cost_usd or 0.0 for e in evs),
                unique_users=len({e.user_id for e in evs if e.user_id}),
                data_sources_accessed=list({ds for e in evs for ds in e.data_sources}),
                anomaly_count=sum(1 for a in _anomalies if a.agent_id == aid),
                violation_count=sum(1 for v in _violations if v.agent_id == aid),
            )
        )
    return summaries


# ── Anomalies ────────────────────────────────────────────────────────────────


@app.get("/anomalies", response_model=list[Anomaly], tags=["anomalies"])
async def list_anomalies(
    agent_id: str | None = None,
    resolved: bool = False,
    limit: int = 100,
    _user=Depends(require_viewer),
) -> list[Anomaly]:
    result = [a for a in _anomalies if a.resolved == resolved]
    if agent_id:
        result = [a for a in result if a.agent_id == agent_id]
    return result[-limit:]


@app.post("/anomalies", response_model=Anomaly, status_code=201, tags=["anomalies"])
async def report_anomaly(anomaly: Anomaly, _user=Depends(require_viewer)) -> Anomaly:
    _anomalies.append(anomaly)
    return anomaly


@app.patch("/anomalies/{anomaly_id}/resolve", tags=["anomalies"])
async def resolve_anomaly(anomaly_id: str, _user=Depends(require_viewer)) -> dict[str, Any]:
    for a in _anomalies:
        if a.anomaly_id == anomaly_id:
            a.resolved = True
            return {"resolved": True}
    raise HTTPException(status_code=404, detail="Anomaly not found")


# ── Violations ───────────────────────────────────────────────────────────────


@app.get("/violations", response_model=list[PolicyViolation], tags=["violations"])
async def list_violations(
    agent_id: str | None = None,
    rule_name: str | None = None,
    resolved: bool = False,
    limit: int = 100,
    _user=Depends(require_viewer),
) -> list[PolicyViolation]:
    result = [v for v in _violations if v.resolved == resolved]
    if agent_id:
        result = [v for v in result if v.agent_id == agent_id]
    if rule_name:
        result = [v for v in result if v.rule_name == rule_name]
    return result[-limit:]


@app.post("/violations", response_model=PolicyViolation, status_code=201, tags=["violations"])
async def report_violation(
    violation: PolicyViolation, _user=Depends(require_viewer)
) -> PolicyViolation:
    _violations.append(violation)
    return violation


# ── Costs ────────────────────────────────────────────────────────────────────


@app.get("/costs", tags=["costs"])
async def get_costs(
    group_by: str = "agent",
    _user=Depends(require_viewer),
) -> dict[str, float]:
    if group_by == "agent":
        costs: dict[str, float] = {}
        for e in _events:
            costs[e.agent_name] = costs.get(e.agent_name, 0.0) + (e.cost_usd or 0.0)
        return costs
    if group_by == "user":
        costs = {}
        for e in _events:
            key = e.user_email or e.user_id or "unknown"
            costs[key] = costs.get(key, 0.0) + (e.cost_usd or 0.0)
        return costs
    if group_by == "department":
        costs = {}
        for e in _events:
            key = e.department or "unknown"
            costs[key] = costs.get(key, 0.0) + (e.cost_usd or 0.0)
        return costs
    raise HTTPException(status_code=400, detail="group_by must be agent|user|department")


# ── Users ────────────────────────────────────────────────────────────────────


@app.get("/users", tags=["users"])
async def get_user_activity(
    limit: int = 100,
    _user=Depends(require_viewer),
) -> list[dict[str, Any]]:
    user_map: dict[str, dict[str, Any]] = {}
    for e in _events:
        uid = e.user_email or e.user_id or "unknown"
        if uid not in user_map:
            user_map[uid] = {
                "user_id": uid,
                "department": e.department,
                "platforms": set(),
                "agents": set(),
                "query_count": 0,
                "total_cost_usd": 0.0,
                "last_seen": e.timestamp,
            }
        entry = user_map[uid]
        entry["platforms"].add(e.platform.value)
        entry["agents"].add(e.agent_name)
        entry["query_count"] += 1
        entry["total_cost_usd"] += e.cost_usd or 0.0
        if e.timestamp > entry["last_seen"]:
            entry["last_seen"] = e.timestamp

    result = []
    for entry in list(user_map.values())[:limit]:
        result.append(
            {
                **entry,
                "platforms": list(entry["platforms"]),
                "agents": list(entry["agents"]),
                "last_seen": entry["last_seen"].isoformat(),
            }
        )
    return result


def run() -> None:
    uvicorn.run(
        "agent_radar.api.main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8080")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )
