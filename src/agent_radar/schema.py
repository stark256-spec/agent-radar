"""Core Pydantic models — the canonical schema for AgentRadar events."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class AgentPlatform(str, Enum):
    COPILOT_GITHUB = "copilot_github"
    COPILOT_M365 = "copilot_m365"
    GLEAN = "glean"
    WINDSURF = "windsurf"
    OPENAI = "openai"
    BEDROCK = "bedrock"
    CUSTOM = "custom"


class AgentEventType(str, Enum):
    QUERY = "query"
    COMPLETION = "completion"
    TOOL_CALL = "tool_call"
    ERROR = "error"
    POLICY_VIOLATION = "policy_violation"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentEvent(BaseModel):
    """A single observable event from an AI agent."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    platform: AgentPlatform
    event_type: AgentEventType
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Identity
    agent_id: str
    agent_name: str
    user_id: str | None = None
    user_email: str | None = None
    department: str | None = None
    tenant_id: str | None = None

    # Payload
    query_text: str | None = None
    response_text: str | None = None
    data_sources: list[str] = Field(default_factory=list)

    # Performance
    latency_ms: float | None = None
    tokens_input: int | None = None
    tokens_output: int | None = None
    model: str | None = None
    success: bool = True
    error_message: str | None = None

    # Cost
    cost_usd: float | None = None

    # Raw payload preserved
    raw: dict[str, Any] = Field(default_factory=dict)

    def to_otel_attributes(self) -> dict[str, Any]:
        attrs: dict[str, Any] = {
            "agent.event_id": self.event_id,
            "agent.platform": self.platform.value,
            "agent.event_type": self.event_type.value,
            "agent.id": self.agent_id,
            "agent.name": self.agent_name,
            "agent.success": self.success,
        }
        if self.user_id:
            attrs["agent.user_id"] = self.user_id
        if self.user_email:
            attrs["agent.user_email"] = self.user_email
        if self.department:
            attrs["agent.department"] = self.department
        if self.latency_ms is not None:
            attrs["agent.latency_ms"] = self.latency_ms
        if self.tokens_input is not None:
            attrs["agent.tokens_input"] = self.tokens_input
        if self.tokens_output is not None:
            attrs["agent.tokens_output"] = self.tokens_output
        if self.model:
            attrs["agent.model"] = self.model
        if self.cost_usd is not None:
            attrs["agent.cost_usd"] = self.cost_usd
        if self.data_sources:
            attrs["agent.data_sources"] = self.data_sources
        return attrs


class AgentRegistration(BaseModel):
    """Registry entry for a known AI agent in the fleet."""

    agent_id: str
    agent_name: str
    platform: AgentPlatform
    description: str | None = None
    owner_team: str | None = None
    data_sensitivity: RiskLevel = RiskLevel.LOW
    tags: list[str] = Field(default_factory=list)
    registered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_seen: datetime | None = None
    is_active: bool = True


class AnomalyType(str, Enum):
    LATENCY_SPIKE = "latency_spike"
    ERROR_RATE_SURGE = "error_rate_surge"
    UNUSUAL_DATA_ACCESS = "unusual_data_access"
    BEHAVIORAL_DRIFT = "behavioral_drift"
    TENSOR_DECOMPOSITION = "tensor_decomposition"
    COST_ANOMALY = "cost_anomaly"
    QUERY_CLUSTER_FAILURE = "query_cluster_failure"


class Anomaly(BaseModel):
    """A detected anomaly in agent behavior."""

    anomaly_id: str = Field(default_factory=lambda: str(uuid4()))
    anomaly_type: AnomalyType
    agent_id: str
    agent_name: str
    platform: AgentPlatform
    severity: RiskLevel
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    description: str
    affected_metric: str | None = None
    baseline_value: float | None = None
    observed_value: float | None = None
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    evidence: dict[str, Any] = Field(default_factory=dict)
    resolved: bool = False


class PolicyViolation(BaseModel):
    """A policy rule breach detected in an agent event."""

    violation_id: str = Field(default_factory=lambda: str(uuid4()))
    rule_name: str
    agent_id: str
    agent_name: str
    user_id: str | None = None
    user_email: str | None = None
    platform: AgentPlatform
    severity: RiskLevel
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    description: str
    event_id: str
    action_taken: str = "alert"
    resolved: bool = False


class AgentMetricsSummary(BaseModel):
    """Aggregated metrics for an agent over a time window."""

    agent_id: str
    agent_name: str
    platform: AgentPlatform
    window_start: datetime
    window_end: datetime

    total_queries: int = 0
    successful_queries: int = 0
    failed_queries: int = 0
    success_rate: float = 0.0

    avg_latency_ms: float | None = None
    p95_latency_ms: float | None = None
    p99_latency_ms: float | None = None

    total_tokens_input: int = 0
    total_tokens_output: int = 0
    total_cost_usd: float = 0.0

    unique_users: int = 0
    data_sources_accessed: list[str] = Field(default_factory=list)

    anomaly_count: int = 0
    violation_count: int = 0
