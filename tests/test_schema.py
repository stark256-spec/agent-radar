"""Tests for AgentEvent and related schema models."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from agent_radar.schema import (
    AgentEvent,
    AgentEventType,
    AgentMetricsSummary,
    AgentPlatform,
    Anomaly,
    AnomalyType,
    PolicyViolation,
    RiskLevel,
)


def _event(**kwargs) -> AgentEvent:
    defaults = dict(
        platform=AgentPlatform.OPENAI,
        event_type=AgentEventType.COMPLETION,
        agent_id="openai-prod",
        agent_name="OpenAI GPT-4o",
        user_id="alice@example.com",
        tokens_input=500,
        tokens_output=200,
        latency_ms=1200.0,
        cost_usd=0.012,
        success=True,
    )
    defaults.update(kwargs)
    return AgentEvent(**defaults)


def test_event_defaults():
    event = _event()
    assert event.event_id is not None
    assert event.timestamp is not None
    assert event.success is True


def test_event_to_otel_attributes():
    event = _event(department="engineering")
    attrs = event.to_otel_attributes()
    assert attrs["agent.platform"] == "openai"
    assert attrs["agent.id"] == "openai-prod"
    assert attrs["agent.latency_ms"] == 1200.0
    assert attrs["agent.cost_usd"] == 0.012
    assert attrs["agent.department"] == "engineering"


def test_event_otel_omits_none():
    event = _event(department=None, model=None)
    attrs = event.to_otel_attributes()
    assert "agent.department" not in attrs
    assert "agent.model" not in attrs


def test_event_data_sources():
    event = _event(data_sources=["hr-db", "finance-db"])
    attrs = event.to_otel_attributes()
    assert "agent.data_sources" in attrs


def test_anomaly_confidence_validation():
    with pytest.raises(ValidationError):
        Anomaly(
            anomaly_type=AnomalyType.LATENCY_SPIKE,
            agent_id="a1",
            agent_name="MyAgent",
            platform=AgentPlatform.OPENAI,
            severity=RiskLevel.HIGH,
            description="test",
            confidence=1.5,  # > 1.0 should fail
        )


def test_policy_violation_roundtrip():
    v = PolicyViolation(
        rule_name="sensitive_data_access",
        agent_id="a1",
        agent_name="HR Bot",
        platform=AgentPlatform.GLEAN,
        severity=RiskLevel.HIGH,
        description="Non-HR user accessed HR data",
        event_id="evt-001",
    )
    json_str = v.model_dump_json()
    loaded = PolicyViolation.model_validate_json(json_str)
    assert loaded.rule_name == "sensitive_data_access"
    assert loaded.severity == RiskLevel.HIGH


def test_metrics_summary_defaults():
    s = AgentMetricsSummary(
        agent_id="a1",
        agent_name="Bot",
        platform=AgentPlatform.COPILOT_GITHUB,
        window_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        window_end=datetime(2025, 1, 2, tzinfo=timezone.utc),
    )
    assert s.total_queries == 0
    assert s.success_rate == 0.0
