"""Tests for the policy violation rules engine."""

from datetime import datetime, timezone

from agent_radar.policy.engine import PolicyEngine
from agent_radar.schema import AgentEvent, AgentEventType, AgentPlatform, RiskLevel

_RULES_YAML = """
rules:
  - name: sensitive_data_access
    description: Non-HR user accessing HR data
    severity: high
    condition:
      data_source_contains: "HR"
    action: alert

  - name: excessive_cost
    description: Single query cost > $1
    severity: critical
    condition:
      cost_usd_gt: 1.0
    action: alert

  - name: after_hours_finance
    description: Financial agent used outside 8-18
    severity: medium
    condition:
      hour_outside: [8, 18]
      agent_name_contains: "finance"
    action: alert

  - name: disabled_rule
    description: Should never fire
    severity: low
    enabled: false
    condition:
      success_is: true
    action: alert

  - name: high_latency
    description: Latency > 5000ms
    severity: medium
    condition:
      latency_ms_gt: 5000
    action: alert
"""


def _engine() -> PolicyEngine:
    import yaml

    return PolicyEngine.from_dict(yaml.safe_load(_RULES_YAML))


def _event(**kwargs) -> AgentEvent:
    defaults = dict(
        platform=AgentPlatform.GLEAN,
        event_type=AgentEventType.QUERY,
        agent_id="glean-1",
        agent_name="Glean Search",
        success=True,
        timestamp=datetime(2025, 6, 1, 10, 0, tzinfo=timezone.utc),
    )
    defaults.update(kwargs)
    return AgentEvent(**defaults)


def test_no_violation_on_clean_event():
    engine = _engine()
    violations = engine.evaluate(_event())
    assert violations == []


def test_sensitive_data_access_fires():
    engine = _engine()
    event = _event(data_sources=["HR-employees-db", "general-db"])
    violations = engine.evaluate(event)
    assert any(v.rule_name == "sensitive_data_access" for v in violations)
    v = next(v for v in violations if v.rule_name == "sensitive_data_access")
    assert v.severity == RiskLevel.HIGH


def test_excessive_cost_fires():
    engine = _engine()
    event = _event(cost_usd=2.50)
    violations = engine.evaluate(event)
    assert any(v.rule_name == "excessive_cost" for v in violations)


def test_after_hours_finance_fires():
    engine = _engine()
    event = _event(
        agent_name="finance-bot",
        timestamp=datetime(2025, 6, 1, 22, 0, tzinfo=timezone.utc),  # 10 PM
    )
    violations = engine.evaluate(event)
    assert any(v.rule_name == "after_hours_finance" for v in violations)


def test_after_hours_finance_no_fire_during_hours():
    engine = _engine()
    event = _event(
        agent_name="finance-bot",
        timestamp=datetime(2025, 6, 1, 14, 0, tzinfo=timezone.utc),  # 2 PM
    )
    violations = engine.evaluate(event)
    assert not any(v.rule_name == "after_hours_finance" for v in violations)


def test_disabled_rule_does_not_fire():
    engine = _engine()
    event = _event()
    violations = engine.evaluate(event)
    assert not any(v.rule_name == "disabled_rule" for v in violations)


def test_high_latency_fires():
    engine = _engine()
    event = _event(latency_ms=8000.0)
    violations = engine.evaluate(event)
    assert any(v.rule_name == "high_latency" for v in violations)


def test_multiple_rules_fire_simultaneously():
    engine = _engine()
    event = _event(
        data_sources=["HR-db"],
        cost_usd=5.0,
        latency_ms=9000.0,
    )
    violations = engine.evaluate(event)
    rule_names = {v.rule_name for v in violations}
    assert "sensitive_data_access" in rule_names
    assert "excessive_cost" in rule_names
    assert "high_latency" in rule_names


def test_evaluate_batch():
    engine = _engine()
    events = [
        _event(data_sources=["HR-db"]),
        _event(cost_usd=3.0),
        _event(),  # clean
    ]
    violations = engine.evaluate_batch(events)
    assert len(violations) == 2


def test_rule_count():
    engine = _engine()
    assert engine.rule_count == 4  # 5 rules minus 1 disabled


def test_custom_evaluator():
    engine = _engine()
    engine.register_evaluator(
        "department_is_engineering",
        lambda val, ev: ev.department == "engineering",
    )
    # Custom evaluator registered; no test rule uses it but it shouldn't break anything
    assert engine.evaluate(_event()) == []
