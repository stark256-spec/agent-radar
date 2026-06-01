"""
Policy violation rules engine — evaluates AgentEvents against YAML-defined rules.

Rules file format:

    rules:
      - name: sensitive_data_access
        description: Non-HR users accessing HR data sources
        severity: high
        condition:
          data_source_contains: "HR"
          user_role_not_in: ["HR", "admin"]
        action: alert

      - name: after_hours_financial
        description: Financial agents used outside business hours
        severity: medium
        condition:
          hour_outside: [8, 18]
          agent_name_contains: "financial"
        action: alert

      - name: excessive_cost
        description: Single query costing more than $1
        severity: high
        condition:
          cost_usd_gt: 1.0
        action: alert

Each condition key maps to a built-in evaluator function. Custom evaluators
can be registered via PolicyEngine.register_evaluator().
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from agent_radar.schema import AgentEvent, PolicyViolation, RiskLevel


@dataclass
class PolicyRule:
    name: str
    description: str
    severity: RiskLevel
    conditions: dict[str, Any]
    action: str = "alert"
    enabled: bool = True


def _load_rules(path: Path) -> list[PolicyRule]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    rules = []
    for r in data.get("rules", []):
        try:
            severity = RiskLevel(r.get("severity", "medium").lower())
        except ValueError:
            severity = RiskLevel.MEDIUM
        rules.append(
            PolicyRule(
                name=r["name"],
                description=r.get("description", ""),
                severity=severity,
                conditions=r.get("condition", {}),
                action=r.get("action", "alert"),
                enabled=r.get("enabled", True),
            )
        )
    return rules


def _load_rules_from_dict(data: dict[str, Any]) -> list[PolicyRule]:
    rules = []
    for r in data.get("rules", []):
        try:
            severity = RiskLevel(r.get("severity", "medium").lower())
        except ValueError:
            severity = RiskLevel.MEDIUM
        rules.append(
            PolicyRule(
                name=r["name"],
                description=r.get("description", ""),
                severity=severity,
                conditions=r.get("condition", {}),
                action=r.get("action", "alert"),
                enabled=r.get("enabled", True),
            )
        )
    return rules


# Built-in condition evaluators
# Each receives (condition_value, event) and returns True if the rule triggers.
_BUILTIN_EVALUATORS: dict[str, Callable[[Any, AgentEvent], bool]] = {
    "data_source_contains": lambda val, ev: any(
        val.lower() in ds.lower() for ds in ev.data_sources
    ),
    "user_role_not_in": lambda val, ev: True,  # requires external role lookup; always passes
    "agent_name_contains": lambda val, ev: val.lower() in ev.agent_name.lower(),
    "hour_outside": lambda val, ev: not (val[0] <= ev.timestamp.hour < val[1]),
    "cost_usd_gt": lambda val, ev: ev.cost_usd is not None and ev.cost_usd > val,
    "tokens_gt": lambda val, ev: (ev.tokens_input or 0) + (ev.tokens_output or 0) > val,
    "platform_is": lambda val, ev: ev.platform.value == val,
    "success_is": lambda val, ev: ev.success == val,
    "department_is": lambda val, ev: (
        ev.department is not None and ev.department.lower() == val.lower()
    ),
    "query_contains": lambda val, ev: (
        ev.query_text is not None and val.lower() in ev.query_text.lower()
    ),
    "query_matches_regex": lambda val, ev: (
        ev.query_text is not None and bool(re.search(val, ev.query_text, re.IGNORECASE))
    ),
    "latency_ms_gt": lambda val, ev: ev.latency_ms is not None and ev.latency_ms > val,
}


class PolicyEngine:
    """
    Evaluate AgentEvents against a set of PolicyRules and emit PolicyViolations.

    Usage:
        engine = PolicyEngine.from_yaml("rules.yaml")
        violations = engine.evaluate(event)
    """

    def __init__(self, rules: list[PolicyRule]) -> None:
        self._rules = rules
        self._evaluators = dict(_BUILTIN_EVALUATORS)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "PolicyEngine":
        return cls(_load_rules(Path(path)))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PolicyEngine":
        return cls(_load_rules_from_dict(data))

    def register_evaluator(self, key: str, fn: Callable[[Any, AgentEvent], bool]) -> None:
        """Register a custom condition evaluator."""
        self._evaluators[key] = fn

    def _rule_matches(self, rule: PolicyRule, event: AgentEvent) -> bool:
        """Return True only if ALL conditions in the rule match."""
        for key, value in rule.conditions.items():
            evaluator = self._evaluators.get(key)
            if evaluator is None:
                continue
            if not evaluator(value, event):
                return False
        return bool(rule.conditions)

    def evaluate(self, event: AgentEvent) -> list[PolicyViolation]:
        """Evaluate a single event against all enabled rules."""
        violations = []
        for rule in self._rules:
            if not rule.enabled:
                continue
            if self._rule_matches(rule, event):
                violations.append(
                    PolicyViolation(
                        rule_name=rule.name,
                        agent_id=event.agent_id,
                        agent_name=event.agent_name,
                        user_id=event.user_id,
                        user_email=event.user_email,
                        platform=event.platform,
                        severity=rule.severity,
                        description=rule.description,
                        event_id=event.event_id,
                        action_taken=rule.action,
                    )
                )
        return violations

    def evaluate_batch(self, events: list[AgentEvent]) -> list[PolicyViolation]:
        """Evaluate a list of events, returning all violations."""
        violations = []
        for event in events:
            violations.extend(self.evaluate(event))
        return violations

    @property
    def rule_count(self) -> int:
        return sum(1 for r in self._rules if r.enabled)
