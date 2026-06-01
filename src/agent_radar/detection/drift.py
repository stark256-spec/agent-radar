"""
Behavioral drift detection — baselines each agent's normal behavior
and alerts when statistical properties shift beyond threshold.

Uses a sliding window comparison: compute metric distributions over
a baseline window (e.g., last 7 days) and compare to the current window
(e.g., last 1 hour) using Cohen's d effect size for significance.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np

from agent_radar.schema import AgentPlatform, Anomaly, AnomalyType, RiskLevel


def _cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d effect size — measures practical significance of difference."""
    if not a or not b:
        return 0.0
    na, nb = len(a), len(b)
    pooled_std = (
        np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
        if na > 1 and nb > 1
        else 1.0
    )
    if pooled_std == 0:
        return 0.0
    return abs(float(np.mean(a)) - float(np.mean(b))) / pooled_std


@dataclass
class DriftDetector:
    """
    Detect behavioral drift by comparing current window metrics to a
    rolling baseline using Cohen's d effect size.

    Effect size thresholds:
        d < 0.2  → negligible
        d < 0.5  → small (ignored)
        d < 0.8  → medium → MEDIUM severity
        d >= 0.8 → large  → HIGH severity
        d >= 1.2 → very large → CRITICAL
    """

    baseline_days: int = 7
    current_hours: int = 1
    min_samples: int = 10

    def detect(
        self,
        agent_id: str,
        agent_name: str,
        platform: AgentPlatform,
        all_events: list[Any],
        now: datetime | None = None,
    ) -> list[Anomaly]:
        now = now or datetime.now(timezone.utc)
        baseline_start = now - timedelta(days=self.baseline_days)
        current_start = now - timedelta(hours=self.current_hours)

        baseline_events = [e for e in all_events if baseline_start <= e.timestamp < current_start]
        current_events = [e for e in all_events if current_start <= e.timestamp <= now]

        if len(baseline_events) < self.min_samples or len(current_events) < 2:
            return []

        anomalies: list[Anomaly] = []
        checks: dict[str, tuple[list[float], list[float]]] = {
            "latency_ms": (
                [e.latency_ms for e in baseline_events if e.latency_ms is not None],
                [e.latency_ms for e in current_events if e.latency_ms is not None],
            ),
            "error_rate": (
                [0.0 if e.success else 1.0 for e in baseline_events],
                [0.0 if e.success else 1.0 for e in current_events],
            ),
            "tokens_per_query": (
                [(e.tokens_input or 0) + (e.tokens_output or 0) for e in baseline_events],
                [(e.tokens_input or 0) + (e.tokens_output or 0) for e in current_events],
            ),
            "cost_usd": (
                [e.cost_usd for e in baseline_events if e.cost_usd is not None],
                [e.cost_usd for e in current_events if e.cost_usd is not None],
            ),
        }

        for metric, (base_vals, curr_vals) in checks.items():
            if not base_vals or not curr_vals:
                continue
            d = _cohens_d(base_vals, curr_vals)
            if d < 0.5:
                continue

            severity = (
                RiskLevel.CRITICAL if d >= 1.2 else RiskLevel.HIGH if d >= 0.8 else RiskLevel.MEDIUM
            )
            base_mean = float(np.mean(base_vals))
            curr_mean = float(np.mean(curr_vals))

            anomalies.append(
                Anomaly(
                    anomaly_type=AnomalyType.BEHAVIORAL_DRIFT,
                    agent_id=agent_id,
                    agent_name=agent_name,
                    platform=platform,
                    severity=severity,
                    description=(
                        f"Behavioral drift detected on {metric}: "
                        f"baseline={base_mean:.2f} → current={curr_mean:.2f} "
                        f"(Cohen's d={d:.2f})"
                    ),
                    affected_metric=metric,
                    baseline_value=base_mean,
                    observed_value=curr_mean,
                    confidence=min(1.0, d / 2.0),
                    evidence={
                        "cohens_d": d,
                        "baseline_n": len(base_vals),
                        "current_n": len(curr_vals),
                        "baseline_mean": base_mean,
                        "current_mean": curr_mean,
                    },
                )
            )

        return anomalies
