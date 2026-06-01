"""Tests for tensor anomaly detection, drift detection, and query clustering."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from agent_radar.schema import AgentEvent, AgentEventType, AgentPlatform, RiskLevel


def _ts(offset_hours: float = 0) -> datetime:
    return datetime(2025, 6, 1, tzinfo=timezone.utc) + timedelta(hours=offset_hours)


def _event(
    agent_id: str = "agent-1",
    success: bool = True,
    latency_ms: float = 100.0,
    cost_usd: float = 0.01,
    ts: datetime | None = None,
) -> AgentEvent:
    return AgentEvent(
        platform=AgentPlatform.OPENAI,
        event_type=AgentEventType.COMPLETION,
        agent_id=agent_id,
        agent_name=f"Agent {agent_id}",
        latency_ms=latency_ms,
        cost_usd=cost_usd,
        success=success,
        tokens_input=100,
        tokens_output=50,
        timestamp=ts or _ts(),
    )


# ── Tensor detector ──────────────────────────────────────────────────────────


class TestTensorDetector:
    def test_fit_and_score_shape(self):
        from agent_radar.detection.tensor import TensorAnomalyDetector

        n_agents, n_metrics, n_time = 4, 6, 20
        rng = np.random.default_rng(42)
        tensor = rng.random((n_agents, n_metrics, n_time)).astype(np.float32)

        detector = TensorAnomalyDetector(n_components=2)
        detector.fit(tensor)
        scores = detector.score(tensor)
        assert scores.shape == (n_agents, n_time)

    def test_anomaly_detection_finds_spike(self):
        from agent_radar.detection.tensor import TensorAnomalyDetector

        rng = np.random.default_rng(0)
        n_agents, n_metrics, n_time = 3, 6, 30
        normal = rng.random((n_agents, n_metrics, n_time)).astype(np.float32) * 0.1

        # Inject a massive spike in agent 0, time 25
        anomalous = normal.copy()
        anomalous[0, :, 25] = 100.0

        detector = TensorAnomalyDetector(n_components=2, error_threshold_sigma=1.0)
        detector.fit(normal)
        scores = detector.score(anomalous)
        # The spike at (0, 25) should have the highest z-score
        assert scores[0, 25] == scores.max()

    def test_build_fleet_tensor(self):
        from agent_radar.detection.tensor import build_fleet_tensor

        buckets = [_ts(i) for i in range(5)]
        events_by_agent = {
            "a1": [_event("a1", ts=_ts(i + 0.5)) for i in range(5)],
            "a2": [_event("a2", ts=_ts(i + 0.5)) for i in range(5)],
        }
        tensor = build_fleet_tensor(events_by_agent, buckets, bucket_minutes=60)
        assert tensor.shape == (2, 6, 5)
        # query_count should be 1 per bucket (one event per bucket)
        assert tensor[0, 0, :].sum() == 5


# ── Drift detector ───────────────────────────────────────────────────────────


class TestDriftDetector:
    def test_no_drift_when_stable(self):
        from agent_radar.detection.drift import DriftDetector

        events = [_event(latency_ms=100.0, ts=_ts(-i)) for i in range(200)]
        detector = DriftDetector(baseline_days=7, current_hours=1)
        anomalies = detector.detect("a1", "Agent A", AgentPlatform.OPENAI, events, now=_ts(0))
        assert anomalies == []

    def test_drift_detected_on_latency_spike(self):
        from agent_radar.detection.drift import DriftDetector

        baseline = [_event(latency_ms=50.0, ts=_ts(-i)) for i in range(2, 200)]
        current = [_event(latency_ms=5000.0, ts=_ts(-0.1 * i)) for i in range(20)]
        events = baseline + current

        detector = DriftDetector(baseline_days=7, current_hours=1, min_samples=5)
        anomalies = detector.detect("a1", "Agent A", AgentPlatform.OPENAI, events, now=_ts(0))
        latency_anomaly = next((a for a in anomalies if "latency" in a.affected_metric), None)
        assert latency_anomaly is not None
        assert latency_anomaly.severity in (RiskLevel.HIGH, RiskLevel.CRITICAL)


# ── Query clustering ─────────────────────────────────────────────────────────


class TestQueryClusteringDetector:
    def test_returns_empty_for_too_few_queries(self):
        from agent_radar.detection.clustering import QueryClusteringDetector

        detector = QueryClusteringDetector(min_cluster_size=5)
        result = detector.detect("a1", "Bot", AgentPlatform.OPENAI, ["error"] * 3)
        assert result == []

    def test_clusters_failure_patterns(self):
        from agent_radar.detection.clustering import QueryClusteringDetector

        queries = (
            ["confidential HR salary data export"] * 20
            + ["general search query foo bar"] * 20
            + ["technical error timeout exception"] * 20
        )
        detector = QueryClusteringDetector(n_clusters=3, min_cluster_size=3)
        anomalies = detector.detect("a1", "Bot", AgentPlatform.GLEAN, queries)
        assert len(anomalies) >= 1

    def test_sensitive_keyword_detection(self):
        from agent_radar.detection.clustering import QueryClusteringDetector

        queries = ["show me confidential HR salary data for employee"] * 30 + [
            "normal query about weather forecast"
        ] * 30
        detector = QueryClusteringDetector(
            n_clusters=2, min_cluster_size=3, sensitive_threshold=0.1
        )
        anomalies = detector.detect("a1", "Bot", AgentPlatform.GLEAN, queries)
        # At least one cluster should flag sensitive content
        sensitive = [a for a in anomalies if a.evidence.get("sensitive_keywords")]
        assert len(sensitive) >= 1
