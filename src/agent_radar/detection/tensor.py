"""
Tensor decomposition anomaly detection for AI agent fleets.

Represents the agent fleet as a 3-way tensor:
    T[agent_idx, metric_idx, time_idx]  shape=(A, M, T)

Applies CP (PARAFAC) decomposition via TensorLy to extract latent
anomaly patterns invisible in individual metric streams. Reconstruction
error per agent-time slice signals anomalies with high sensitivity to
correlated multi-metric deviations.

This is the novel contribution: applying tensor decomposition — standard
in multivariate systems monitoring — to AI agent fleet observability,
where the "sensors" are latency, error rate, token count, and cost.

Reference: Kolda & Bader (2009), "Tensor Decompositions and Applications"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from agent_radar.schema import Anomaly, AnomalyType, RiskLevel

try:
    import tensorly as tl
    from tensorly.decomposition import parafac

    _TENSORLY_AVAILABLE = True
except ImportError:
    _TENSORLY_AVAILABLE = False


# Metrics tracked per agent per time bucket
METRIC_NAMES = [
    "query_count",
    "error_rate",
    "avg_latency_ms",
    "tokens_per_query",
    "cost_usd",
    "unique_users",
]
N_METRICS = len(METRIC_NAMES)


@dataclass
class TensorAnomalyDetector:
    """
    Fit a CP decomposition baseline on historical agent fleet data,
    then score new observations against the reconstruction error.

    Usage:
        detector = TensorAnomalyDetector(n_components=4)
        detector.fit(historical_tensor)        # shape (A, M, T_train)
        anomalies = detector.detect(new_tensor) # shape (A, M, T_test)
    """

    n_components: int = 4
    error_threshold_sigma: float = 3.0
    _factors: list | None = field(default=None, repr=False)
    _baseline_mean: float = 0.0
    _baseline_std: float = 1.0

    def fit(self, tensor: np.ndarray) -> "TensorAnomalyDetector":
        """
        Fit CP decomposition on a (agents × metrics × time) tensor.

        Stores the factor matrices and computes baseline reconstruction
        error statistics for anomaly thresholding.
        """
        if not _TENSORLY_AVAILABLE:
            raise ImportError("tensorly is required: pip install tensorly")

        tl.set_backend("numpy")
        weights, factors = parafac(tensor, rank=self.n_components, n_iter_max=200, tol=1e-8)
        self._factors = factors

        reconstructed = tl.cp_to_tensor((weights, factors))
        errors = np.abs(tensor - reconstructed).mean(axis=1)  # shape (agents, time)
        self._baseline_mean = float(errors.mean())
        self._baseline_std = float(errors.std()) or 1.0
        return self

    def score(self, tensor: np.ndarray) -> np.ndarray:
        """
        Compute per-agent-per-timestep z-score anomaly scores.

        Returns array of shape (agents, time) where high values indicate
        anomalous behavior relative to the fitted baseline.
        """
        if not _TENSORLY_AVAILABLE:
            raise ImportError("tensorly is required: pip install tensorly")
        if self._factors is None:
            raise RuntimeError("Call fit() before score()")

        tl.set_backend("numpy")
        n_agents, n_metrics, n_time = tensor.shape

        # Project onto fitted factor space (inference without refitting)
        projected = self._project(tensor)
        errors = np.abs(tensor - projected).mean(axis=1)  # (agents, time)
        z_scores = (errors - self._baseline_mean) / self._baseline_std
        return z_scores

    def detect(
        self,
        tensor: np.ndarray,
        agent_ids: list[str],
        agent_names: list[str],
        timestamps: list[datetime],
        platform: str = "unknown",
    ) -> list[Anomaly]:
        """
        Score a tensor and return Anomaly objects for all exceeded thresholds.
        """
        from agent_radar.schema import AgentPlatform

        z_scores = self.score(tensor)
        anomalies: list[Anomaly] = []
        n_agents, n_time = z_scores.shape

        for a_idx in range(min(n_agents, len(agent_ids))):
            for t_idx in range(min(n_time, len(timestamps))):
                z = float(z_scores[a_idx, t_idx])
                if z < self.error_threshold_sigma:
                    continue

                severity = (
                    RiskLevel.CRITICAL
                    if z > 5.0
                    else RiskLevel.HIGH
                    if z > 4.0
                    else RiskLevel.MEDIUM
                )
                # Identify which metric drove the anomaly
                per_metric_err = np.abs(
                    tensor[a_idx, :, t_idx] - self._project(tensor)[a_idx, :, t_idx]
                )
                top_metric_idx = int(per_metric_err.argmax())
                top_metric = (
                    METRIC_NAMES[top_metric_idx] if top_metric_idx < N_METRICS else "unknown"
                )

                try:
                    plat = AgentPlatform(platform)
                except ValueError:
                    plat = AgentPlatform.CUSTOM

                anomalies.append(
                    Anomaly(
                        anomaly_type=AnomalyType.TENSOR_DECOMPOSITION,
                        agent_id=agent_ids[a_idx],
                        agent_name=agent_names[a_idx],
                        platform=plat,
                        severity=severity,
                        detected_at=timestamps[t_idx],
                        description=(
                            f"Tensor decomposition detected multi-metric anomaly "
                            f"(z={z:.2f}σ, driven by {top_metric})"
                        ),
                        affected_metric=top_metric,
                        observed_value=z,
                        baseline_value=self.error_threshold_sigma,
                        confidence=min(1.0, (z - self.error_threshold_sigma) / 3.0),
                        evidence={
                            "z_score": z,
                            "top_metric": top_metric,
                            "per_metric_errors": dict(
                                zip(METRIC_NAMES[:N_METRICS], per_metric_err.tolist())
                            ),
                        },
                    )
                )

        return anomalies

    def _project(self, tensor: np.ndarray) -> np.ndarray:
        """Reconstruct tensor from the fitted factor matrices (inference mode)."""
        if not _TENSORLY_AVAILABLE or self._factors is None:
            raise RuntimeError("Not fitted")
        n_agents, n_metrics, n_time = tensor.shape
        A_fit, M_fit, T_fit = [f.shape[0] for f in self._factors]

        # Slice/pad agent axis to match new tensor shape
        A = self._factors[0][:n_agents] if n_agents <= A_fit else self._factors[0]
        M = self._factors[1]
        T = self._factors[2][:n_time] if n_time <= T_fit else self._factors[2]

        # Khatri-Rao product reconstruction
        rank = A.shape[1]
        result = np.zeros((A.shape[0], M.shape[0], T.shape[0]))
        for r in range(rank):
            result += np.einsum("i,j,k->ijk", A[:, r], M[:, r], T[:, r])
        return result


def build_fleet_tensor(
    events_by_agent: dict[str, list],
    time_buckets: list[datetime],
    bucket_minutes: int = 60,
) -> np.ndarray:
    """
    Build a (agents × metrics × time) numpy tensor from raw AgentEvent lists.

    Args:
        events_by_agent: Dict mapping agent_id → list[AgentEvent]
        time_buckets: List of bucket start times (must be evenly spaced)
        bucket_minutes: Width of each time bucket

    Returns:
        np.ndarray of shape (n_agents, N_METRICS, n_time_buckets)
    """
    from datetime import timedelta

    agent_ids = sorted(events_by_agent.keys())
    n_agents = len(agent_ids)
    n_time = len(time_buckets)
    tensor = np.zeros((n_agents, N_METRICS, n_time), dtype=np.float32)

    bucket_width = timedelta(minutes=bucket_minutes)

    for a_idx, agent_id in enumerate(agent_ids):
        events = events_by_agent[agent_id]
        for t_idx, bucket_start in enumerate(time_buckets):
            bucket_end = bucket_start + bucket_width
            bucket_events = [e for e in events if bucket_start <= e.timestamp < bucket_end]
            if not bucket_events:
                continue

            n = len(bucket_events)
            errors = sum(1 for e in bucket_events if not e.success)
            latencies = [e.latency_ms for e in bucket_events if e.latency_ms is not None]
            tokens = [(e.tokens_input or 0) + (e.tokens_output or 0) for e in bucket_events]
            cost = sum(e.cost_usd or 0.0 for e in bucket_events)
            users = len({e.user_id for e in bucket_events if e.user_id})

            tensor[a_idx, 0, t_idx] = n  # query_count
            tensor[a_idx, 1, t_idx] = errors / n  # error_rate
            tensor[a_idx, 2, t_idx] = float(np.mean(latencies)) if latencies else 0.0
            tensor[a_idx, 3, t_idx] = float(np.mean(tokens)) if tokens else 0.0
            tensor[a_idx, 4, t_idx] = cost
            tensor[a_idx, 5, t_idx] = users

    return tensor
