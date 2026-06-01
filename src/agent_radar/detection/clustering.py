"""
Query failure clustering — NLP clustering of failed agent queries to surface
systemic failure patterns invisible in individual error logs.

Pipeline:
  1. TF-IDF vectorize failed query texts
  2. KMeans clustering
  3. Extract cluster centroids as representative failure patterns
  4. Score each cluster for data-access boundary violations via keyword heuristics

Example insight: "15% of failures contain 'confidential' or 'HR data' —
possible data access boundary issue" → PolicyViolation candidate.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_radar.schema import AgentPlatform, Anomaly, AnomalyType, RiskLevel

_SENSITIVE_KEYWORDS = [
    "confidential",
    "hr",
    "salary",
    "payroll",
    "ssn",
    "social security",
    "password",
    "credential",
    "secret",
    "private",
    "restricted",
    "medical",
    "hipaa",
    "gdpr",
    "pii",
    "personally identifiable",
]


@dataclass
class QueryClusteringDetector:
    """
    Cluster failed queries using TF-IDF + KMeans and flag clusters that
    suggest systemic issues (data access violations, prompt injection, etc.).

    Requires: scikit-learn
    """

    n_clusters: int = 8
    min_cluster_size: int = 3
    sensitive_threshold: float = 0.2

    def detect(
        self,
        agent_id: str,
        agent_name: str,
        platform: AgentPlatform,
        failed_queries: list[str],
    ) -> list[Anomaly]:
        """
        Cluster failed_queries and return Anomaly objects for significant clusters.
        Returns empty list if fewer than min_cluster_size * 2 samples.
        """
        if len(failed_queries) < self.min_cluster_size * 2:
            return []

        try:
            from sklearn.cluster import KMeans
            from sklearn.feature_extraction.text import TfidfVectorizer
        except ImportError as exc:
            raise ImportError("scikit-learn is required: pip install scikit-learn") from exc

        vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
        )
        X = vectorizer.fit_transform(failed_queries)
        k = min(self.n_clusters, len(failed_queries) // self.min_cluster_size)
        if k < 2:
            return []

        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = kmeans.fit_predict(X)
        feature_names = vectorizer.get_feature_names_out()
        anomalies: list[Anomaly] = []

        for cluster_id in range(k):
            cluster_mask = labels == cluster_id
            cluster_queries = [q for q, m in zip(failed_queries, cluster_mask) if m]
            if len(cluster_queries) < self.min_cluster_size:
                continue

            # Top terms for this cluster centroid
            centroid = kmeans.cluster_centers_[cluster_id]
            top_indices = centroid.argsort()[-5:][::-1]
            top_terms = [feature_names[i] for i in top_indices]

            # Check for sensitive keyword presence
            all_text = " ".join(cluster_queries).lower()
            sensitive_hits = [kw for kw in _SENSITIVE_KEYWORDS if kw in all_text]
            sensitive_ratio = len(
                [q for q in cluster_queries if any(kw in q.lower() for kw in _SENSITIVE_KEYWORDS)]
            ) / len(cluster_queries)

            severity = (
                RiskLevel.HIGH
                if sensitive_ratio > self.sensitive_threshold
                else RiskLevel.MEDIUM
                if len(cluster_queries) > len(failed_queries) * 0.2
                else RiskLevel.LOW
            )

            pct = len(cluster_queries) / len(failed_queries) * 100
            sensitive_note = (
                f" Sensitive keywords detected: {sensitive_hits}." if sensitive_hits else ""
            )

            anomalies.append(
                Anomaly(
                    anomaly_type=AnomalyType.QUERY_CLUSTER_FAILURE,
                    agent_id=agent_id,
                    agent_name=agent_name,
                    platform=platform,
                    severity=severity,
                    description=(
                        f"Failure cluster #{cluster_id}: {len(cluster_queries)} queries "
                        f"({pct:.0f}% of failures). Top terms: {', '.join(top_terms)}."
                        f"{sensitive_note}"
                    ),
                    affected_metric="failed_queries",
                    observed_value=float(len(cluster_queries)),
                    confidence=min(1.0, len(cluster_queries) / len(failed_queries) * 3),
                    evidence={
                        "cluster_id": cluster_id,
                        "cluster_size": len(cluster_queries),
                        "pct_of_failures": pct,
                        "top_terms": top_terms,
                        "sensitive_keywords": sensitive_hits,
                        "sensitive_ratio": sensitive_ratio,
                        "sample_queries": cluster_queries[:3],
                    },
                )
            )

        return anomalies
