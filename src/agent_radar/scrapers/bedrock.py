"""
AWS Bedrock scraper — collects model invocation metrics from CloudWatch
and the Bedrock model invocation logging API.

Auth: AWS credentials (boto3 default credential chain).
Requires: AmazonBedrockReadOnly + CloudWatchReadOnlyAccess.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from agent_radar.schema import AgentEvent, AgentEventType, AgentPlatform
from agent_radar.scrapers.base import BaseScraper


class BedrockScraper(BaseScraper):
    """Collect AWS Bedrock invocation metrics via CloudWatch and Bedrock logs.

    Uses boto3 — install with: pip install boto3
    """

    platform_name = "bedrock"

    def __init__(
        self,
        region: str = "us-east-1",
        *,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        account_id: str | None = None,
    ) -> None:
        self._region = region
        self._account_id = account_id or "unknown"
        self._aws_key = aws_access_key_id
        self._aws_secret = aws_secret_access_key

    def _boto_session(self) -> Any:
        try:
            import boto3
        except ImportError as exc:
            raise ImportError("boto3 is required: pip install boto3") from exc

        kwargs: dict[str, Any] = {"region_name": self._region}
        if self._aws_key:
            kwargs["aws_access_key_id"] = self._aws_key
            kwargs["aws_secret_access_key"] = self._aws_secret
        return boto3.Session(**kwargs)

    async def scrape(
        self,
        *,
        since: datetime | None = None,
        limit: int = 500,
    ) -> list[AgentEvent]:
        """
        Query CloudWatch for Bedrock InvocationLatency and TokenCount metrics.

        Returns one AgentEvent per model-per-day aggregate. For per-request
        logs, enable Bedrock model invocation logging to CloudWatch Logs and
        extend this scraper to parse those log groups.
        """
        import asyncio
        from functools import partial

        session = self._boto_session()
        cw = session.client("cloudwatch")
        start = since or (datetime.now(timezone.utc) - timedelta(hours=24))
        end = datetime.now(timezone.utc)

        def _get_metrics() -> list[dict[str, Any]]:
            paginator = cw.get_paginator("list_metrics")
            metrics = []
            for page in paginator.paginate(Namespace="AWS/Bedrock"):
                metrics.extend(page.get("Metrics", []))
            return metrics

        loop = asyncio.get_event_loop()
        bedrock_metrics = await loop.run_in_executor(None, _get_metrics)

        events: list[AgentEvent] = []
        seen_models: set[str] = set()

        for metric in bedrock_metrics[:limit]:
            dims = {d["Name"]: d["Value"] for d in metric.get("Dimensions", [])}
            model_id = dims.get("ModelId", "unknown")
            if model_id in seen_models:
                continue
            seen_models.add(model_id)

            def _get_stat(model: str, metric_name: str) -> float:
                resp = cw.get_metric_statistics(
                    Namespace="AWS/Bedrock",
                    MetricName=metric_name,
                    Dimensions=[{"Name": "ModelId", "Value": model}],
                    StartTime=start,
                    EndTime=end,
                    Period=86400,
                    Statistics=["Sum"],
                )
                pts = resp.get("Datapoints", [])
                return pts[0]["Sum"] if pts else 0.0

            tokens_in = await loop.run_in_executor(
                None, partial(_get_stat, model_id, "InputTokenCount")
            )
            tokens_out = await loop.run_in_executor(
                None, partial(_get_stat, model_id, "OutputTokenCount")
            )
            invocations = await loop.run_in_executor(
                None, partial(_get_stat, model_id, "Invocations")
            )

            events.append(
                AgentEvent(
                    platform=AgentPlatform.BEDROCK,
                    event_type=AgentEventType.COMPLETION,
                    agent_id=f"bedrock-{self._account_id}",
                    agent_name=f"AWS Bedrock ({model_id})",
                    model=model_id,
                    tokens_input=int(tokens_in),
                    tokens_output=int(tokens_out),
                    success=True,
                    raw={
                        "model_id": model_id,
                        "invocations": invocations,
                        "region": self._region,
                    },
                )
            )

        return events
