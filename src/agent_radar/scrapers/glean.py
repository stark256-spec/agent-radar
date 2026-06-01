"""
Glean Activity scraper — collects search and agent invocation events
from the Glean Activity API.

Endpoint: POST https://{instance}.glean.com/rest/api/v1/getactivity
Auth: Bearer token (Glean API key with activity:read scope).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from agent_radar.schema import AgentEvent, AgentEventType, AgentPlatform
from agent_radar.scrapers.base import BaseScraper


class GleanScraper(BaseScraper):
    """Collect Glean search and AI agent invocations.

    Ships one AgentEvent per Glean activity record.
    """

    platform_name = "glean"

    def __init__(
        self,
        instance: str,
        api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._base = f"https://{instance}.glean.com/rest/api/v1"
        self._api_key = api_key
        self._client = client

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    async def _post(self, path: str, payload: dict[str, Any]) -> Any:
        url = f"{self._base}{path}"
        if self._client:
            r = await self._client.post(url, headers=self._headers(), json=payload, timeout=30)
            r.raise_for_status()
            return r.json()
        async with httpx.AsyncClient() as c:
            r = await c.post(url, headers=self._headers(), json=payload, timeout=30)
            r.raise_for_status()
            return r.json()

    async def scrape(
        self,
        *,
        since: datetime | None = None,
        limit: int = 500,
    ) -> list[AgentEvent]:
        start = since or (datetime.now(timezone.utc) - timedelta(hours=24))
        payload = {
            "startTime": int(start.timestamp()),
            "endTime": int(datetime.now(timezone.utc).timestamp()),
            "pageSize": min(limit, 100),
        }
        data = await self._post("/getactivity", payload)
        activities = data.get("activities", [])
        events: list[AgentEvent] = []

        for activity in activities[:limit]:
            activity_type = activity.get("activityType", "SEARCH")
            user = activity.get("user", {})
            email = user.get("email", "unknown")

            # Map Glean activity types to AgentEventType
            event_type = (
                AgentEventType.QUERY
                if activity_type in ("SEARCH", "AGENT_QUERY")
                else AgentEventType.TOOL_CALL
            )

            events.append(
                AgentEvent(
                    platform=AgentPlatform.GLEAN,
                    event_type=event_type,
                    agent_id=f"glean-{activity.get('agentId', 'search')}",
                    agent_name=activity.get("agentName", "Glean Search"),
                    user_id=email,
                    user_email=email,
                    department=user.get("department"),
                    query_text=activity.get("query"),
                    latency_ms=activity.get("latencyMs"),
                    success=activity.get("success", True),
                    error_message=activity.get("errorMessage"),
                    data_sources=activity.get("dataSources", []),
                    raw=activity,
                )
            )

        return events
