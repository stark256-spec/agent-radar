"""
OpenAI Usage scraper — collects token usage, cost, and model distribution
from the OpenAI Usage API (requires org admin API key).

Endpoint: GET https://api.openai.com/v1/usage?date=YYYY-MM-DD
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from agent_radar.schema import AgentEvent, AgentEventType, AgentPlatform
from agent_radar.scrapers.base import BaseScraper

_OPENAI_BASE = "https://api.openai.com/v1"

# Approximate cost per 1K tokens (input/output) by model family as of mid-2025.
# These are used for cost attribution; update as pricing changes.
_MODEL_COST_PER_1K = {
    "gpt-4o": (0.005, 0.015),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-4": (0.03, 0.06),
    "gpt-3.5-turbo": (0.0005, 0.0015),
    "o1": (0.015, 0.06),
    "o3": (0.01, 0.04),
}


def _estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    for prefix, (in_rate, out_rate) in _MODEL_COST_PER_1K.items():
        if model.startswith(prefix):
            return (tokens_in / 1000) * in_rate + (tokens_out / 1000) * out_rate
    return 0.0


class OpenAIScraper(BaseScraper):
    """Collect OpenAI usage from the Usage API.

    Ships one AgentEvent per user-model-day aggregate.
    Requires an org-level API key with usage:read permission.
    """

    platform_name = "openai"

    def __init__(
        self,
        api_key: str,
        *,
        org_id: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._org_id = org_id
        self._client = client

    def _headers(self) -> dict[str, str]:
        h = {"Authorization": f"Bearer {self._api_key}"}
        if self._org_id:
            h["OpenAI-Organization"] = self._org_id
        return h

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{_OPENAI_BASE}{path}"
        if self._client:
            r = await self._client.get(url, headers=self._headers(), params=params, timeout=30)
            r.raise_for_status()
            return r.json()
        async with httpx.AsyncClient() as c:
            r = await c.get(url, headers=self._headers(), params=params, timeout=30)
            r.raise_for_status()
            return r.json()

    async def scrape(
        self,
        *,
        since: datetime | None = None,
        limit: int = 500,
    ) -> list[AgentEvent]:
        target = since or (datetime.now(timezone.utc) - timedelta(days=1))
        date_str = target.strftime("%Y-%m-%d")
        data = await self._get("/usage", params={"date": date_str})

        events: list[AgentEvent] = []
        for entry in data.get("data", [])[:limit]:
            model = entry.get("snapshot_id", "unknown")
            tokens_in = entry.get("n_context_tokens_total", 0)
            tokens_out = entry.get("n_generated_tokens_total", 0)
            n_requests = entry.get("n_requests", 1)
            user_id = entry.get("user_public_id", "org-level")
            cost = _estimate_cost(model, tokens_in, tokens_out)

            events.append(
                AgentEvent(
                    platform=AgentPlatform.OPENAI,
                    event_type=AgentEventType.COMPLETION,
                    agent_id=f"openai-{self._org_id or 'default'}",
                    agent_name="OpenAI API",
                    user_id=user_id,
                    model=model,
                    tokens_input=tokens_in,
                    tokens_output=tokens_out,
                    cost_usd=cost,
                    success=True,
                    raw={
                        "date": date_str,
                        "n_requests": n_requests,
                        **entry,
                    },
                )
            )

        return events
