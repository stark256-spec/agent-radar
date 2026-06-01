"""Tests for platform scrapers — mocked HTTP with respx."""

import httpx
import pytest
import respx

from agent_radar.schema import AgentEventType, AgentPlatform
from agent_radar.scrapers.copilot import GithubCopilotScraper
from agent_radar.scrapers.glean import GleanScraper
from agent_radar.scrapers.openai import OpenAIScraper

# ── GitHub Copilot ───────────────────────────────────────────────────────────

_COPILOT_RESPONSE = [
    {
        "date": "2025-06-01",
        "breakdown": [
            {
                "user": {"login": "alice"},
                "language": "python",
                "suggestions_count": 120,
                "acceptances_count": 90,
                "lines_accepted": 200,
            },
            {
                "user": {"login": "bob"},
                "language": "typescript",
                "suggestions_count": 80,
                "acceptances_count": 40,
                "lines_accepted": 75,
            },
        ],
    }
]


@pytest.mark.asyncio
async def test_github_copilot_scraper():
    with respx.mock:
        respx.get("https://api.github.com/orgs/acme/copilot/usage").mock(
            return_value=httpx.Response(200, json=_COPILOT_RESPONSE)
        )
        async with httpx.AsyncClient() as client:
            scraper = GithubCopilotScraper("tok", "acme", client=client)
            events = await scraper.scrape()

    assert len(events) == 2
    assert all(e.platform == AgentPlatform.COPILOT_GITHUB for e in events)
    assert events[0].user_id == "alice"
    assert events[0].event_type == AgentEventType.COMPLETION
    assert events[0].raw["acceptances_count"] == 90


@pytest.mark.asyncio
async def test_github_copilot_scraper_limit():
    with respx.mock:
        respx.get("https://api.github.com/orgs/acme/copilot/usage").mock(
            return_value=httpx.Response(200, json=_COPILOT_RESPONSE)
        )
        async with httpx.AsyncClient() as client:
            scraper = GithubCopilotScraper("tok", "acme", client=client)
            events = await scraper.scrape(limit=1)

    assert len(events) == 1


# ── OpenAI ───────────────────────────────────────────────────────────────────

_OPENAI_USAGE = {
    "data": [
        {
            "snapshot_id": "gpt-4o-2024-05-13",
            "n_context_tokens_total": 10000,
            "n_generated_tokens_total": 3000,
            "n_requests": 5,
            "user_public_id": "user-abc",
        },
        {
            "snapshot_id": "gpt-3.5-turbo",
            "n_context_tokens_total": 5000,
            "n_generated_tokens_total": 2000,
            "n_requests": 10,
            "user_public_id": "user-xyz",
        },
    ]
}


@pytest.mark.asyncio
async def test_openai_scraper():
    with respx.mock:
        respx.get("https://api.openai.com/v1/usage").mock(
            return_value=httpx.Response(200, json=_OPENAI_USAGE)
        )
        async with httpx.AsyncClient() as client:
            scraper = OpenAIScraper("sk-test", org_id="org-1", client=client)
            events = await scraper.scrape()

    assert len(events) == 2
    assert all(e.platform == AgentPlatform.OPENAI for e in events)
    gpt4_event = next(e for e in events if "gpt-4o" in (e.model or ""))
    assert gpt4_event.tokens_input == 10000
    assert gpt4_event.tokens_output == 3000
    assert gpt4_event.cost_usd > 0


@pytest.mark.asyncio
async def test_openai_cost_estimate():
    with respx.mock:
        respx.get("https://api.openai.com/v1/usage").mock(
            return_value=httpx.Response(200, json=_OPENAI_USAGE)
        )
        async with httpx.AsyncClient() as client:
            scraper = OpenAIScraper("sk-test", client=client)
            events = await scraper.scrape()

    # gpt-4o: input=10K tokens @ $0.005/1K = $0.05, output=3K @ $0.015/1K = $0.045 → $0.095
    gpt4_event = next(e for e in events if "gpt-4o" in (e.model or ""))
    assert abs(gpt4_event.cost_usd - 0.095) < 0.001


# ── Glean ────────────────────────────────────────────────────────────────────

_GLEAN_ACTIVITIES = {
    "activities": [
        {
            "activityType": "SEARCH",
            "user": {"email": "charlie@example.com", "department": "Engineering"},
            "query": "kubernetes deployment config",
            "latencyMs": 350.0,
            "success": True,
            "dataSources": ["confluence", "jira"],
            "agentId": "glean-search",
            "agentName": "Glean Search",
        },
        {
            "activityType": "AGENT_QUERY",
            "user": {"email": "diana@example.com"},
            "query": "summarize Q2 financial report",
            "latencyMs": 1200.0,
            "success": False,
            "errorMessage": "Access denied",
            "dataSources": ["finance-reports"],
            "agentId": "glean-ai",
            "agentName": "Glean AI",
        },
    ]
}


@pytest.mark.asyncio
async def test_glean_scraper():
    with respx.mock:
        respx.post("https://acme.glean.com/rest/api/v1/getactivity").mock(
            return_value=httpx.Response(200, json=_GLEAN_ACTIVITIES)
        )
        async with httpx.AsyncClient() as client:
            scraper = GleanScraper("acme", "glean-token", client=client)
            events = await scraper.scrape()

    assert len(events) == 2
    search_event = events[0]
    assert search_event.platform == AgentPlatform.GLEAN
    assert search_event.user_email == "charlie@example.com"
    assert search_event.department == "Engineering"
    assert "confluence" in search_event.data_sources

    failed_event = events[1]
    assert not failed_event.success
    assert failed_event.error_message == "Access denied"
