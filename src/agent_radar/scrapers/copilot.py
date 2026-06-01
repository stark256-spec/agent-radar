"""
Microsoft Copilot scraper — collects GitHub Copilot and M365 Copilot usage
via the Microsoft Graph API and GitHub Copilot Usage API.

Auth: Service principal with Azure AD (client credentials flow).
Required scopes: Reports.Read.All (M365), GitHub org admin (Copilot).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

import httpx

from agent_radar.schema import AgentEvent, AgentEventType, AgentPlatform
from agent_radar.scrapers.base import BaseScraper

_GRAPH_BASE = "https://graph.microsoft.com/v1.0"
_GH_API = "https://api.github.com"


class GithubCopilotScraper(BaseScraper):
    """Collect GitHub Copilot usage metrics via GitHub REST API.

    Requires a GitHub token with org admin:read scope.
    Endpoint: GET /orgs/{org}/copilot/usage
    """

    platform_name = "copilot_github"

    def __init__(
        self,
        github_token: str,
        org: str,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._token = github_token
        self._org = org
        self._client = client

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "X-GitHub-Api-Version": "2022-11-28",
        }

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        url = f"{_GH_API}{path}"
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
        """
        Fetch Copilot usage per seat from /orgs/{org}/copilot/usage.

        The GitHub API returns daily aggregates per user. We emit one AgentEvent
        per user-day record with the suggestion/acceptance counts as metrics.
        """
        data = await self._get(f"/orgs/{self._org}/copilot/usage")
        events: list[AgentEvent] = []

        for day in data:
            date_str = day.get("date", "")
            breakdown = day.get("breakdown", [])

            for entry in breakdown:
                if len(events) >= limit:
                    return events
                user = entry.get("user", {})
                user_login = user.get("login", "unknown") if isinstance(user, dict) else str(user)
                suggestions = entry.get("suggestions_count", 0)
                acceptances = entry.get("acceptances_count", 0)
                lines_accepted = entry.get("lines_accepted", 0)
                language = entry.get("language", "unknown")

                events.append(
                    AgentEvent(
                        platform=AgentPlatform.COPILOT_GITHUB,
                        event_type=AgentEventType.COMPLETION,
                        agent_id=f"github-copilot-{self._org}",
                        agent_name="GitHub Copilot",
                        user_id=user_login,
                        user_email=f"{user_login}@github",
                        success=True,
                        tokens_output=lines_accepted,
                        raw={
                            "date": date_str,
                            "language": language,
                            "suggestions_count": suggestions,
                            "acceptances_count": acceptances,
                            "lines_accepted": lines_accepted,
                            "acceptance_rate": (
                                acceptances / suggestions if suggestions > 0 else 0.0
                            ),
                        },
                    )
                )

        return events


class M365CopilotScraper(BaseScraper):
    """Collect Microsoft 365 Copilot usage via Microsoft Graph Reports API.

    Requires an Entra ID service principal with Reports.Read.All scope.
    Endpoint: GET /reports/getMicrosoft365CopilotUsageUserDetail(period='D7')
    """

    platform_name = "copilot_m365"

    def __init__(
        self,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._tenant_id = tenant_id
        self._client_id = client_id
        self._client_secret = client_secret
        self._http = client
        self._access_token: str | None = None

    async def _get_token(self) -> str:
        token_url = f"https://login.microsoftonline.com/{self._tenant_id}/oauth2/v2.0/token"
        payload = {
            "client_id": self._client_id,
            "client_secret": self._client_secret,
            "scope": "https://graph.microsoft.com/.default",
            "grant_type": "client_credentials",
        }
        if self._http:
            r = await self._http.post(token_url, data=payload, timeout=15)
        else:
            async with httpx.AsyncClient() as c:
                r = await c.post(token_url, data=payload, timeout=15)
        r.raise_for_status()
        return r.json()["access_token"]

    async def _get(self, path: str) -> Any:
        if not self._access_token:
            self._access_token = await self._get_token()
        headers = {"Authorization": f"Bearer {self._access_token}"}
        url = f"{_GRAPH_BASE}{path}"
        if self._http:
            r = await self._http.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            return r.json()
        async with httpx.AsyncClient() as c:
            r = await c.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            return r.json()

    async def scrape(
        self,
        *,
        since: datetime | None = None,
        limit: int = 500,
    ) -> list[AgentEvent]:
        data = await self._get("/reports/getMicrosoft365CopilotUsageUserDetail(period='D7')")
        users = data.get("value", [])
        events: list[AgentEvent] = []

        for user in users[:limit]:
            upn = user.get("userPrincipalName", "unknown")
            word_count = user.get("microsoftWordCopilotActivityUserCount", 0)
            teams_count = user.get("microsoftTeamsCopilotActivityUserCount", 0)

            events.append(
                AgentEvent(
                    platform=AgentPlatform.COPILOT_M365,
                    event_type=AgentEventType.QUERY,
                    agent_id=f"m365-copilot-{self._tenant_id}",
                    agent_name="Microsoft 365 Copilot",
                    user_id=upn,
                    user_email=upn,
                    tenant_id=self._tenant_id,
                    success=True,
                    raw={
                        "word_activity": word_count,
                        "teams_activity": teams_count,
                        **user,
                    },
                )
            )

        return events
