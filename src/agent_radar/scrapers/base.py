"""Base class for all platform scrapers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from agent_radar.schema import AgentEvent


class BaseScraper(ABC):
    """Poll a vendor API and return AgentEvent objects."""

    platform_name: str = "unknown"

    @abstractmethod
    async def scrape(
        self,
        *,
        since: datetime | None = None,
        limit: int = 500,
    ) -> list[AgentEvent]:
        """Fetch events since `since`, up to `limit`."""
        ...

    async def health_check(self) -> bool:
        try:
            await self.scrape(limit=1)
            return True
        except Exception:
            return False
