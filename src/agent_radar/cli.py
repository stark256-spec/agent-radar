"""AgentRadar CLI — scrape, detect, and policy-check agent fleets."""

from __future__ import annotations

import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="agent-radar",
    help="Open-source observability for enterprise AI agent fleets.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def health():
    """Check API server health."""
    import httpx

    try:
        r = httpx.get("http://localhost:8080/health", timeout=5)
        data = r.json()
        console.print(f"[green]OK[/green]  version={data.get('version', '?')}")
    except Exception as exc:
        console.print(f"[red]UNREACHABLE[/red]  {exc}")
        raise typer.Exit(1)


@app.command()
def scrape(
    platform: str = typer.Argument(..., help="copilot-github | openai | glean"),
    token: str = typer.Option(..., "--token", "-t", envvar="RADAR_TOKEN"),
    org: Optional[str] = typer.Option(None, "--org", help="GitHub org or Glean tenant"),
    org_id: Optional[str] = typer.Option(None, "--org-id", help="OpenAI org-id"),
    limit: int = typer.Option(100, "--limit", "-n"),
    output: str = typer.Option("table", "--output", "-o", help="table | json"),
):
    """Scrape agent events from a platform and print them."""

    async def _run():
        import httpx

        async with httpx.AsyncClient() as client:
            if platform == "copilot-github":
                if not org:
                    console.print("[red]--org required for copilot-github[/red]")
                    raise typer.Exit(1)
                from agent_radar.scrapers.copilot import GithubCopilotScraper

                scraper = GithubCopilotScraper(token, org, client=client)
            elif platform == "openai":
                from agent_radar.scrapers.openai import OpenAIScraper

                scraper = OpenAIScraper(token, org_id=org_id, client=client)
            elif platform == "glean":
                if not org:
                    console.print("[red]--org (tenant) required for glean[/red]")
                    raise typer.Exit(1)
                from agent_radar.scrapers.glean import GleanScraper

                scraper = GleanScraper(org, token, client=client)
            else:
                console.print(f"[red]Unknown platform: {platform}[/red]")
                raise typer.Exit(1)

            events = await scraper.scrape(limit=limit)

        if output == "json":
            import json

            console.print(json.dumps([e.model_dump(mode="json") for e in events], indent=2))
            return

        table = Table(title=f"{platform} events ({len(events)})")
        table.add_column("Time", style="dim")
        table.add_column("Agent")
        table.add_column("User")
        table.add_column("Type")
        table.add_column("Latency ms", justify="right")
        table.add_column("Cost $", justify="right")
        table.add_column("OK")

        for e in events:
            table.add_row(
                e.timestamp.strftime("%Y-%m-%d %H:%M") if e.timestamp else "-",
                e.agent_name,
                e.user_id or e.user_email or "-",
                e.event_type.value,
                f"{e.latency_ms:.0f}" if e.latency_ms else "-",
                f"{e.cost_usd:.4f}" if e.cost_usd else "-",
                "[green]✓[/green]" if e.success else "[red]✗[/red]",
            )
        console.print(table)

    asyncio.run(_run())


@app.command()
def detect(
    agent_id: str = typer.Argument(..., help="Agent ID to analyse"),
    baseline_days: int = typer.Option(7, "--baseline-days"),
    current_hours: int = typer.Option(1, "--current-hours"),
):
    """Run drift detection against the local API server."""

    async def _run():
        import httpx

        async with httpx.AsyncClient() as client:
            r = await client.get(
                "http://localhost:8080/events",
                params={"agent_id": agent_id, "limit": 5000},
            )
            if r.status_code != 200:
                console.print(f"[red]API error {r.status_code}[/red]")
                raise typer.Exit(1)

            from datetime import datetime, timezone

            from agent_radar.detection.drift import DriftDetector
            from agent_radar.schema import AgentEvent

            events = [AgentEvent(**e) for e in r.json()]
            if not events:
                console.print("[yellow]No events found for this agent.[/yellow]")
                return

            detector = DriftDetector(baseline_days=baseline_days, current_hours=current_hours)
            platform = events[0].platform
            agent_name = events[0].agent_name
            anomalies = detector.detect(
                agent_id, agent_name, platform, events, now=datetime.now(timezone.utc)
            )

            if not anomalies:
                console.print("[green]No drift detected.[/green]")
                return

            table = Table(title=f"Drift anomalies for {agent_id}")
            table.add_column("Metric")
            table.add_column("Severity")
            table.add_column("Baseline")
            table.add_column("Current")
            table.add_column("Cohen's d", justify="right")

            for a in anomalies:
                ev = a.evidence
                table.add_row(
                    a.affected_metric,
                    f"[yellow]{a.severity.value}[/yellow]",
                    f"{ev.get('baseline_mean', 0):.2f}",
                    f"{ev.get('current_mean', 0):.2f}",
                    f"{ev.get('cohens_d', 0):.2f}",
                )
            console.print(table)

    asyncio.run(_run())


@app.command()
def policy(
    rules_file: str = typer.Argument(..., help="Path to YAML rules file"),
    events_file: Optional[str] = typer.Option(
        None, "--events", help="JSON file of events (reads from API if omitted)"
    ),
    limit: int = typer.Option(500, "--limit"),
):
    """Evaluate policy rules against recent events."""

    async def _run():
        import json

        from agent_radar.policy.engine import PolicyEngine
        from agent_radar.schema import AgentEvent

        engine = PolicyEngine.from_yaml(rules_file)
        console.print(f"Loaded [bold]{engine.rule_count}[/bold] active rules from {rules_file}")

        if events_file:
            with open(events_file) as f:
                raw = json.load(f)
            events = [AgentEvent(**e) for e in raw]
        else:
            import httpx

            async with httpx.AsyncClient() as client:
                r = await client.get("http://localhost:8080/events", params={"limit": limit})
                events = [AgentEvent(**e) for e in r.json()]

        violations = engine.evaluate_batch(events)

        if not violations:
            console.print("[green]No violations found.[/green]")
            return

        table = Table(title=f"{len(violations)} violation(s)")
        table.add_column("Rule")
        table.add_column("Severity")
        table.add_column("Agent")
        table.add_column("User")
        table.add_column("Event")

        for v in violations:
            table.add_row(
                v.rule_name,
                v.severity.value.upper(),
                v.agent_name,
                v.user_email or v.user_id or "-",
                v.event_id or "-",
            )
        console.print(table)

    asyncio.run(_run())


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8080, "--port"),
    reload: bool = typer.Option(False, "--reload"),
):
    """Start the AgentRadar API server."""
    import uvicorn

    uvicorn.run(
        "agent_radar.api.main:app",
        host=host,
        port=port,
        reload=reload,
    )
