# AgentRadar

Open-source, vendor-neutral observability for enterprise AI agent fleets.

Monitor Copilot, Glean, Windsurf, OpenAI, and AWS Bedrock from a single platform — no vendor lock-in, OpenTelemetry-native.

## Features

- **Multi-platform scrapers** — GitHub Copilot, Microsoft 365 Copilot, OpenAI, Glean, AWS Bedrock
- **Tensor decomposition anomaly detection** — CP/PARAFAC across the entire fleet simultaneously
- **Behavioral drift detection** — Cohen's d effect size on latency, error rate, token usage, cost
- **Query failure clustering** — TF-IDF + KMeans with sensitive-keyword flagging
- **Policy rules engine** — YAML-driven, pluggable, audit-trail ready
- **FastAPI backend** — JWT+RBAC, OpenAPI docs, async SQLAlchemy
- **OpenTelemetry native** — every event exports `agent.*` semantic attributes

## Quickstart

```bash
pip install agent-radar

# Start the API server (SQLite by default)
agent-radar serve

# Scrape GitHub Copilot usage
agent-radar scrape copilot-github --token ghp_... --org my-org

# Run drift detection against a specific agent
agent-radar detect my-agent-id

# Evaluate policy rules
agent-radar policy rules.yaml
```

## Docker Compose

```bash
docker compose up
```

API available at `http://localhost:8080` · Docs at `http://localhost:8080/docs`

## Policy Rules

```yaml
rules:
  - name: sensitive_data_access
    description: Non-HR user accessing HR data
    severity: high
    condition:
      data_source_contains: "HR"
    action: alert

  - name: excessive_cost
    description: Single query cost > $1
    severity: critical
    condition:
      cost_usd_gt: 1.0
    action: alert

  - name: after_hours_finance
    description: Financial agent used outside business hours
    severity: medium
    condition:
      hour_outside: [8, 18]
      agent_name_contains: "finance"
    action: alert
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      AgentRadar                          │
│                                                          │
│  Scrapers          Detection          Policy             │
│  ─────────         ─────────          ──────             │
│  Copilot  ──┐      Tensor CP  ──┐     YAML rules ──┐    │
│  OpenAI   ──┤ →    Drift KDE  ──┤ →   Evaluators  ──┤   │
│  Glean    ──┤      Clustering ──┘     Violations  ──┘   │
│  Bedrock  ──┘                                            │
│                                                          │
│  FastAPI + SQLAlchemy ← OpenTelemetry export             │
└─────────────────────────────────────────────────────────┘
```

## Development

```bash
git clone https://github.com/stark256-spec/agent-radar
cd agent-radar
pip install -e ".[dev]"
pytest tests/ -v
ruff check . && ruff format --check .
```

## License

Apache 2.0
