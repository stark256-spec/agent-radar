# Contributing to AgentRadar

Contributions are welcome. This guide covers the basics.

## Setup

```bash
git clone https://github.com/stark256-spec/agent-radar
cd agent-radar
pip install -e ".[dev]"
```

## Before You Open a PR

```bash
pytest tests/ -v          # all 31 tests must pass
ruff check . && ruff format --check .   # no lint errors
```

## Adding a Scraper

1. Create `src/agent_radar/scrapers/<platform>.py`
2. Subclass `BaseScraper` and implement `scrape()`
3. Map platform events to `AgentEvent` with `AgentPlatform.<PLATFORM>`
4. Add mocked HTTP tests in `tests/test_scrapers.py` using `respx`

## Adding a Policy Condition

Register a new built-in evaluator in `src/agent_radar/policy/engine.py`:

```python
"my_condition": lambda val, ev: <boolean expression>,
```

Or register one at runtime:

```python
engine.register_evaluator("my_condition", lambda val, ev: ...)
```

## License

Apache 2.0. By contributing you agree to license your work under the same terms.
