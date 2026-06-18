# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this project is

A **multi-agent task orchestrator**. A *planner* agent decomposes a natural-language
goal into an ordered plan; an *executor* agent runs each step by calling tools; the
planner observes results and decides to continue, retry, re-plan, or finish. It runs
entirely on **Groq's free tier** today, behind a **provider-agnostic LLM interface** so
an `ANTHROPIC_API_KEY` (or any provider) can be dropped in later with zero orchestrator
changes ("Anthropic-ready").

See [SPEC.md](SPEC.md) for the full specification. SPEC.md is the source of truth — if
this file and SPEC.md disagree, SPEC.md wins.

## Non-negotiable design rules

1. **All provider differences live behind the `LLMProvider` interface** (`src/providers/base.py`).
   The orchestrator, planner, and executor must NEVER import a concrete provider or any
   provider SDK directly. They depend only on `LLMProvider` and the normalized return types.
2. **Groq is the only provider wired by default.** `AnthropicProvider` is interface-complete
   but only instantiated when `ANTHROPIC_API_KEY` is set. Do not require an Anthropic key.
3. **One `GroqProvider` class powers both agents** — it takes a model ID, so the same class
   is the planner (large model) and the executor (small model) with different config.
4. **Roles → model IDs come from config/env, never hardcoded** in agent logic.
5. **The binding rate-limit constraint is TPM (tokens/minute), not RPM.** Every provider call
   goes through retry/backoff that respects `429` + `retry-after` + `x-ratelimit-*` headers,
   and the orchestrator paces multi-step runs against the token budget. Keep system prompts
   **stable** so Groq's automatic prefix caching applies (cached tokens don't count toward TPM).
6. **Write tests alongside each module.** No module is "done" until it has tests and they pass.

## Model IDs (default, overridable via env)

| Role | Default model ID | Free-tier limits to respect |
|------|------------------|------------------------------|
| Planner | `openai/gpt-oss-120b` | 30 RPM · 8K TPM · 1K RPD · 200K TPD |
| Executor | `llama-3.1-8b-instant` | 30 RPM · 6K TPM · 14.4K RPD · 500K TPD |

Alternative planner: `llama-3.3-70b-versatile` (30 RPM · 12K TPM · 1K RPD · 100K TPD).
Don't hardcode — read from env; `GET /openai/v1/models` can validate IDs at startup but is
not a tier/health check.

## Tech stack

- **Python 3.12+** (3.14 available locally). Package layout under `src/`.
- **`groq` SDK** (or OpenAI SDK pointed at `https://api.groq.com/openai/v1`).
- `pydantic` (models/validation), `tenacity` (retry/backoff), `rich` (trace/demo output),
  `python-dotenv` (config), `httpx` (tool HTTP calls).
- **`pytest`** for tests; provider calls mocked in unit tests (no live API needed to run them).
- Lint/format: `ruff`. Type-check: optional `mypy` on `src/`.

## Repository layout

```
src/orchestrator/   orchestrator.py, planner.py, executor.py, state.py, models.py, config.py
src/providers/      base.py, groq_provider.py, anthropic_provider.py
src/tools/          base.py, registry.py, builtins/ (calculator, http_request, weather, ...)
examples/           demo_tasks.py
tests/              one test module per source module
```

## Conventions

- Public functions/classes get type hints and concise docstrings; match surrounding style.
- Tools declare a JSON-schema for params; the registry serializes to the provider tool format.
- Validate tool args before calling `run()`; handle hallucinated/unknown tool names gracefully.
- Guardrails are mandatory in the loop: `max_iterations`, `max_tool_calls`, per-step timeouts.
- Every run produces a `RunTrace` (all agent decisions + tool I/O) dumpable to JSON.
- Secrets only via env / `.env` (gitignored). Never commit keys. `.env.example` lists vars.

## Commands

- Install: `pip install -e ".[dev]"`
- Test: `pytest -q`
- Lint: `ruff check src tests` · Format: `ruff format`
- Demo: `python examples/demo_tasks.py`

## Workflow for building this

- Build **phase by phase** per SPEC.md §"Build phases". Each phase = one branch = one PR.
- Branch naming: `feat/phase-N-short-slug`. Keep PRs small and reviewable.
- Do the **Phase 2 Groq smoke test before** the planning loop, so API wiring and
  orchestration are never debugged at the same time.
- A phase's PR is only opened when its "Done when" criteria in SPEC.md are met and tests pass.
