# SPEC — Multi-Agent Task Orchestrator

**Status:** implemented (all phases 0–6 merged) · **Owner:** natan497 · **Runtime cost:** $0 (Groq free tier)

## 1. Summary

A multi-agent orchestrator that turns a natural-language goal into actions via a
**plan → execute → observe → re-plan** loop:

- **Planner agent** (large reasoning model) decomposes the goal into an ordered plan and,
  after each step, decides: continue / retry / re-plan / done.
- **Executor agent** (small fast model) takes the current step + available tools and emits a
  tool call; the tool runs; the result is observed.

Both agents run on **Groq's free tier** through a **provider-agnostic `LLMProvider`** interface.
A second provider (`AnthropicProvider`) is interface-complete and activates only if a key is
present — so the project is "Anthropic-ready" with zero orchestrator changes.

## 2. Goals / Non-goals

**Goals**
- Provider-agnostic LLM layer; Groq wired by default, Anthropic-ready.
- Schema-driven tool calling with validation and a tool registry.
- Plan-execute-observe loop with re-planning and hard guardrails.
- Rate-limit-aware retry/backoff and **TPM-aware** pacing.
- Full, inspectable run trace + token accounting; small eval harness.
- 3–4 real, mostly keyless demo tools and runnable example tasks.

**Non-goals (v1)**
- No web UI, no persistence/DB, no multi-user, no parallel agents/concurrency.
- No fine-tuning. No streaming UI (a `rich` trace is enough).
- No vector store / RAG.

## 3. Architecture

```
Goal
 └─> Planner (gpt-oss-120b)  ──> ordered Plan (steps)
        │
        ▼  for each step (loop):
     Executor (llama-3.1-8b-instant) ─> tool_call ─> Tool.run() ─> Observation
        │
        ▼
     Planner evaluates Observation ─> {continue | retry | re-plan | done}
        │
        ▼  (until done | max_iterations | max_tool_calls)
 Final answer  +  RunTrace (JSON) + token usage
```

### Components & responsibilities
- `providers/base.py` — `LLMProvider` ABC. Single normalized method:
  `complete(messages, tools=None, **opts) -> Completion` where
  `Completion = {text: str|None, tool_calls: list[ToolCall], usage: Usage, raw: Any}`.
  Providers translate to/from their own wire format internally.
- `providers/groq_provider.py` — implements `LLMProvider` over the Groq (OpenAI-compatible)
  API. Takes a `model` ID at construction. Owns retry/backoff + rate-limit header handling.
- `providers/anthropic_provider.py` — implements the same interface; only constructed if
  `ANTHROPIC_API_KEY` is set. Translates tools to/from Anthropic's format.
- `orchestrator/models.py` — Pydantic models: `Tool`, `Step`, `Plan`, `ToolCall`,
  `Observation`, `Usage`, `RunTrace`, `OrchestratorConfig`.
- `orchestrator/config.py` — loads env → roles→model map, limits, guardrails, keys.
- `orchestrator/planner.py` — builds the planner prompt; produces a structured `Plan`;
  evaluates observations into a control decision.
- `orchestrator/executor.py` — builds the executor prompt; produces a single validated
  `ToolCall` for the current step.
- `orchestrator/state.py` — run state + `RunTrace` accumulation, token tally, dump-to-JSON.
- `orchestrator/orchestrator.py` — the loop, guardrails, pacing, error handling.
- `tools/base.py` — `Tool` ABC: `name`, `description`, `params_schema` (JSON schema), `run(args)`.
- `tools/registry.py` — register/lookup; serialize tool set to provider tool format;
  validate args against schema before dispatch.
- `tools/builtins/` — concrete tools (see §6).

## 4. LLM provider interface (contract)

```python
class LLMProvider(ABC):
    @abstractmethod
    def complete(self, messages: list[Message],
                 tools: list[ToolSpec] | None = None,
                 **opts) -> Completion: ...
```
- `messages`: normalized role/content dicts. `tools`: provider-neutral specs from the registry.
- Returns normalized `Completion`. Errors are normalized too: `RateLimitError`,
  `ProviderError`, `ToolFormatError`. The orchestrator never sees raw SDK exceptions.

## 5. Rate-limit & cost strategy (load-bearing)

- **TPM is the binding constraint**, not RPM. Free-tier TPM is 6–12K depending on model;
  one fat planner prompt can be several K tokens.
- Every `complete()` call wrapped with **tenacity** exponential backoff that honors HTTP `429`,
  `retry-after`, and `x-ratelimit-remaining-*` / `x-ratelimit-reset-*` headers.
- Orchestrator **paces against the token budget** between steps (token-aware delay, not a
  fixed sleep) so a multi-step run stays under per-minute TPM.
- **Stable system prompts** ⇒ Groq automatic prefix caching applies; cached input tokens do
  **not** count toward TPM and cost 50% less. Planner/executor system prompts are constants.
- Keep prompts lean: only the tools relevant to the goal are sent; observations summarized.
- Token accounting per run surfaced in the trace; eval harness budgets against RPD too
  (planner models cap at 1,000 requests/day).

## 6. Built-in tools (v1)

| Tool | Key needed | Purpose |
|------|-----------|---------|
| `calculator` | none | safe arithmetic eval (no `eval()`); demonstrates basic round-trip |
| `http_request` | none | generic REST GET/POST with allowlist + timeout |
| `weather` (Open-Meteo) | none | real keyless API: geocode + current/forecast |
| `wikipedia_search` (optional) | none | real keyless API: search + summary |

Each tool: JSON-schema params, input validation, typed result, and its own tests
(network tools use mocked responses in unit tests; one opt-in live integration test).

## 7. Guardrails & failure handling

- `max_iterations` (default 10), `max_tool_calls` (default 20), per-tool timeout.
- Unknown/hallucinated tool name → structured error fed back to planner (no crash).
- Invalid tool args → validation error fed back; bounded retries before re-plan.
- Provider exhausted after backoff → run ends gracefully with partial trace + reason.
- Loop detection: same failing step N times → force re-plan or abort.

## 8. Observability

- `RunTrace`: ordered records of every planner decision, executor tool call, tool result,
  retries, and token usage; `to_json()` dumps the full run.
- `rich` console renderer for live runs and the demo.
- Per-run token + request counters (input/cached/output) in the summary.

## 9. Eval harness (cheap, high-value)

- 5–10 fixed tasks in `examples/` with expected outcomes / assertions.
- Runner executes all, reports pass/fail + tokens + requests, respects RPD/TPM budgets.

## 10. Configuration / env

`.env.example`:
```
GROQ_API_KEY=            # required
ANTHROPIC_API_KEY=       # optional; enables AnthropicProvider when set
PLANNER_MODEL=openai/gpt-oss-120b
EXECUTOR_MODEL=llama-3.1-8b-instant
MAX_ITERATIONS=10
MAX_TOOL_CALLS=20
```

## 11. Build phases (each phase = one branch = one PR)

| Phase | Branch | Deliverable | Done when |
|-------|--------|-------------|-----------|
| **0** Scaffold | `feat/phase-0-scaffold` | `pyproject.toml`, layout, ruff/pytest config, `.env.example`, CI stub, gitignore | `pip install -e ".[dev]"` works; empty `pytest` green; ruff clean |
| **1** Models + interface | `feat/phase-1-models-interface` | `models.py`, `providers/base.py`, design docstrings | interfaces import & type-check; model unit tests pass |
| **2** Groq provider + rate limits | `feat/phase-2-groq-provider` | `groq_provider.py`, `config.py`, backoff, `anthropic_provider.py` stub | smoke test (mocked) passes; forced 429 retries cleanly; opt-in live smoke documented |
| **3** Tool layer | `feat/phase-3-tools` | `tools/base.py`, `registry.py`, `calculator` | calculator round-trips through provider (mocked); registry serialization tested |
| **4** Orchestration loop | `feat/phase-4-orchestrator` | `planner.py`, `executor.py`, `state.py`, `orchestrator.py`, guardrails, pacing | a 2-tool task runs end-to-end (mocked planner/executor) with full trace; guardrail tests pass |
| **5** Real tools + demos | `feat/phase-5-tools-demos` | `http_request`, `weather`, optional `wikipedia`, `examples/demo_tasks.py` | demo chains ≥2 tools; tools unit-tested (mocked) + opt-in live tests |
| **6** Eval + README + demo | `feat/phase-6-eval-readme` | eval harness, README (diagram, quickstart, sample trace), demo GIF | stranger clones, adds one key, runs a demo in <5 min; evals run within budget |

PRs are opened only when "Done when" is met and `pytest` + `ruff` pass.

## 12. Open prerequisites (resolve before PR-per-feature flow)

1. **`gh` CLI not installed** (neither WSL nor Windows). Needed to auto-open PRs. Options:
   (a) install + auth `gh`; (b) provide a token so PRs are created via GitHub API. Branch
   *pushes* already work over the existing SSH remote alias.
2. Confirm a `GROQ_API_KEY` is available for the opt-in live smoke tests (unit tests don't need it).
