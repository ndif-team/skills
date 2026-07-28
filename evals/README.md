# nnsight resource testbed

Measures **how much a body of reference material helps an agent use nnsight**.

The independent variable is what the agent is given — nothing, the skills in this
repo, the nnsight `docs/` tree, the nnsight.net tutorials, the nnsight source, or
combinations. The dependent variables are whether the agent solves the task and
what it cost to get there: tokens, dollars, wall-clock, turns, and which files it
had to open.

Descended from `tests/agent-evals` in the nnsight repo (dev branch), rebuilt for
0.8 with debugging tasks and per-run cost accounting.

## Quick start

```bash
conda activate ndif2

# Validate the suite itself — no LLM calls, no cost
python -m evalkit.audit -j 4

# See the grid and what it should cost
python run.py --dry-run --models sonnet opus --repeats 3

# A cheap real sweep
python run.py --conditions none skills docs --repeats 1 \
    --tasks basic_01_trace_and_save debug_04_unbounded_iter --output results/smoke.jsonl

# The numbers
python report.py results/smoke.jsonl --order none docs tutorials source skills everything
```

## Resource conditions

| condition | mode | what the agent gets |
|---|---|---|
| `none` | static | nothing — parametric knowledge only (the floor) |
| `skills` | agentic | this repo's plugin, loaded natively via `--plugin-dir` |
| `docs` | agentic | `nnsight/docs/` + its `CLAUDE.md` router, via Read/Grep/Glob |
| `tutorials` | agentic | nnsight.net feature walkthroughs and paper notebooks |
| `source` | agentic | the `src/nnsight` tree |
| `docs+tutorials` | agentic | everything published |
| `skills+docs` | agentic | both routers |
| `everything` | agentic | all four |
| `docs-static` | static | the docs concatenated into the system prompt, no tools |
| `skills-static` | static | every SKILL.md concatenated, no references, no tools |

**Agentic is the honest test.** Skills and the docs router are both designed for
progressive disclosure — a thin entry point plus material fetched on demand.
Pasting everything into a system prompt measures a different thing (and does not
fit: `docs-static` truncates at 600 KB). The static conditions exist so you can
separate "the content is good" from "the routing works".

Tools are read-only on purpose (`Read`, `Grep`, `Glob`, `Skill`; `Bash`/`Write`
denied). The question is whether a resource lets an agent write correct code
*from documentation*. Allowing execution would measure iterate-until-green
instead, and make conditions incomparable.

## Task kinds

**code** (33) — write nnsight code; the result is executed against a real gpt2 and
checked by a verifier.

**debug** (15) — given broken code and the symptom, return a fixed version. Every
bug is one reproduced against 0.8: missing `.save()`, out-of-order access,
`.output[0]` on a tensor, unbounded `tracer.all()`, tuple item assignment,
in-place writes breaking autograd, a barrier count that never releases, a partial
`.skip()`, a legacy-API port. Several are **silent** — the buggy code raises
nothing and simply produces the wrong answer, which is the class documentation
most needs to prevent. One is not a bug at all but an inefficiency (12 forward
passes where 1 would do); it is verified by counting real forward calls.

**mcq** (32) — one of N choices. Measures whether the agent *knows* the rule, as
opposed to being able to *operationalize* it. A large mcq-over-code gap means the
material explains without giving usable templates.

## Metrics

Per run: pass/fail, agent wall-clock, execution wall-clock, input / output /
cache-creation / cache-read tokens, dollars, turns, tool calls, the resource files
opened, the skills invoked, and a failure class for anything that did not pass.

Aggregated by condition, kind, difficulty, model, and tag:

- **pass rate with a Wilson 95% interval** — agents are stochastic; a bare
  percentage over a handful of runs is noise
- **tokens per solve** and **dollars per solve** — the efficiency numbers that
  decide whether a resource earns its context
- median latency, mean turns, mean file reads
- a failure taxonomy (which exception classes the generated code hit)
- what the agents actually opened, per condition — the direct read on whether a
  documentation set's routing works

## Adding to the suite

A code task:

```python
register_task(Task(
    id="advanced_15_my_task",
    name="Human readable",
    difficulty=Difficulty.ADVANCED,
    prompt="Write nnsight code that ...",
    setup_code=GPT2_SETUP,
    verify=lambda result: has_shape(result.get("expected"), last_dim=768),
    tags=["cache"],
    reference_solution="""
with model.trace("Hello"):
    expected = model.transformer.h[0].output.save()
""",
))
```

A debug task takes `symptom` + `buggy_code` instead of a prompt. An MCQ takes
`question` / `choices` / `correct_index`.

**Always write the `reference_solution`.** `python -m evalkit.audit` runs every
one of them through the real runner and asserts it passes, which catches a
verifier no correct answer can satisfy, and catches dependency drift before you
spend money on a grid. It is how the barrier tasks in this suite were found to be
wrong (an 8-token receiver prompt where the transfer needs 9), and how the
transformers 4→5 tuple change was caught in the original suite.

## Design notes

**Subprocess execution.** Agent code runs in its own process because nnsight
compiles a trace block by reading its source off disk (an `exec` of a string
raises `WithBlockNotFoundError`), because a bad intervention can segfault inside
an interleaving greenlet, and because a long grid must not lose everything to one
crash.

**Resumable.** Results are appended as JSONL, one record per cell. `--resume`
skips cells already present, so a sweep can be stopped and continued. `--max-cost`
sets a spend ceiling.

**Cost.** The full grid — 80 tasks × 7 conditions × 2 models × 3 repeats = 3,360
runs — estimates around $320 at Sonnet rates and more with Opus in the mix. Check
`--dry-run` before committing to it. MCQs are ~20x cheaper than code tasks, so
`--kinds mcq` is a cheap way to smoke-test changes.

**Auth.** `claude-code` uses your existing Claude Code login. If calls fail with
"Not logged in", run `claude /login`. Do not pass `--bare` — it skips keychain
reads and breaks auth.

## Layout

```
evals/
├── run.py             # grid runner (resumable, cost-capped)
├── report.py          # aggregation and markdown report
└── evalkit/
    ├── conditions.py  # resource bundles and delivery modes
    ├── providers.py   # claude-code (stream-json) and anthropic backends
    ├── registry.py    # Task / TaskKind / selection
    ├── runner.py      # subprocess execution, verification, MCQ parsing
    ├── audit.py       # reference-solution validation, no LLM calls
    └── tasks/         # code_basic, code_intermediate, code_advanced, debug, mcqs
```
