# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this repo is

Agent skills for neural network interpretability with [nnsight](https://nnsight.net/)
and NDIF, packaged for Claude Code (`.claude-plugin/`) and OpenAI Codex
(`.codex/skills/` symlinks).

**These skills target nnsight 0.8.** Older idioms (`.value`, `nnsight.list()`,
`tracer.next()`, `with tracer.all():`, `LanguageModel`, proxies) are wrong here and
the test suite rejects them in runnable examples. When in doubt about current
behavior, check the nnsight source and docs rather than memory, then verify by
running the code.

## Layout

```
plugins/nnsight/skills/<skill-name>/
├── SKILL.md          # what the agent reads every time the skill fires
├── references/*.md   # loaded on demand — depth, tables, worked examples
└── scripts/*.py      # runnable tools, not reading material
```

`.codex/skills/<skill-name>` is a symlink to each skill directory.

## The rule that matters: everything is executed

Every fenced ```python block in every `SKILL.md` and `references/*.md` is run by
`tests/test_skills.py` against real models. Blocks in one file share a namespace
and run in document order, so later blocks can build on earlier ones.

Directives go in an HTML comment immediately above the fence:

```markdown
<!-- test: skip -->                    don't run (still syntax-checked)
<!-- test: skip nocompile -->          don't run, don't compile
<!-- test: setup -->                   a block later ones build on
<!-- test: remote -->                  only with NDIF_HOST set
<!-- test: gpu -->                     only with CUDA
<!-- test: slow -->                    only with --run-slow
<!-- test: expect-error OutOfOrderError --> must raise this
```

Use ```python-legacy for old-API code shown deliberately (porting guides) — it is
never executed and is exempt from the pre-0.8 API check.

Two traps this harness has already caught, worth remembering:

- **nnsight needs the block's source on disk**, so each block is written to a
  temp file and run with `runpy` — not `exec`ed from a string.
- **The shared namespace can mask a false claim.** A block demonstrating "this
  variable never gets assigned" will silently pass if an earlier block bound that
  name. Assert with `"name" in globals()` instead of relying on a `NameError`.

## Commands

```bash
make test              # everything, NDIF_HOST=http://localhost:8001 by default
make test-local        # skip blocks needing an NDIF deployment
make test-structure    # packaging only — fast, no model loading
make test-skill SKILL=nnsight
make report            # per-file table of blocks run / skipped
```

Use the `ndif2` conda env. A local NDIF for remote tests runs at
`http://localhost:8001` (no API key needed).

Models used in examples: `openai-community/gpt2` and
`HuggingFaceTB/SmolLM2-135M-Instruct` for anything executed;
`EleutherAI/pythia-70m-deduped` for a second architecture. Illustrative
(non-executed) snippets may name Llama-3.1-8B/70B. Never reference private or
org-internal checkpoints.

## Adding a skill

1. `plugins/nnsight/skills/<skill-name>/SKILL.md` with frontmatter `name` (must
   match the directory) and `description` (what an agent selects on — say when to
   load it, not just what it is).
2. Depth goes in `references/`, tools in `scripts/`. Keep `SKILL.md` to what is
   worth reading on every activation.
3. `cd .codex/skills && ln -s ../../plugins/nnsight/skills/<skill-name> .`
4. Add a row to the README table.
5. `make test`.

`tests/test_structure.py` enforces all of the above plus link resolution and the
pre-0.8 API ban.

## Writing style for skills

- Lead with what breaks, not with what exists — agents need the failure modes.
- Every example runs; no pseudo-code presented as code.
- Prefer verified claims ("a GPT-2 block returns a plain tensor") over hedges.
- Cross-link between skills by name (`the nnsight-debugging skill`) so an agent
  knows where to go next.

## Source material

- nnsight source and docs: `/home/localjadenfk/wd/nnsight` (branch `0.8`,
  `CLAUDE.md` routes to `docs/`)
- nnterp (0.8 branch): `/home/localjadenfk/wd/nnterp`
- Tutorials and paper implementations: `/home/localjadenfk/wd/nnsight-website/docs`
