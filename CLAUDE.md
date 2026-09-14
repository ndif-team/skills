# CLAUDE.md

Guidance for Claude Code when working in this repository.

## What this repo is

Agent skills for the NDIF ecosystem, packaged for Claude Code
(`.claude-plugin/`) and OpenAI Codex (`.codex/skills/` and `.agents/skills/`
symlinks). **Two plugins**, with different audiences and different rules:

| Plugin | For | Skills | Code blocks |
|---|---|---|---|
| `nnsight` | writing interpretability code against model internals | 18 | **executed** against real models |
| `ndif` | running your *own* NDIF server (the backend behind `remote=True`) | 4 | docs only — every block is `<!-- test: skip -->` |

The split matters when you edit: a claim in an nnsight skill is proved by the
test suite running it, a claim in an ndif skill is only as good as the source it
was checked against. Both are registered in `.claude-plugin/marketplace.json`,
and `tests/test_structure.py` validates every plugin listed there.

**The nnsight skills target nnsight 0.8.** Older idioms (`.value`, `nnsight.list()`,
`tracer.next()`, `with tracer.all():`, `LanguageModel`, proxies) are wrong here and
the test suite rejects them in runnable examples. When in doubt about current
behavior, check the nnsight source and docs rather than memory, then verify by
running the code.

**They also assume transformers >= 5.** Model internals moved between 4.x and 5:
a GPT-2 block returns `(hidden_states,)` in 4.x and a plain tensor in 5, and its
attention dropout went from `module.attn_dropout(...)` (source op
`module_attn_dropout_0`) to `nn.functional.dropout(...)`
(`nn_functional_dropout_0`). Anything that indexes `.output` or names a `.source`
operation is version-bound — say which version a claim was verified on.

## Layout

```
plugins/<plugin>/
├── .claude-plugin/plugin.json
└── skills/<skill-name>/
    ├── SKILL.md          # what the agent reads every time the skill fires
    ├── references/*.md   # loaded on demand — depth, tables, worked examples
    └── scripts/*.py      # runnable tools, not reading material
```

`.codex/skills/<skill-name>` and `.agents/skills/<skill-name>` are symlinks to
each skill directory — **both** trees, for every skill in every plugin.

## The rule that matters: everything in `plugins/nnsight` is executed

Every fenced ```python block in every `SKILL.md` and `references/*.md` **under
`plugins/nnsight`** is run by `tests/test_skills.py` against real models. Blocks
in one file share a namespace and run in document order, so later blocks can
build on earlier ones. `tests/docblocks.py` scopes this with `SKILLS_ROOT`; keep
it that way.

`plugins/ndif` is outside that scope. Its python blocks carry
`<!-- test: skip -->` anyway, so nothing would run if the harness were ever
widened, and its shell blocks (```bash) are never executed in either plugin.

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

1. `plugins/<plugin>/skills/<skill-name>/SKILL.md` with frontmatter `name` (must
   match the directory) and `description` (what an agent selects on — say when to
   load it, not just what it is).
2. Depth goes in `references/`, tools in `scripts/`. Keep `SKILL.md` to what is
   worth reading on every activation, roughly 150–300 lines.
3. Symlink it into **both** Codex trees:
   `for d in .codex/skills .agents/skills; do ln -s ../../plugins/<plugin>/skills/<skill-name> $d/; done`
4. Add a row to the README table (the structure test checks every skill name
   appears there).
5. `make test` — or `make test-structure` alone for an ndif skill, which has
   nothing to execute.

`tests/test_structure.py` enforces all of the above plus link resolution, the
plugin manifests, and the pre-0.8 API ban (nnsight only). Adding a *plugin* means
a `plugins/<name>/.claude-plugin/plugin.json`, an entry in
`.claude-plugin/marketplace.json`, and a `/plugin install <name>@ndif-team` line
in the README — the structure test checks all three.

## Writing style for skills

- Lead with what breaks, not with what exists — agents need the failure modes.
- Every nnsight example runs; no pseudo-code presented as code.
- Prefer verified claims ("a GPT-2 block returns a plain tensor") over hedges.
- Cross-link between skills by name (`the debugging skill`) so an agent
  knows where to go next.

## Writing style, ndif plugin

Same rules, plus two of its own:

- **Cite the ndif doc an agent should open for depth**, as a path relative to the
  ndif repo (`docs/operating/models-and-deployment.md`) in backticks — not as a
  markdown link, which the structure test would try to resolve locally.
- **Prefer a verified claim to a doc's claim.** The ndif docs cite `file:line`
  and are mostly current, but a few pages have drifted from the code. When they
  disagree, the code wins.

## Source material

- NDIF server source and docs: `/home/localjadenfk/wd/ndif` (`CLAUDE.md` is a
  router into `docs/`; `docs/concepts/request-lifecycle.md` is the keystone)
- nnsight source and docs: `/home/localjadenfk/wd/nnsight` (branch `0.8`,
  `CLAUDE.md` routes to `docs/`)
- nnterp (0.8 branch): `/home/localjadenfk/wd/nnterp`
- Tutorials and paper implementations: `/home/localjadenfk/wd/nnsight-website/docs`
