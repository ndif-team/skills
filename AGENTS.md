# Working in this repository

Agent skills for the NDIF ecosystem. The README is for people who install the
skills; this file is for whoever edits them.

## Layout

```text
.claude-plugin/marketplace.json      the four plugins; Claude Code and Codex both read it
plugins/<library>/
├── .claude-plugin/plugin.json
└── skills/<library>/
    ├── SKILL.md                     read whenever the skill loads
    ├── references/<guide>.md        one guide per technique or task, opened on demand
    ├── references/<guide>-<topic>.md   depth for one guide
    └── scripts/*.py                 tools the agent runs
```

The libraries are `nnsight`, `nnterp`, `nnspec` and `ndif`. Each has one plugin
holding one skill, and the plugin, the skill and its directory all carry the
library's name.

## One skill per library

An agent sees every installed skill's name and description in every session, in a
listing with a small budget: 1% of the context window in Claude Code, 2% or 8,000
characters in Codex. When the listing overflows, descriptions are dropped, and a
skill with no description is not chosen. Four descriptions fit; thirty did not.

So new material goes into a guide under `references/`, not into a new skill. Add a
skill only for a new library.

## SKILL.md

- **`description`** is what the agent selects on. Say what the skill does, then
  when to use it, in the words a user would type. State when a neighbouring skill
  is the better choice. Keep it under 1,024 characters and write it as a
  double-quoted YAML string, because a bare `: ` inside it is invalid YAML and
  some agents then drop the skill.
- **Frontmatter** holds `name` and `description` only.
- **The body** is the library's core and a `## Guides` section that lists every
  guide with the task it serves. A guide that is not listed there will not be read.
  Keep the body under 500 lines.
- **Which library.** The three interpretability skills carry the same
  `## Which library` table. The order is nnspec, then nnterp, then nnsight: take
  the highest layer that fits the task. A library the request names always wins, and each
  description says so, because an agent decides from the description before it reads the table.
  Change the table in all three files together.

## Guides

- A guide is self-contained: it loads its own model and states its own traps.
- Files in `references/` are flat. A guide's depth lives beside it as
  `<guide>-<topic>.md`, linked from the guide.
- Link between files with relative Markdown links. Name another library's material
  in prose (the `nnsight` skill's debugging guide), because a link across plugins
  does not survive installation.

## Content

- The skills describe nnsight 0.8 with transformers 5 or later. Idioms from older
  versions (`.value`, `nnsight.list()`, `tracer.next()`, `LanguageModel`, proxies)
  appear only in the porting guide.
- Describe current behaviour. Leave out what a feature used to be called or do.
- Lead with what breaks. An agent needs the failure modes more than the tour.
- Every example is real code that was run before it was written down, with the
  output it produced. No pseudo-code.
- Runnable examples use `openai-community/gpt2` and
  `HuggingFaceTB/SmolLM2-135M-Instruct`. Larger models may appear in snippets
  that say they are illustrative. Never name a private checkpoint.
- The `ndif` skill cites the ndif repository's docs as paths in backticks
  (`docs/operating/quickstart.md`). When the docs and the code disagree, the code
  is right.
- Nothing about one machine: no host names, personal paths or environment names.

## Versions

`plugin.json` has no `version` field. Without one, every commit is a new version
and users receive it on their next update. A pinned version keeps them on the copy
they first installed.

## Before pushing

```bash
claude plugin validate .
```

It checks the marketplace, the four manifests and every skill's frontmatter. The
"No version specified" warning is expected.

## Source material

Work from checkouts of the upstream repositories rather than from memory, and run
what you write.

- nnsight: `ndif-team/nnsight`
- nnterp: `ndif-team/nnterp`
- nnspec: `JadenFiotto-Kaufman/nnspec`; `nnspec describe <section>` prints the
  format the installed version accepts
- NDIF server: `ndif-team/ndif`; `docs/concepts/request-lifecycle.md` is the
  keystone
