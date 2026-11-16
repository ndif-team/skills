# Skills for the NDIF ecosystem

Agent skills that teach a coding agent to do neural network interpretability with
[nnsight](https://nnsight.net/), [nnterp](https://github.com/ndif-team/nnterp) and
[nnspec](https://github.com/JadenFiotto-Kaufman/nnspec), and to run its own
[NDIF](https://ndif.us/) server. They follow the
[Agent Skills specification](https://agentskills.io/), so they work in Claude Code,
OpenAI Codex, and any other agent that loads skills.

## The four skills

There is one skill per library, named after it.

| Skill | What the agent uses it for |
| --- | --- |
| `nnspec` | Running an experiment as one JSON document, with no intervention code: lenses, patching, ablation, attention, steering, probes, SAEs, DAS. |
| `nnterp` | Writing interpretability code once, against one module vocabulary and one set of standard values, so it runs on 98 transformer families. |
| `nnsight` | Reading and editing the internals of any PyTorch model: tracing, interventions, gradients, remote execution on NDIF, debugging, vLLM, tensor parallelism, and a guide for each technique. |
| `ndif` | Running your own NDIF server: standing it up, deploying and sizing models, fixing it, and changing its code. |

The three interpretability skills name each other. When a request does not name a
library, the agent is told to take the highest layer that fits: `nnspec` for a
standard experiment, `nnterp` when the experiment needs Python that a document
cannot express, and `nnsight` when nnterp does not support the model family or the
work needs access below nnterp's standard values. Install all three so the agent
can make that choice.

The skills describe nnsight 0.8 with transformers 5 or later.

## Install

### Claude Code

```text
/plugin marketplace add ndif-team/skills
/plugin install nnspec@ndif-team
/plugin install nnterp@ndif-team
/plugin install nnsight@ndif-team
/plugin install ndif@ndif-team
```

The last one is only for people who run their own NDIF server.

### OpenAI Codex

```bash
codex plugin marketplace add ndif-team/skills
codex plugin add nnspec@ndif-team
codex plugin add nnterp@ndif-team
codex plugin add nnsight@ndif-team
codex plugin add ndif@ndif-team
```

### Other agents

[`npx skills`](https://github.com/vercel-labs/skills) installs the same four skills
into Cursor, GitHub Copilot, Gemini CLI, OpenCode, Amp, Cline and others:

```bash
npx skills add ndif-team/skills
```

To install by hand, copy a skill's folder, for example
`plugins/nnsight/skills/nnsight`, into the directory your agent reads skills from.
For most agents that is `.agents/skills/` in a project or `~/.agents/skills/` for
every project.

### Updating

Every commit to this repository is a new version of each plugin.

- Claude Code: `/plugin marketplace update ndif-team`, or turn on auto-update for
  the marketplace under `/plugin`.
- Codex: `codex plugin marketplace upgrade`.
- `npx skills`: `npx skills update`.

## Getting the agent to use them

You do not call a skill by name. At the start of a session the agent sees each
installed skill's name and description, and loads a skill when the request matches
its description. Describe the task and the agent does the rest:

- "Run a logit lens on GPT-2 for this prompt"
- "Find which attention heads matter for this task with activation patching"
- "This nnsight script from a paper repo crashes, fix it"
- "Run this experiment on Llama-70B through NDIF without downloading 3 GB of logits"
- "Deploy Llama-3.1-8B on my own NDIF and pin it"

Agents are more likely to skip a skill when a request does not use the words in
its description. In a project that works on model internals, add this to the
project's `AGENTS.md` or `CLAUDE.md` so that the choice is not left to chance:

```markdown
## Interpretability work

For any task that reads or changes a model's internals (activations, attention,
logit lens, patching, ablation, steering, probes, SAEs), load the matching skill
before writing code: `nnspec` for a standard experiment, `nnterp` when the
experiment needs custom Python, `nnsight` for an unsupported architecture, raw
module access, or existing nnsight code. For a self-hosted NDIF server, load `ndif`.
```

## What each skill contains

A skill is a `SKILL.md` that the agent reads when the skill loads, and guides under
`references/` that it opens for the task at hand.

**`nnspec`**: the document format (model, data, sites, positions, reads and writes,
ops, sweeps, saves), running and the engines, reading a run back. Guides: patching,
ablation, lenses and attention, steering and generation, trained featurizers (DAS,
masks, SAEs, probes, LoRA), debugging.

**`nnterp`**: the vocabulary, the standard values, vision-language models,
`support()`, the families. Guides: patterns (logit lens, direct logit attribution,
ablation, patching, steering, probing, cross-family sweeps) and extending (adding or
overriding a family).

**`nnsight`**: the execution model, reading and modifying activations, batching,
generation, gradients, caching, `.source`, module paths per architecture, the API
tables. Guides: debugging, remote (NDIF), logit lens, activation patching,
attribution patching, causal tracing, ablation, attention analysis, circuit
discovery, probing, SAEs and dictionary learning, model steering, model editing and
LoRA, experiment design, vLLM, tensor parallelism, quantization, diffusion and
multimodal. Two scripts print a model's module paths without downloading weights
and report the environment.

**`ndif`**: guides for self-hosting (the published image, the compose stack, a
from-source install), operating (deploy, evict, pin, scale, sizing, auth),
troubleshooting (symptom to cause to fix, reading logs) and developing (the request
lifecycle, model actors, tests, releases).

## Contributing

[AGENTS.md](AGENTS.md) has the layout and the rules for editing a skill.

## Resources

- [NNsight documentation](https://nnsight.net/)
- [NDIF](https://ndif.us/), remote access to large models
- [Agent Skills specification](https://agentskills.io/)
