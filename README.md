# Skills for the NDIF Ecosystem

Agent skills for neural network interpretability with [NNsight](https://nnsight.net/)
and [NDIF](https://ndif.us/).

Compatible with both **Claude Code** and **OpenAI Codex** via the
[Agent Skills Specification](https://agentskills.io/).

Two plugins: **nnsight**, for writing interpretability code, and **ndif**, for
running your own NDIF server.

Every code example in the nnsight skills is executed by the test suite against a
real model, so what an agent reads is what actually runs.

**Requires** nnsight 0.8 and transformers ≥ 5.

## Installation

### Claude Code

```bash
claude

# Add the marketplace (one time)
/plugin marketplace add https://github.com/ndif-team/skills.git

# Install the interpretability skills
/plugin install nnsight@ndif-team

# Install the NDIF self-hosting skills (optional — only if you run your own server)
/plugin install ndif@ndif-team

# Check it worked — the skills should be listed
/plugin
```

### OpenAI Codex

```bash
codex

$skill-installer install https://github.com/ndif-team/skills

# Check it worked — the nnsight skills should be listed
$skill-installer list
```

## Using them

You do not invoke a skill by name. Both Claude Code and Codex read every installed
skill's `description` and load the ones that match what you are asking for, so you
just describe the task — see [Example prompts](#example-prompts) below.

## nnsight skills

**Foundation**

| Skill | Use when... |
| --- | --- |
| `nnsight` | Anything touching model internals: tracing, reading and modifying activations, batching interventions, gradients, caching, generation, module paths. Start here. |
| `debugging` | Code errors, hangs, returns nothing, or silently misbehaves — and for porting pre-0.8 nnsight code. |
| `remote` | Running on NDIF: sessions, request batching, download size, non-blocking jobs. |

**Techniques**

| Skill | Use when... |
| --- | --- |
| `logit-lens` | Decoding what each layer predicts; tracking where an answer emerges. |
| `activation-patching` | Locating the layers, positions, or heads that carry a behavior; DAS. |
| `attribution-patching` | Scaling patching to whole models with a gradient approximation. |
| `causal-tracing` | Corrupt-and-restore factual localization (ROME-style). |
| `ablation` | Testing necessity — zero, mean, resample, and noise ablation. |
| `attention-analysis` | Attention patterns, per-head metrics, induction/copy head detection. |
| `circuit-discovery` | Finding and validating the subgraph behind a task (IOI-style). |
| `probing` | Training classifiers on activations; what is linearly decodable, and whether it is used. |
| `sae-and-dictionary-learning` | Feature-level analysis; attaching, training, and evaluating SAEs. |
| `model-steering` | Steering vectors, function vectors, persistent behavioral edits. |
| `model-editing-and-lora` | Weight edits, ROME-style updates, adapters trained through a frozen model. |
| `interp-experiment-design` | Choosing a metric, controls, and sanity checks before running anything. |

**Runtimes and tooling**

| Skill | Use when... |
| --- | --- |
| `nnterp` | Writing one script that runs unchanged across GPT-2, Llama, Qwen, Gemma. |
| `vllm` | Throughput, continuous batching, CUDA-graph taps, tensor parallelism, `model.edit()` sweeps, nnsight-serve, async streaming — and what a block sees differently on vLLM. |
| `tensor-parallel` | A model too big for one GPU, sharded across several with `transformers` TP under `torchrun`. |
| `quantization` | A model too big for one GPU, held in 4 or 8 bits — `dtype="nf4"`, `"int8"`, ... |
| `diffusion-and-multimodal` | VLMs, diffusion pipelines, the diffusion lens, non-text tasks. |

The `nnsight` skill carries a `references/` tree (execution model, batching,
gradients, source tracing, per-architecture module paths, full API tables) that
agents load on demand, plus runnable helper scripts:

```bash
# module paths, execution order, tensor-vs-tuple — without downloading weights
python plugins/nnsight/skills/nnsight/scripts/inspect_model.py meta-llama/Llama-3.1-8B --prompt "Hello"

# versions, GPUs, NDIF key/host, deployed models, local-vs-NDIF package diff
python plugins/nnsight/skills/nnsight/scripts/check_env.py --remote
```

## NDIF (self-hosting)

A second plugin, for people and agents who run their **own** NDIF server — the
backend behind nnsight's `remote=True` — rather than using the public
[ndif.us](https://ndif.us/) service. (For that, the `remote` skill above is what
you want.)

```bash
/plugin install ndif@ndif-team
```

| Skill | Use when... |
| --- | --- |
| `selfhost` | Standing a server up: the published `ndif/ndif` image, the compose dev stack, or a from-source `ndif start`. Prerequisites, tags, ports, volumes, configuration, and the first remote trace. |
| `operate` | Running models on it: deploy, evict, pin, scale, `models.yaml`, sizing and padding, HOT/WARM/COLD, the dashboard, telemetry, turning on auth. |
| `troubleshoot` | It won't start, requests hang, a deploy OOMs, a result won't download, versions disagree — symptom to cause to fix, and where the logs actually are. |
| `develop` | Changing the server itself: the request lifecycle, the process map, trusted vs untrusted execution, the model-actor hooks, the test suite, release mechanics. |

These skills document a server, so — unlike the nnsight ones — their code blocks
are reference material rather than executed examples.

Example prompts:

- "Run NDIF on my own GPU with docker"
- "Point nnsight at my local NDIF instead of ndif.us"
- "Deploy Llama-3.1-8B on my NDIF and pin it"
- "My NDIF says the compute backend is reconnecting"
- "Why does my trace OOM with 'MiB allowed' on an empty GPU?"

## Example prompts

Once installed, ask naturally:

- "Use logit lens to see what GPT-2 predicts at each layer"
- "Find which attention heads matter for this task with activation patching"
- "Build a steering vector that makes the model more positive"
- "This nnsight script from a paper repo crashes — fix it"
- "Run this experiment on Llama-70B via NDIF without downloading 3 GB of logits"

## Development

Every fenced `python` block in every **nnsight** skill is executed by the test
suite (`tests/docblocks.py` sets `SKILLS_ROOT`). Blocks in one file share a
namespace and run in document order; directives control execution:

```markdown
<!-- test: skip -->                   don't run (still syntax-checked)
<!-- test: skip nocompile -->         don't run, don't compile
<!-- test: setup -->                  a block later ones build on
<!-- test: remote -->                 only with NDIF_HOST set
<!-- test: gpu -->                    only with CUDA
<!-- test: slow -->                   only with --run-slow
<!-- test: expect-error ValueError --> must raise this
```

```bash
make test              # everything, including NDIF_HOST=http://localhost:8001
make test-local        # skip anything needing an NDIF deployment
make test-structure    # packaging only — fast, no model loading
make test-skill SKILL=nnsight
make report            # per-file table of blocks run / skipped
```

`tests/test_structure.py` enforces packaging for **every** plugin listed in
`.claude-plugin/marketplace.json`: the plugin manifest exists and its name
matches, frontmatter matches directory names, Codex symlinks resolve, relative
links work, and (for the nnsight plugin) no pre-0.8 API appears in a runnable
example.

The ndif skills document a server rather than a library, so their `python` blocks
are all marked `<!-- test: skip -->` and are reference material, not examples the
suite runs.

### Adding a skill

1. Create `plugins/<plugin>/skills/<skill-name>/SKILL.md` with frontmatter:

   ```yaml
   ---
   name: skill-name
   description: What it does and when an agent should load it.
   ---
   ```

2. Put depth in `references/*.md` and runnable tools in `scripts/`; keep
   `SKILL.md` to what an agent should read every time.
3. Link it into both Codex trees:
   `for d in .agents/skills .codex/skills; do ln -s ../../plugins/<plugin>/skills/<skill-name> $d/; done`
4. Add a row to the table above.
5. `make test`.

## Structure

```text
skills/
├── .claude-plugin/marketplace.json   # Claude Code marketplace
├── .agents/skills/                   # Codex skills (symlinks)
├── .codex/skills/                    # Codex skills, older CLI path (symlinks)
├── .github/workflows/test.yml        # CPU CI
├── plugins/
│   ├── nnsight/
│   │   ├── .claude-plugin/plugin.json
│   │   └── skills/
│   │       ├── nnsight/
│   │       │   ├── SKILL.md
│   │       │   ├── references/*.md
│   │       │   └── scripts/*.py
│   │       ├── debugging/
│   │       └── ...
│   └── ndif/
│       ├── .claude-plugin/plugin.json
│       └── skills/
│           ├── selfhost/
│           ├── operate/
│           ├── troubleshoot/
│           └── develop/
├── tests/                            # executes every code block
└── Makefile
```

## Resources

- [NNsight documentation](https://nnsight.net/)
- [NNsight tutorials](https://nnsight.net/tutorials/)
- [NDIF](https://ndif.us/) — remote access to large models
- [Agent Skills Specification](https://agentskills.io/)
