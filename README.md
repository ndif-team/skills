# Skills for the NDIF Ecosystem

Agent skills for neural network interpretability with [NNsight](https://nnsight.net/)
and [NDIF](https://ndif.us/).

Compatible with both **Claude Code** and **OpenAI Codex** via the
[Agent Skills Specification](https://agentskills.io/).

Four plugins: **nnsight**, for writing interpretability code, **nnterp**, for writing it
once against one module vocabulary that runs on every transformer family, **nnspec**, for
running an experiment as a JSON document instead of code, and **ndif**, for running your
own NDIF server.

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

# Install the nnterp skills (one vocabulary across transformer families, on nnsight)
/plugin install nnterp@ndif-team

# Install the nnspec skills (experiments as JSON documents, no code)
/plugin install nnspec@ndif-team

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

## nnterp (one vocabulary across families)

[nnterp](https://github.com/ndif-team/nnterp/tree/0.8-refactor) is a layer on nnsight 0.8: a
`StandardizedTransformer` whose modules answer to Llama's names on every family
(`model.layers[i].self_attn`, `model.layers[i].mlp`, `self_attn.o_proj`, `mlp.down_proj`,
`model.norm`, `model.lm_head`) and carry standard values (`layer_output`, `attention_output`,
`mlp_output`, the module boundaries between them such as `layer_mid`, `attention_probabilities`,
the attention interior, `mlp_neurons`, `logits`, the gated-DeltaNet state on hybrids) that mean
the same thing on all 98 transformer families it knows, with `model.support()` saying what a
checkpoint has before any trace. It needs nnsight `dev` at `d565f5f8` or later.

```bash
/plugin install nnterp@ndif-team
```

| Skill | Use when... |
| --- | --- |
| `nnterp` | Writing code against model internals that must run on more than one architecture, or that wants the residual stream, sublayer contributions, attention pattern or DeltaNet state without per-family module paths and tuple unwrapping. Start here. |
| `patterns` | Logit lens, steering, attention patterns, ablation, activation patching, direct logit attribution, probing, cross-family sweeps and DeltaNet state patching, written once against the standard values. |
| `extending` | A checkpoint nnterp does not know, a family whose value the base gets wrong, or a value of your own: writing a family module, overriding values, finding `.source` operation names, registering from outside the package. |

Its code blocks are executed like the nnsight ones, against `openai-community/gpt2` and
`HuggingFaceTB/SmolLM2-135M-Instruct`; they skip when `nnterp` is not installed.

## nnspec (experiments as documents)

[nnspec](https://github.com/JadenFiotto-Kaufman/nnspec) runs an interpretability experiment from one JSON
document: the checkpoint, the data, the forwards and generations, the reads and writes on the model's
internals, the ops that score them, and what to save. The same document runs on a local model (through
nnterp and nnsight) or on an NDIF server, and a run writes tensors with a manifest that `nnspec table`
prints. The skills are for agents *using* nnspec, not changing it.

```bash
/plugin install nnspec@ndif-team
```

| Skill | Use when... |
| --- | --- |
| `nnspec` | Writing or running any nnspec document: the format (model, data, sites, positions, reads and writes, ops, sweeps, saves), `nnspec run`, the engines, reading a run back. Start here. |
| `nnspec-patching` | Which layer, position, head, neuron or path carries a behaviour: activation, head, neuron, path and attention-pattern patching, layer-by-position maps, attribution patching. |
| `nnspec-ablation` | What a component contributes when removed: zero, mean, resample, optimal-constant, direction, noise and clamp ablation, head and key knockout, expert ablation, the hydra effect. |
| `nnspec-lenses-and-attention` | What each layer predicts and where a head looks: logit and tuned lenses, readout, attention patterns, previous-token and induction stripes. |
| `nnspec-steering-and-generation` | Pushing a model with a direction and scoring what it generates: steering vectors and their controls, did-it-say-the-answer, confidence, chat turns, teacher forcing. |
| `nnspec-trained-featurizers` | Anything trained inside a run or loaded from a bundle: DAS and DBM, head masks, SAEs, probes, LoRA, cached harvests. |
| `nnspec-debugging` | A refusal, a run that stops, numbers that are wrong when nothing was refused, two runs that differ, reproducing or cross-checking a result. |

Every complete document in these skills is compiled by the test suite against nnspec (no model loads), so
field, op and component names cannot drift; the numbers quoted come from runs on `openai-community/gpt2`
and `HuggingFaceTB/SmolLM2-135M-Instruct`.

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
- "Write an nnspec document that patches each layer's residual stream at the last token and tables the effect"

## Development

Every fenced `python` block in every **nnsight** and **nnterp** skill is executed by
the test suite (`tests/docblocks.py` lists them in `EXECUTED_PLUGINS`). Blocks in one file share a
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

The nnspec skills' examples are JSON documents. `tests/test_documents.py` parses every
`json` block under `plugins/nnspec` and compiles each complete document (one with `model`
and `steps`) through `nnspec.Plan(...).compile()`, which needs no model; python blocks
there are syntax-checked only.

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
│   ├── nnterp/
│   │   ├── .claude-plugin/plugin.json
│   │   └── skills/
│   │       ├── nnterp/
│   │       ├── patterns/
│   │       └── extending/
│   ├── nnspec/
│   │   ├── .claude-plugin/plugin.json
│   │   └── skills/
│   │       ├── nnspec/
│   │       ├── nnspec-patching/
│   │       └── ...
│   └── ndif/
│       ├── .claude-plugin/plugin.json
│       └── skills/
│           ├── selfhost/
│           ├── operate/
│           ├── troubleshoot/
│           └── develop/
├── tests/                            # executes every code block; compiles every nnspec document
└── Makefile
```

## Resources

- [NNsight documentation](https://nnsight.net/)
- [NNsight tutorials](https://nnsight.net/tutorials/)
- [NDIF](https://ndif.us/) — remote access to large models
- [Agent Skills Specification](https://agentskills.io/)
