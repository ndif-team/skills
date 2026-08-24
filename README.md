# Skills for the NDIF Ecosystem

Agent skills for neural network interpretability with [NNsight](https://nnsight.net/)
and [NDIF](https://ndif.us/).

Compatible with both **Claude Code** and **OpenAI Codex** via the
[Agent Skills Specification](https://agentskills.io/).

These skills target **nnsight 0.8** and **transformers ≥ 5**. Every code example is
executed by the test suite against a real model, so what an agent reads is what
actually runs.

The transformers floor is not cosmetic: in 4.x a GPT-2 block returns
`(hidden_states,)` and its attention dropout is `module.attn_dropout(...)`, so the
`.output` and `.source` examples throughout these skills are wrong on 4.x.

## Installation

### Claude Code

```bash
claude

# Add the marketplace (one time)
/plugin marketplace add https://github.com/ndif-team/skills.git

# Install all skills
/plugin install nnsight@skills
```

### OpenAI Codex

```bash
codex

skill-installer install https://github.com/ndif-team/skills.git
```

## Skills

**Foundation**

| Skill | Use when... |
| --- | --- |
| `nnsight` | Anything touching model internals: tracing, reading and modifying activations, batching interventions, gradients, caching, generation, module paths. Start here. |
| `nnsight-debugging` | Code errors, hangs, returns nothing, or silently misbehaves — and for porting pre-0.8 nnsight code. |
| `nnsight-remote` | Running on NDIF: sessions, request batching, download size, non-blocking jobs. |

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
| `vllm` | Throughput, continuous batching, CUDA-graph taps, tensor parallelism, async streaming. |
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

## Example prompts

Once installed, ask naturally:

- "Use logit lens to see what GPT-2 predicts at each layer"
- "Find which attention heads matter for this task with activation patching"
- "Build a steering vector that makes the model more positive"
- "This nnsight script from a paper repo crashes — fix it"
- "Run this experiment on Llama-70B via NDIF without downloading 3 GB of logits"

## Development

Every fenced `python` block in every skill is executed by the test suite. Blocks
in one file share a namespace and run in document order; directives control
execution:

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

`tests/test_structure.py` also enforces packaging: frontmatter matches directory
names, Codex symlinks resolve, manifests are valid, relative links work, and no
pre-0.8 API appears in a runnable example.

### Adding a skill

1. Create `plugins/nnsight/skills/<skill-name>/SKILL.md` with frontmatter:

   ```yaml
   ---
   name: skill-name
   description: What it does and when an agent should load it.
   ---
   ```

2. Put depth in `references/*.md` and runnable tools in `scripts/`; keep
   `SKILL.md` to what an agent should read every time.
3. Link it: `cd .codex/skills && ln -s ../../plugins/nnsight/skills/<skill-name> .`
4. Add a row to the table above.
5. `make test`.

## Structure

```text
skills/
├── .claude-plugin/marketplace.json   # Claude Code marketplace
├── .codex/skills/                    # Codex skills (symlinks)
├── .github/workflows/test.yml        # CPU CI
├── plugins/nnsight/
│   ├── .claude-plugin/plugin.json
│   └── skills/
│       ├── nnsight/
│       │   ├── SKILL.md
│       │   ├── references/*.md
│       │   └── scripts/*.py
│       ├── nnsight-debugging/
│       └── ...
├── tests/                            # executes every code block
└── Makefile
```

## Resources

- [NNsight documentation](https://nnsight.net/)
- [NNsight tutorials](https://nnsight.net/tutorials/)
- [NDIF](https://ndif.us/) — remote access to large models
- [Agent Skills Specification](https://agentskills.io/)
