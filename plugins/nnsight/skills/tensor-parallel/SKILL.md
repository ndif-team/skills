---
name: tensor-parallel
description: Trace a model too big for one GPU by sharding it across several with transformers tensor parallelism — TransformersModel(..., distributed_config=DistributedConfig(tp_size=N)) launched under torchrun. Sharded activations are gathered so interventions read and edit whole tensors exactly as on a single GPU. Use when a checkpoint does not fit on one card, when a user asks about multi-GPU tracing, tp_size, tp_plan, device_map for large models, or hits activation shapes that are a fraction of the expected width (hidden_size/N, intermediate_size/N). Covers the three rules SPMD imposes on intervention code — no rank-dependent control flow, seed before sampling, clone before editing a gathered value — the transformers >= 5.16 requirement, sharded weights as DTensors, and what is not supported.
---

# Tensor Parallelism

Transformers tensor parallelism splits every attention and MLP projection *within*
each layer across GPUs, so a model that does not fit on one card runs on several
at once. (Contrast `device_map="auto"`, which puts whole *layers* on different
GPUs and runs them in sequence — that needs nothing special from nnsight.)

nnsight gathers sharded activations before your intervention sees them and
re-splits what you leave behind, so **the trace you write is the trace you would
write against one GPU**. Nothing to install, import, or enable.

> **Not executed by the repo's test suite in CI** — these examples need ≥2 GPUs
> and a `torchrun` launcher. They were run on 2 and 4 A100s against
> Llama-3.2-3B and Llama-3.3-70B-Instruct, transformers 5.16.1, torch 2.9.1.
> The `max_tp_size` block below is config-only and does run.

## Loading

One process per GPU, so the script is launched with `torchrun`. **Every rank runs
the whole script, including your intervention code.**

<!-- test: skip -->
```python
# tp_trace.py  —  torchrun --nproc_per_node=4 tp_trace.py
import torch
from transformers.distributed import DistributedConfig
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel(
    "meta-llama/Llama-3.2-3B",
    task="text-generation",
    dispatch=True,
    dtype=torch.bfloat16,
    distributed_config=DistributedConfig(tp_size=4),
)
```

**Ask with `distributed_config`, not `tp_plan`.** transformers also accepts a
bare `tp_plan="auto"` and it does shard a checkpoint that publishes a plan — but
that argument goes straight to transformers, so nnsight never sees the request
and never runs the degree check below. On a checkpoint with no plan the result is
silent: gpt2 at `--nproc_per_node=2` loads its full 0.5 GB on *both* ranks,
answers correctly, and leaves `model.interleaver.fragments.enabled` at `False` —
two GPUs doing one GPU's work, with nothing to see. `distributed_config` is the
form that gets checked.

`tp_size` must divide the model's attention heads, key/value heads, and
intermediate size. Check before you allocate anything — this reads the config
alone, so it needs no GPU, no weights and no `torchrun`:

<!-- test: setup -->
```python
from transformers import AutoConfig
from nnsight.modeling.tp import max_tp_size

assert max_tp_size(AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")) == 2
assert max_tp_size(AutoConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")) == 3
assert max_tp_size(AutoConfig.from_pretrained("openai-community/gpt2")) is None
```

Qwen2.5-0.5B stops at 2 because it has 2 key/value heads; SmolLM2-135M at 3, for
the same reason. Every workable degree is a divisor of that number. `None` means
the checkpoint cannot be split at all — it publishes no `base_model_tp_plan`, or its plan uses a
style nnsight refuses. Asking anyway raises `UnshardableCheckpoint` naming the
degrees that would have worked. gpt2 is the familiar case, but it is not only an
old-model problem: several recent Qwen releases publish no plan either.

## Reading and editing

Reading is identical to single-GPU code. A column-parallel output arrives at full
width even though each rank computed a slice of it:

<!-- test: skip -->
```python
with model.trace("The Eiffel Tower is in the city of"):
    gate = model.model.layers[5].mlp.gate_proj.output.save()   # (1, 11, 8192)
    logits = model.lm_head.output.save()
```

**Editing takes a clone.** A gathered value is the output of a collective, and
torch refuses an in-place write into one:

<!-- test: skip -->
```python
gate_proj = model.model.layers[5].mlp.gate_proj

with model.trace(prompt):
    gate_proj.output[..., :3000] = 0          # RuntimeError: Output 0 of
                                              # SliceBackward0 is a view and is
                                              # being modified inplace
```

Clone, edit, assign back:

<!-- test: skip -->
```python
with model.trace(prompt):
    edited = gate_proj.output.clone()
    edited[..., :3000] = 0
    gate_proj.output = edited
    logits = model.lm_head.output.save()
```

Replacing the whole value (`gate_proj.output = torch.zeros_like(...)`) needs no
clone. nnsight does not clone for you: on a large model that is a copy the size
of the activation, on every gather, for the many traces that only read.

## What is actually sharded

Most of what people read is already whole and costs nothing — row-parallel layers
all-reduce their output, so decoder layers, `self_attn`, `mlp` and the final norm
arrive complete.

| Value | Sharded? |
|---|---|
| Column-parallel **output** (`q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`) | yes — gathered for you |
| Row-parallel **input** (`o_proj.input`, `down_proj.input`) | yes — gathered for you |
| Row-parallel output (`o_proj.output`, `down_proj.output`) | no, all-reduced |
| Whole modules (`layers[i].output`, `mlp.output`, `norm.output`) | no |
| `lm_head.output` | no, gathered by transformers |
| **Parameters** (`q_proj.weight`) | **yes — not gathered**, see below |
| Values *between* two sharded modules (`mlp.act_fn.output`, `query_states_0`) | **yes — not gathered**, see below |
| `embed_tokens.output` | whole — unless the plan shards it, and then not readable, see below |

The gather only fires when an intervention is parked on that location, so reading
a few locations does not pay for the hundreds you ignore. `tracer.cache()` gathers
only the modules it selects.

**A vocab-parallel embedding's own output cannot be read.** Only some plans shard
the embedding — those that do name it `embedding_rowwise`, which in practice means
the tied-embedding checkpoints (Llama-3.2-1B and -3B, and most small models).
Where it is sharded, the value's layout is single-use, and parking on
`embed_tokens.output` raises a bare `AssertionError` from torch's embedding op.
The same takes out `tracer.cache()` called with *no* arguments, which selects
every module and so reaches the embedding.

<!-- test: skip -->
```python
"embed_tokens" in (model.config.base_model_tp_plan or {})
# True  — Llama-3.2-1B, Llama-3.2-3B     embed_tokens.output raises
# False — Llama-3.3-70B, Qwen3-8B        embed_tokens.output reads whole
```

`model.model.layers[0].input` works either way — the same tensor one module later,
whole at any degree. Name the modules you want to cache rather than caching all.

## Values between two sharded modules

A module's own `.input`/`.output` is made whole. A value *between* two sharded
modules is not — nothing on it records which axis holds the shard, and the axis
moves (attention's `view`/`transpose` puts it on the head dimension). You know
what the forward did, so you name the axis:

<!-- test: skip -->
```python
from nnsight.modeling.tp import gather, shard

attn = model.model.layers[1].self_attn

with model.trace(prompt):
    q = attn.source.query_states_0.output       # (1, heads/tp_size, seq, head_dim)
    heads = gather(model, q, dim=1).clone()     # (1, heads, seq, head_dim)
    heads[:, 3] = 0                             # ablate head 3, whoever holds it
    attn.source.query_states_0.output = shard(model, heads, dim=1)
    logits = model.lm_head.output.save()
```

Both are **collectives**, so rule 1 below applies with full force: every rank
must reach them, so call them unconditionally. Both are no-ops on an unsharded
model, so the same script runs at any degree.

A `.source` value taken from *inside* a sharded module is the case that catches
people out. It is a `DTensor`: its `.shape` reports the **whole** while its data
is this rank's slice, so nothing looks wrong and `float(v.sum())` is quietly
wrong. Comparing widths at tp=1 and tp=2 does not detect it — the width never
changes. Test with `v.placements` or `isinstance(v, DTensor)`, and reassemble
with `gather(model, v, dim=...)` (which reads the layout off the value itself) or
`v.full_tensor()`.

## Parameters are DTensors

`layer.weight` is a `DTensor`: this rank holds `1/tp_size` of it, but **`.shape`
reports the whole**, so shape alone will not tell you it is split.

<!-- test: skip -->
```python
w = model.model.layers[0].self_attn.q_proj.weight
w.shape                 # (8192, 8192) — the global shape
w.placements            # (Shard(dim=0),)
w.to_local().shape      # (2048, 8192) — what this rank holds
w.full_tensor()         # the whole thing; every rank must call it, and it
                        # allocates the full tensor on each of them
```

nnsight does not reassemble weights for you: they are what tensor parallelism
exists to split.

**A reduction over a sharded weight returns this rank's answer, with no error.**
`w.mean()`, `w.norm()` and `w.abs().max()` come back as a `DTensor` with a
`Partial` placement — the reduction over this rank's slice, still waiting to be
combined. `float()` reads that partial number, which differs between ranks and
matches none of them. Llama-3.2-3B at tp=2:

<!-- test: skip -->
```python
float(w.mean())                  # rank 0:  2.256e-05   rank 1: -1.171e-05
float(w.mean().full_tensor())    # 5.421e-06 on both — the real mean
float(w.norm())                  # rank 0: 81.80        rank 1: 68.97
float(w.norm().full_tensor())    # 106.996
```

Call `.full_tensor()` on the **reduction**, not on the weight: the result is a
scalar, so it costs one small collective rather than a copy of the layer. Any
weight-norm sweep or layer-magnitude plot needs this or it plots per-rank noise.
(`w.cpu()` is still a `DTensor`, so moving it off the GPU does not help.)

**A sharded weight cannot be edited in place through the `DTensor`.**
`weight[:, :10] = 0` raises `NotImplementedError: Operator aten.fill_.Tensor does
not have a sharding strategy registered`. Write through `.to_local()`, which is
the right thing anyway — each rank edits the rows or columns it holds:

<!-- test: skip -->
```python
with torch.no_grad():
    model.model.layers[0].mlp.gate_proj.weight.to_local()[:, :10] = 0
```

Out-of-place arithmetic on the whole (`weight.mul_(0.5)`) works unchanged. A
rank-one (ROME-style) update written against a whole weight matrix has to be
expressed per-rank the same way — see the `model-editing-and-lora` skill for the
single-GPU form.

## The three rules for intervention code

Every rank runs your block — that is what keeps the collectives lined up.

**1. No rank-dependent control flow.** Nothing may branch on rank or take a
different path on different ranks; the ranks stop agreeing on when to gather and
the run deadlocks. It deadlocks quietly — no exception, no watchdog for minutes —
and killing `torchrun` does not take the rank processes with it. Find their pids
and kill those, or the cards stay occupied by the run you thought you ended.

**2. Seed before you sample.** This is a *correctness* bug, not an inconsistency.
`torch.initial_seed()` differs per rank under `torchrun` — the ranks are seeded
randomly, not from the rank. If sampling diverges the ranks generate different
tokens, and the model's own all-reduces then sum activations computed from
different sequences: the output is wrong on every rank, not merely different.
Many checkpoints ship `do_sample: true` in `generation_config.json`, so it bites
without asking:

<!-- test: skip -->
```python
# Either force greedy...
with model.generate(prompt, max_new_tokens=20, do_sample=False) as tracer:
    out = tracer.result.save()

# ...or seed identically on every rank before sampling.
torch.manual_seed(0)
with model.generate(prompt, max_new_tokens=20) as tracer:
    out = tracer.result.save()
```

**3. Clone before editing a gathered value** (see "Reading and editing").

Every rank computes the **same saved values** (they come from gathered tensors),
so print or write results from one rank or you get N copies:

<!-- test: skip -->
```python
import os
if int(os.environ.get("RANK", 0)) == 0:
    print(logits.argmax(-1))
```

That `if` is around the *print*, outside the trace — the one place a rank test is
safe. Never inside a block.

## Requires transformers >= 5.16

5.16 rebuilt tensor parallelism on DTensor: the sharding plan moved from a stamp
on each module to a single `_tp_plan` on the model, and the style's collectives
moved into a wrapper around `module.forward`. nnsight reads the layout the way
that backend describes it, so on an older transformers it would find nothing
sharded — which does not fail, it hands intervention code one rank's slice as
though it were the whole tensor. nnsight refuses to load rather than let that
through, though the refusal is not always worded as a version: on transformers
5.15 the load stops at `ModuleNotFoundError: No module named
'transformers.distributed.tensor_parallel'`. Check `transformers.__version__`
before reading further into it.

## Expert parallelism

A mixture-of-experts checkpoint can be split by *expert* instead of along the
feature dimension:

<!-- test: skip -->
```python
model = TransformersModel(
    "Qwen/Qwen1.5-MoE-A2.7B",
    task="text-generation",
    distributed_config=DistributedConfig(tp_size=2, enable_expert_parallel=True),
    dispatch=True,
)
```

Traces read the same as any other — `mlp.experts.output` and `mlp.gate.output`
come back at their single-GPU shapes and values, and edits to them carry through.
The degree has to divide the **expert count** rather than the head counts, and
nnsight checks that before the weights are fetched. For the pre-flight, ask
`max_tp_size(config, expert_parallel=True)` — it reads `base_model_ep_plan`, which
a checkpoint can publish without any tensor-parallel plan at all.

Plain tensor parallelism on an MoE checkpoint is also fine. `moe_tp_experts` —
what Mixtral, DeepSeek-V3, Qwen3-MoE and ~25 other shipped configs use — is a
partial sum at the handoff that the style's own transform all-reduces; nnsight
all-reduces it for a waiting worker and hands back the whole on rank 0 and zeros
elsewhere, so the read is whole and the model's reduce still lands exactly once.

## Not supported

Four styles are refused rather than guessed at: `megamoe_router`,
`megamoe_experts`, `moe_identity_expert`, and MLA's split kv projection
(`mla_kv_a_proj`). They slice by expert or into a fused projection, so neither
the gather nor the re-split means anything for them. A plan containing one
supports no degree at all, so a load asking for tensor parallelism is turned away
by `UnshardableCheckpoint` — the same refusal as a checkpoint with no plan.
DeepSeek-V2-Lite is the one to know: `mla_kv_a_proj` puts tensor parallelism out
of reach, while expert parallelism is still open to it.

They are refused because no model in the test set exercises them, not because
they are known to be ungatherable. `ep_router` and `grouped_gemm` sat in this
list until a model using them was run end to end, at which point both turned out
to need no gather at all.

## Diagnosing

Three of these arrive wrapped in a pipeline `ValueError: Could not load model
...`, with the real message near the end of it.

| Symptom | Cause |
|---|---|
| `KeyError: 'RANK'` | Started with `python`, not `torchrun`. TP needs the calling process to be a rank, which is also why it cannot run in a notebook kernel |
| `tp_size (2) * fsdp_size (1) is not equal to world_size (4)` | `--nproc_per_node` must equal `tp_size` |
| `` `tp_plan` and `device_map` are mutually exclusive `` | Pick one — they are two different ways of spreading a model |
| `UnshardableCheckpoint` | No `base_model_tp_plan`, a refused style, or a degree that divides nothing. The message lists the degrees that work. Load on one GPU or with `device_map` |
| Nothing raised, but `fragments.enabled` is `False` and every rank holds the whole model | `tp_plan=` instead of `distributed_config=` — the degree check never ran |
| Activation width is `hidden_size/N` or `intermediate_size/N` | A value between two sharded modules. `gather(model, v, dim=...)`, naming the axis |
| Shape looks right but the numbers are wrong, and differ per rank | A `DTensor` reporting the global shape: a `.source` value inside a sharded module, or a reduction over a weight. Check `v.placements`; fix with `.full_tensor()` |
| Bare `AssertionError`, empty message | `embed_tokens.output` on a plan that shards the embedding, or a `tracer.cache()` with no arguments that reaches it. Read `layers[0].input` and name the modules to cache |
| `RuntimeError: Output 0 of SliceBackward0 is a view and is being modified inplace` | Editing a gathered value in place — clone, edit, assign back |
| `NotImplementedError: Operator aten.fill_.Tensor does not have a sharding strategy` | Editing a sharded weight in place — write through `.to_local()` |
| Run hangs with no error | The ranks diverged — rank-dependent control flow, or an exception on one rank only. Kill the rank pids, not just `torchrun` |
| Generated text differs per rank | Sampling without an identical seed (rule 2) |
| `ModuleNotFoundError: transformers.distributed.tensor_parallel` | transformers < 5.16 — upgrade, or load on one GPU |
| Results differ from single-GPU by ~1e-3 in bfloat16 | Expected: an all-reduce sums in a different order than one matmul. In float32 the same gap is ~1e-6; anything larger there is a real difference. Greedy token choices are identical either way |

## Choosing between this and vLLM

Both shard across GPUs. Use **tensor parallelism** when you want ordinary
`TransformersModel` semantics — `tracer.result`, list inputs, the full tracing API
— on a model that does not fit. Use **vLLM** (see the `vllm` skill) when you need
throughput, continuous batching, or async streaming, and can accept its
differences (one prompt per invoke, `model.logits` / `model.samples`).
