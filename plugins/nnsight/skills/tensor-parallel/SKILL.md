---
name: tensor-parallel
description: Trace a model too big for one GPU by sharding it with transformers tensor parallelism — TransformersModel(..., distributed_config=DistributedConfig(tp_size=N)) under torchrun, with the full API (gradients, .source, list batches); for throughput, serving or streaming without those, see the vllm skill. Use when a checkpoint does not fit on one card, for multi-GPU tracing, tp_size or tp_plan, or activation shapes a fraction of the expected width (hidden_size/N). Covers the rules SPMD puts on intervention code (no rank-dependent control flow, seed before sampling, clone before editing any value a sharded module produced), the transformers >= 5.19 requirement, sharded weights as DTensors, and what is not supported.
---

# Tensor Parallelism

Transformers tensor parallelism splits every attention and MLP projection *within*
each layer across GPUs. nnsight gathers sharded activations before your
intervention sees them and re-splits what you leave behind, so **the trace you
write is the trace you would write against one GPU**. (`device_map="auto"`, which
puts whole layers on different GPUs, needs nothing from this skill.)

> Not executed in CI: needs ≥2 GPUs and torchrun. The `max_tp_size` block below
> is config-only and does run.

## Loading

One process per GPU under `torchrun`. **Every rank runs the whole script,
including your intervention code.**

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

`tp_plan="auto"` is degree-checked against the world size the same way; a custom
plan (a dict of patterns) loads unchecked. Prefer `DistributedConfig(tp_size=N)`:
it states the degree, and it is the only form that can ask for expert parallelism.

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

Both stop at their key/value head count; every workable degree divides it.
`None` means the checkpoint cannot be split at all — no `base_model_tp_plan`
(gpt2, and several recent Qwen releases), or a refused style. Asking anyway raises
`UnshardableCheckpoint` naming the degrees that would have worked.

## Reading and editing

Reading is identical to single-GPU code: `gate_proj.output` arrives at its full
`(1, seq, 8192)` even though each rank computed a slice of it.

**Editing takes a clone — on any value a sharded module produced.** Whether
nnsight gathered it or transformers all-reduced it, the whole value you see is the
output of a collective, and torch refuses an in-place write into one. That covers
the gathered outputs and also `o_proj.output`, `down_proj.output`, `mlp.output`,
the tensor in `self_attn.output`, an MoE's `experts.output` and `lm_head.output` —
all of which take in-place writes on one GPU. `gate_proj.output[..., :3000] = 0`
raises `RuntimeError: Output 0 of SliceBackward0 is a view and is being modified
inplace`. Clone, edit, assign back:

<!-- test: skip -->
```python
gate_proj = model.model.layers[5].mlp.gate_proj

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

| Value | Arrives |
|---|---|
| Column-parallel **output** (`q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`) | sharded — gathered for you |
| Row-parallel **input** (`o_proj.input`, `down_proj.input`) | sharded — gathered for you |
| Row-parallel output (`o_proj.output`, `down_proj.output`) | whole by all-reduce — clone to edit |
| Whole modules (`layers[i].output`, `self_attn`, `mlp.output`, `norm.output`) | whole — clone to edit `self_attn`/`mlp` |
| `lm_head.output` | whole, gathered by transformers — clone to edit |
| **Parameters** (`q_proj.weight`) | **sharded, not gathered** — see below |
| Values *between* two sharded modules (`mlp.act_fn.output`, `query_states_0`) | **sharded, not gathered** — see below |
| `embed_tokens.output` | whole — unless the plan shards it, and then not readable, see below |

The gather only fires where an intervention is parked, so reading a few
locations does not pay for the hundreds you ignore.

**A vocab-parallel embedding's own output cannot be read.** Plans that shard the
embedding name it `embedding_rowwise` — in practice the tied-embedding checkpoints
(Llama-3.2-1B and -3B, most small models). There, parking on `embed_tokens.output`
raises a bare `AssertionError` from torch's embedding op, and so does
`tracer.cache()` with *no* arguments, which reaches the embedding. Check with
`"embed_tokens" in (model.config.base_model_tp_plan or {})`. Read
`model.model.layers[0].input` instead — the same tensor one module later, whole at
any degree — and name the modules you cache.

## Values between two sharded modules

A module's own `.input`/`.output` is made whole; a value *between* two sharded
modules is not, and nothing on it records which axis holds the shard (attention's
`view`/`transpose` moves it to the head dimension). Name the axis yourself:

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

Both are **collectives**: every rank must reach them (rule 1), so call them
unconditionally. Both are no-ops on an unsharded model.

**A one-GPU run validates nothing.** At `tp_size=1` the value comes back
untouched, so a wrong `dim` is accepted silently and only raises (or garbles the
tensor) on a sharded run; an in-place edit with no clone works at `tp_size=1` and
fails under TP. Smoke-test a gather/shard block at `tp_size=2` before trusting it.

A `.source` value from *inside* a sharded module is a `DTensor`: its
`.shape` reports the **whole** while its data is this rank's slice, so
`float(v.sum())` is quietly wrong and comparing widths across degrees shows
nothing. Test with `isinstance(v, DTensor)` and reassemble with
`gather(model, v, dim=...)` or `v.full_tensor()`.

## Parameters are DTensors

`layer.weight` is a `DTensor`: this rank holds `1/tp_size` of it, but **`.shape`
reports the whole**, so shape alone will not tell you it is split. Llama-3.2-3B
at tp=2:

<!-- test: skip -->
```python
w = model.model.layers[0].self_attn.q_proj.weight
w.shape                 # (3072, 3072) — the global shape
w.placements            # (Shard(dim=0),)
w.to_local().shape      # (1536, 3072) — what this rank holds
w.full_tensor()         # the whole, allocated on every rank (a collective)
```

**A reduction over a sharded weight returns this rank's answer, with no error.**
`w.mean()`, `w.norm()`, `w.abs().max()` return a `Partial` `DTensor`, and `float()`
reads this rank's uncombined number:

<!-- test: skip -->
```python
float(w.mean())                  # rank 0:  2.256e-05   rank 1: -1.171e-05
float(w.mean().full_tensor())    # 5.421e-06 on both — the real mean
float(w.norm())                  # rank 0: 81.80        rank 1: 68.97
float(w.norm().full_tensor())    # 106.996
```

Call `.full_tensor()` on the **reduction**, not the weight — one small collective
rather than a copy of the layer — or a weight-norm sweep plots per-rank noise.

**A sharded weight cannot be edited in place through the `DTensor`.**
`weight[:, :10] = 0` raises `NotImplementedError: Operator aten.fill_.Tensor does
not have a sharding strategy registered`. Write through `.to_local()` under
`torch.no_grad()` — each rank edits the rows or columns it holds. A rank-one
(ROME-style) update has to be expressed per-rank the same way; the
`model-editing-and-lora` skill has the single-GPU form.

## The three rules for intervention code

Every rank runs your block — that is what keeps the collectives lined up.

**1. No rank-dependent control flow.** Nothing may branch on rank or take a
different path on different ranks; the ranks stop agreeing on when to gather and
the run deadlocks — quietly, with no exception and no watchdog for minutes. A
plain `kill` (SIGTERM) of `torchrun` takes the ranks down with it; after
`kill -9` they survive holding the cards, so find the rank pids and kill those.

**2. Seed before you sample.** `torch.initial_seed()` differs per rank under
`torchrun`, so unseeded sampling makes the ranks generate different tokens, and
the model's all-reduces then sum activations from different sequences: the output
is *wrong* on every rank, not merely different. Many checkpoints ship
`do_sample: true` in `generation_config.json`, so it bites without asking. Pass
`do_sample=False`, or call `torch.manual_seed(0)` on every rank before generating.

**3. Clone before editing any value a sharded module produced** (see "Reading
and editing").

Every rank computes the **same saved values**, so print or write results from one
rank (`if int(os.environ.get("RANK", 0)) == 0: print(...)`) or you get N copies.
That `if` goes around the *print*, outside the trace — the one place a rank test is
safe. Never inside a block.

## Requires transformers >= 5.19

nnsight reads the exact layout of 5.19's distributed backend, and a layout it
cannot read would hand your code one rank's slice as though it were the whole
tensor. So 5.16–5.18 are refused with `UnsupportedTransformersVersion` the moment
a sharded model is seen; releases before the DTensor backend (≤ 5.15) stop earlier,
at `ModuleNotFoundError: No module named 'transformers.distributed.tensor_parallel'`.

## Expert parallelism

A mixture-of-experts checkpoint can be split by *expert* instead of along the
feature dimension:

<!-- test: skip -->
```python
model = TransformersModel(
    "Qwen/Qwen3-30B-A3B",
    task="text-generation",
    distributed_config=DistributedConfig(tp_size=2, ep_size=2),
    dispatch=True,
)
```

`mlp.experts.output` and `mlp.gate.output` come back at their single-GPU shapes
and values, and edits to them (cloned, rule 3) carry through. The degree has to
divide the **expert count**; pre-flight with
`max_tp_size(config, expert_parallel=True)`, which reads `base_model_ep_plan`.
Plain tensor parallelism on an MoE checkpoint is also fine: the experts shard
along their feature dimensions and behave like any MLP.

## Not supported

Four styles are refused: `megamoe_router`, `megamoe_experts`,
`moe_identity_expert`, and MLA's split kv projection (`mla_kv_a_proj`). A plan
containing one supports no degree, so the load raises `UnshardableCheckpoint`.
DeepSeek-V2-Lite is the one to know: `mla_kv_a_proj` rules out tensor
parallelism, while expert parallelism is still open to it.

## Diagnosing

Three of these arrive wrapped in a pipeline `ValueError: Could not load model
...`, with the real message near the end of it.

| Symptom | Cause |
|---|---|
| `KeyError: 'RANK'` | Started with `python`, not `torchrun`. TP needs the calling process to be a rank, which is also why it cannot run in a notebook kernel |
| `tp_size (2) * fsdp_size (1) is not equal to world_size (4)` | `--nproc_per_node` must equal `tp_size` |
| `` `tp_plan` and `device_map` are mutually exclusive `` | Pick one — they are two different ways of spreading a model |
| `UnshardableCheckpoint` | No `base_model_tp_plan`, a refused style, or a degree that divides nothing. The message lists the degrees that work. Load on one GPU or with `device_map` |
| `UnsupportedTransformersVersion` | transformers 5.16–5.18 — upgrade to 5.19, or load on one GPU |
| `ModuleNotFoundError: transformers.distributed.tensor_parallel` | transformers ≤ 5.15 — upgrade, or load on one GPU |
| Nothing raised, but `fragments.enabled` is `False` and every rank holds the whole model | A custom `tp_plan` dict whose patterns match nothing — a dict plan is not degree-checked |
| Activation width is `hidden_size/N` or `intermediate_size/N` | A value between two sharded modules. `gather(model, v, dim=...)`, naming the axis |
| Shape looks right but the numbers are wrong, and differ per rank | A `DTensor` reporting the global shape: a `.source` value inside a sharded module, or a reduction over a weight. Check `v.placements`; fix with `.full_tensor()` |
| Bare `AssertionError`, empty message | `embed_tokens.output` on a plan that shards the embedding, or a `tracer.cache()` with no arguments that reaches it. Read `layers[0].input` and name the modules to cache |
| `RuntimeError: Output 0 of ... is a view and is being modified inplace` | Editing a value a sharded module produced in place — clone, edit, assign back |
| `NotImplementedError: Operator aten.fill_.Tensor does not have a sharding strategy` | Editing a sharded weight in place — write through `.to_local()` |
| Run hangs with no error | The ranks diverged — rank-dependent control flow, or an exception on one rank only. `kill` torchrun; after a `kill -9`, kill the rank pids too |
| Generated text differs per rank | Sampling without an identical seed (rule 2) |
| Results differ from single-GPU by ~1e-3 in bfloat16 | Expected: an all-reduce sums in a different order than one matmul. In float32 the same gap is ~1e-6; anything larger there is a real difference. Greedy token choices are identical either way |

## Choosing between this and vLLM

Both shard across GPUs. Use **tensor parallelism** when you want ordinary
`TransformersModel` semantics — gradients, `.source`, `tracer.result`, list
inputs — on a model that does not fit. Use **vLLM** (the `vllm` skill, with
`tensor_parallel_size`) when you need throughput, continuous batching, serving or
async streaming, and can accept its differences (no gradients, one prompt per
invoke, `model.logits` / `model.samples`).
