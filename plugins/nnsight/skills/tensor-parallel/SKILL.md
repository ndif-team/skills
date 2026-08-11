---
name: tensor-parallel
description: Trace a model too big for one GPU by sharding it across several with transformers tensor parallelism — TransformersModel(..., distributed_config=DistributedConfig(tp_size=N)) launched under torchrun. Sharded activations are gathered so interventions read and edit whole tensors exactly as on a single GPU. Use when a checkpoint does not fit on one card, when a user asks about multi-GPU tracing, tp_size, tp_plan, device_map for large models, or hits activation shapes that are a fraction of the expected width (hidden_size/N, intermediate_size/N). Covers the two rules SPMD imposes on intervention code — no rank-dependent control flow, and seed before sampling — and what is not supported (MoE / expert-parallel).
---

# Tensor Parallelism

Transformers tensor parallelism splits every attention and MLP projection *within*
each layer across GPUs, so a model that does not fit on one card runs on several
at once. (Contrast `device_map="auto"`, which puts whole *layers* on different
GPUs and runs them in sequence — that needs nothing special from nnsight.)

nnsight gathers sharded activations before your intervention sees them and
re-splits what you leave behind, so **the trace you write is the trace you would
write against one GPU**. Nothing to install, import, or enable.

> **Not executed by the repo's test suite in CI** — these examples need ≥2 GPUs.
> They are drawn from `nnsight/modeling/tp/` and
> `tests/test_transformers_tensor_parallel.py`, which run on multi-GPU machines.

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

**`tp_plan="auto"` and `tp_size=` are not `from_pretrained` arguments** in
transformers 5.x — passing them raises `TypeError: ... unexpected keyword argument
'tp_plan'`. The `from_pretrained` docstring still documents them; it is stale. Use
`distributed_config=DistributedConfig(tp_size=N)`.

`tp_size` must divide the model's attention heads, key/value heads, and
intermediate size. Llama-3.2-3B (24 heads, 8 kv heads, 8192 intermediate) shards
cleanly at 2, 4, or 8; Qwen2.5-0.5B has 2 kv heads and so stops at 2.

## Reading and editing

Identical to single-GPU code. A column-parallel output arrives at full width even
though each rank computed a slice of it:

<!-- test: skip -->
```python
with model.trace("The Eiffel Tower is in the city of"):
    gate = model.model.layers[5].mlp.gate_proj.output.save()   # (1, 11, 8192)
    logits = model.lm_head.output.save()
```

Edits that span rank boundaries need no special handling:

<!-- test: skip -->
```python
with model.trace(prompt):
    model.model.layers[5].mlp.gate_proj.output[..., :3000] = 0
    logits = model.lm_head.output.save()
```

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
| `lm_head.output`, `embed_tokens.output` | no |

The gather only fires when an intervention is parked on that location, so reading
a few locations does not pay for the hundreds you ignore. `tracer.cache()` gathers
only the modules it selects.

## The two rules for intervention code

Every rank runs your block — that is what keeps the collectives lined up.

**1. No rank-dependent control flow.** Nothing may branch on rank or take a
different path on different ranks; the ranks stop agreeing on when to gather and
the run deadlocks.

**2. Seed before you sample.** This is a *correctness* bug, not an inconsistency.
If sampling diverges, the ranks generate different tokens, and the model's own
all-reduces then sum activations computed from different sequences — the output is
wrong on every rank, not merely different. Many checkpoints ship
`do_sample: true` in `generation_config.json`, so it bites without asking:

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

Every rank computes the **same saved values** (they come from gathered tensors),
so print or write results from one rank or you get N copies:

<!-- test: skip -->
```python
import os
if int(os.environ.get("RANK", 0)) == 0:
    print(logits.argmax(-1))
```

## Not supported

Mixture-of-experts sharding (`grouped_gemm`, `ep_router`, `moe_tp_experts`,
`megamoe_*`, `moe_identity_expert`) and MLA's split kv projection (`mla_kv_a_proj`)
slice by *expert* rather than along the feature dimension. Loading such a model
tensor-parallel raises `UnsupportedParallelStyle` naming the module and style,
rather than handing back a fragment of a tensor.

## Diagnosing

| Symptom | Cause |
|---|---|
| Activation width is `hidden_size/N` or `intermediate_size/N` | The gather did not run — check the model really loaded with `distributed_config`, and that `model.interleaver.enabled` is `True` |
| `TypeError: ... unexpected keyword argument 'tp_plan'` | transformers 5.x — use `distributed_config=DistributedConfig(tp_size=N)` |
| Run hangs with no error | The ranks diverged — rank-dependent control flow, or an exception on one rank only |
| Generated text differs per rank | Sampling without an identical seed (see rule 2) |
| `UnsupportedParallelStyle` | MoE / expert-parallel model; use vLLM (see the `vllm` skill) or a single GPU |
| Results differ from single-GPU by ~1e-3 | Expected: an all-reduce sums in a different order than one matmul. Token choices are unaffected |

## Choosing between this and vLLM

Both shard across GPUs. Use **tensor parallelism** when you want ordinary
`TransformersModel` semantics — `tracer.result`, list inputs, the full tracing API
— on a model that does not fit. Use **vLLM** (see the `vllm` skill) when you need
throughput, continuous batching, or async streaming, and can accept its
differences (one prompt per invoke, `model.logits` / `model.samples`).
