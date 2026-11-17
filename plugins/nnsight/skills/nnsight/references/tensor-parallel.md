# Tensor parallelism

Transformers tensor parallelism splits every attention and MLP projection
*within* each layer across processes (ranks), one GPU each. nnsight gathers a
sharded module's input and output before the trace body sees them and re-splits
what the body leaves behind, so most of a trace reads as it would on one GPU.
This guide is the rest. `device_map="auto"`, which puts whole layers on
different GPUs, needs none of it. For throughput or serving, where gradients and
`.source` are not needed, shard inside vLLM instead: [vllm](vllm.md).

Requires transformers 5.19 or later. An older install raises
`UnsupportedTransformersVersion` when the sharded model is first seen.

## Load under torchrun

```python
# tp_trace.py        torchrun --nproc_per_node=3 tp_trace.py
import os
import torch
import nnsight
from transformers.distributed import DistributedConfig
from nnsight import TransformersModel

model = TransformersModel(
    "HuggingFaceTB/SmolLM2-135M-Instruct",
    task="text-generation",
    dispatch=True,
    dtype=torch.float32,
    distributed_config=DistributedConfig(tp_size=3),       # equals --nproc_per_node
)
prompt = "The Eiffel Tower is in the city of"

with model.trace(prompt):
    gate = model.model.layers[5].mlp.gate_proj.output.save()   # each rank computed 512 of these 1536
    logits = model.lm_head.output.save()

assert tuple(gate.shape) == (1, 9, 1536)
if int(os.environ["RANK"]) == 0:                   # every rank holds the same saved values
    print(model.tokenizer.decode(logits[0, -1].argmax()))
```

```
 Paris
```

**Every rank runs the whole script, the trace bodies included.** SmolLM2 stands
in for a model that does not fit on one card. With `CUDA_VISIBLE_DEVICES=` set
empty the ranks run on CPU, which is how to test a block on a machine with fewer
GPUs than ranks. `tp_plan="auto"` in place of `distributed_config=` shards over
however many ranks were launched; `DistributedConfig` states the degree in the
code and is the only form that can ask for expert parallelism.

## Check the degree before allocating

`tp_size` has to divide the attention heads, the key/value heads and the MLP
width. `max_tp_size` reads the config alone (no weights, GPU or `torchrun`) and
returns the largest degree; every degree that works divides it:

```python
from transformers import AutoConfig
from nnsight.modeling.tp import max_tp_size

assert max_tp_size(AutoConfig.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")) == 3
assert max_tp_size(AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")) == 2
assert max_tp_size(AutoConfig.from_pretrained("openai-community/gpt2")) is None
```

`None` means the checkpoint cannot be split: its config has no
`base_model_tp_plan` (gpt2), or the plan uses a [refused style](#not-supported).
Asking anyway raises `UnshardableCheckpoint` before any weights are read, and the
message lists the degrees that would work.

## What the trace body sees

| Value | Arrives | To edit it |
|---|---|---|
| the output of a sharded module (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`, `lm_head`, `embed_tokens`), the input of `o_proj` and `down_proj`, `mlp.output`, the tensor in `self_attn.output`, and `layers[0].input`, which is the embedding's output | whole | clone, edit, assign back (rule 3) |
| `layers[i].output`, `mlp.input`, `norm.output` | whole | in place, as on one GPU |
| the input of `q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`, `lm_head` | whole, as a `DTensor` | edit the same value where it is a plain tensor: `mlp.input`, a norm's `.output` |
| a value *between* two sharded modules: `mlp.act_fn.output`, `mlp.source.self_gate_proj_0.output`, attention's `query_states_0` | this rank's slice, `1/tp_size` wide on one axis | [gather, edit, shard](#a-value-between-two-sharded-modules) |
| every parameter, and a `.source` value *inside* a sharded module | a [`DTensor`](#parameters-and-other-dtensors): `.shape` is the whole, the data is this rank's | a weight: through `.to_local()` |

Calling a sharded module on a value (`model.lm_head(model.model.norm(hidden))`)
returns the full-width result.

## Three rules for the trace body

Every rank runs your block, which is what keeps their collective operations
lined up.

**1. No rank-dependent control flow.** Nothing inside a trace may branch on the
rank or take a different path on different ranks. The ranks then stop agreeing
on when to gather and the run hangs, with no exception and no output. Stop it
with a plain `kill` of `torchrun`, which takes the ranks down with it; after
`kill -9` the ranks survive holding their GPUs and have to be killed by pid. The
one safe rank test is around a `print` or a file write *after* the trace, as
above.

**2. Seed before sampling.** `torch.initial_seed()` differs per rank, so
unseeded sampling makes each rank generate different tokens while their
all-reduces sum activations across those different sequences. Pass
`do_sample=False`, or call `torch.manual_seed(0)` in the script before
generating.

**3. Clone before editing what a sharded module produced.** A value in the
first row of the table is the output of a collective operation, and torch
refuses an in-place write into one. The same write is accepted on one GPU, so
code that passes there fails here. Clone, edit, assign back:

```python
gate_proj = model.model.layers[5].mlp.gate_proj

with model.trace(prompt):
    edited = gate_proj.output.clone()
    edited[..., :1000] = 0                         # spans rank boundaries; no matter
    gate_proj.output = edited
    ablated = model.lm_head.output.save()

assert not torch.allclose(ablated, logits)
```

Assigning a whole new value (`gate_proj.output = torch.zeros_like(...)`) needs
no clone.

## A value between two sharded modules

Nothing on such a value records which axis holds the shard, and attention's
`view` and `transpose` move it to the head axis. Name the axis with `gather`, and
hand back this rank's piece with `shard`:

```python
from nnsight.modeling.tp import gather, shard

attn = model.model.layers[1].self_attn

with model.trace(prompt):
    q = attn.source.query_states_0.output          # (1, 3, 9, 64): 3 of the 9 heads
    heads = gather(model, q, dim=1).clone()        # (1, 9, 9, 64)
    shapes = nnsight.save((tuple(q.shape), tuple(heads.shape)))
    heads[:, 3] = 0                                # ablate head 3, whichever rank holds it
    attn.source.query_states_0.output = shard(model, heads, dim=1)
    head_ablated = model.lm_head.output.save()

assert shapes == ((1, 3, 9, 64), (1, 9, 9, 64))
assert not torch.allclose(head_ablated, logits)
```

Both are collective operations, so call them unconditionally (rule 1). To find
the axis, print the value's shape with and without sharding: the axis that
shrank by `tp_size` is the one.

**A one-process run validates none of this.** On an unsharded model `gather`
and `shard` return their argument untouched, so a wrong `dim` passes, and so
does an in-place edit that rule 3 forbids. Test the block at two or more ranks.

## Parameters and other DTensors

`layer.weight` is a `DTensor`. nnsight does not reassemble it, and its `.shape`
reports the whole, so the shape will not tell you it is split:

```python
from torch.distributed.tensor import DTensor

w = model.model.layers[0].self_attn.q_proj.weight
assert isinstance(w, DTensor) and tuple(w.shape) == (576, 576)
assert tuple(w.to_local().shape) == (192, 576)            # what this rank holds

local, whole = float(w.norm()), float(w.norm().full_tensor())
assert local < whole                                      # float() read one rank's share
```

**A reduction over a `DTensor` returns this rank's number, with no error.**
`w.norm()`, `w.mean()` and `w.abs().max()` differ per rank and match no real
value. Call `.full_tensor()` on the *reduction*, which is one small collective
in place of a copy of the layer.

A `.source` value inside a sharded module (`gate_proj.source.F_linear_0`) is a
`DTensor` in the same way, and under `torch.no_grad()` it is a plain tensor of
this rank's width. `gather(model, value, dim=-1)` returns the whole in both
cases.

**A sharded weight cannot be edited through the `DTensor`.** Write through
`.to_local()`, each rank editing the rows it holds:

```python
gate_weight = model.model.layers[0].mlp.gate_proj.weight
with torch.no_grad():
    gate_weight.to_local()[:, :10] = 0

assert float(gate_weight.full_tensor()[:, :10].abs().sum()) == 0
```

A rank-one update (the single-GPU form is in
[model-editing-and-lora](model-editing-and-lora.md)) has to be written per rank
the same way.

## Expert parallelism

A mixture-of-experts checkpoint can be split by *expert* instead. This needs a
checkpoint whose config has a `base_model_ep_plan` and two GPUs, and is shown
for its shape:

```python
model = TransformersModel(
    "Qwen/Qwen3-30B-A3B",
    task="text-generation",
    dispatch=True,
    distributed_config=DistributedConfig(tp_size=2, ep_size=2),
)
```

`mlp.experts.output` and `mlp.gate.output` come back at their single-GPU shapes
and values, and a cloned edit of `experts.output` carries through. The degree has
to divide the expert count: `max_tp_size(config, expert_parallel=True)`. Plain
tensor parallelism on the same checkpoint also works, the experts being sharded
along their feature dimensions like any MLP.

## Not supported

Plans that use `megamoe_router`, `megamoe_experts`, `moe_identity_expert` or
MLA's split kv projection (`mla_kv_a_proj`) are refused: `max_tp_size` returns
`None` and the load raises `UnshardableCheckpoint`. DeepSeek-V2-Lite is the
common case: no tensor parallelism, while expert parallelism is open to it.

A PEFT adapter is refused on a tensor-parallel load, by `peft=` and by
`model.load_adapter(...)` alike, with a `ValueError`.

## Diagnosing

The first three arrive inside a pipeline `ValueError: Could not load model ...`,
with the real message at the end of it.

| Symptom | Cause |
|---|---|
| `KeyError: 'RANK'`, then `OSError: We tried to initialize torch.distributed for you, but it failed` | Started with `python`, not `torchrun`. The calling process has to be a rank, so this cannot run in a notebook kernel |
| `The parallel mesh requires 2 processes, but world_size is 4` | `--nproc_per_node` has to equal `tp_size` |
| ``Tensor parallelism and `device_map` are mutually exclusive`` | Drop `device_map` |
| `UnshardableCheckpoint` | No plan, a refused style, or a degree that does not divide. Load on one GPU or with `device_map` |
| `RuntimeError: Output 0 of ... is a view and is being modified inplace` | Rule 3 |
| `NotImplementedError: Operator aten.fill_.Tensor does not have a sharding strategy registered` | An in-place write to a `DTensor`. A weight: go through `.to_local()`. The input of a column-parallel module: third row of the table |
| `... got mixed torch.Tensor and DTensor` | A plain tensor assigned to the input of a column-parallel module (third row of the table), or a sharded model wrapped with plain `NNsight(module)`: load it with `TransformersModel`, or pass `interleaver=Interleaver(fragments=TPFragments()), envoys=tp_envoys()` (from `nnsight.intervention.interleaver`, `nnsight.modeling.tp`, `nnsight.modeling.tp.envoys`) |
| An activation is `1/tp_size` of the expected width | A value between two sharded modules: `gather` |
| Shape is right, numbers are wrong and differ per rank | A `DTensor` reduced or converted without `.full_tensor()` |
| The run hangs with no error, or one rank dies with `Connection closed by peer` | Rule 1: the ranks took different paths |
| Generated text differs per rank | Rule 2 |
| Values differ slightly from a single-device run | An all-reduce sums in a different order. On SmolLM2 at three ranks the logits differ by 1e-6 (relative) in float32 and 3e-2 in bfloat16, with the same greedy tokens: compare in float32 |
