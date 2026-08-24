---
name: vllm
description: Run nnsight interventions on top of the vLLM inference engine — high-throughput continuous batching, tensor parallelism across GPUs, and async token streaming, with arbitrary Python interventions inline with the forward pass. Use when an experiment needs throughput or many concurrent prompts, when a model must be sharded across GPUs, or when streaming generated tokens with interventions. Covers the ways the vLLM path differs from TransformersModel: model.logits/model.samples for per-step values, one prompt per invoke, sampling kwargs on the trace, per-invoke saved containers, model.edit() to install a block on the engine so it runs for every request, and what is not supported.
---

# vLLM

`VLLM` runs your interventions inside vLLM's worker process. You get PagedAttention,
continuous batching, and tensor parallelism; the intervention code is serialized
onto each request, executed against the real modules, and the saved values shipped
back.

> **The examples in this skill are not executed by the repo's test suite** — vLLM
> is not installed in the CI environment and requires a GPU. They are drawn from
> the nnsight 0.8 sources and its vLLM test suite. Verify against your own
> deployment before relying on exact output.

## Loading

<!-- test: skip -->
```python
from nnsight.modeling.vllm import VLLM

model = VLLM("gpt2", gpu_memory_utilization=0.1, dispatch=True)
```

`VLLM(repo_id, *, mode="sync"|"async", dispatch=False, tensor_parallel_size=1,
gpu_memory_utilization=0.9, **vllm_kwargs)`. With `dispatch=False` only a
meta-tensor tree is built — no GPU memory until the first trace. `mode` is fixed at
construction; you cannot switch per trace. The engine runs eagerly (every location
served) unless you declare `taps=[...]`, which keeps CUDA-graph replay and serves the
declared locations from it — see *Graph taps* below.

## The five differences from TransformersModel

| | `TransformersModel` | `VLLM` |
|---|---|---|
| read the output | `tracer.result` | `tracer.result` is the finished `RequestOutput`; **`model.logits` / `model.samples`** give per-step values |
| several prompts | a list, or several invokes | **one prompt per invoke**; a list is rejected. One invoke takes a string, token ids, a tokenizer's output, or a vLLM `TokensPrompt`/`TextPrompt` |
| sampling settings | on `generate(...)` | on `trace(...)` / `invoke(...)`, becoming `SamplingParams` |
| collecting per-prompt values | one shared saved container works | one name saved in each invoke comes back as a **list**, one entry per invoke (and per sampled sequence when `n>1`) |
| `generate` vs `trace` | different methods | `generate` traces in a `with` block, and just runs (returning `RequestOutput`s) without one; driven by `max_tokens` |

## Canonical pattern

<!-- test: skip -->
```python
with model.trace("The Eiffel Tower is located in the city of", temperature=0.0, top_p=1):
    model.transformer.h[8].output[:] = 0          # intervene as usual
    logits = model.logits.save()

print(model.tokenizer.decode(logits.argmax(dim=-1)))
```

`model.logits` is the pre-sampling logit tensor for the current step; `model.samples`
is the token the sampler drew. Both are readable **and assignable** — assigning
`model.samples` forces the token the engine continues from, which is how you steer
generation at the sampling level.

## Multi-token generation

<!-- test: skip -->
```python
with model.trace("Madison Square Garden is located in the city of",
                 temperature=0.0, top_p=1.0, max_tokens=3) as tracer:
    steps = nnsight.save([])
    for _ in tracer.iter[0:3]:
        steps.append(model.logits)

print(model.tokenizer.batch_decode([step.argmax(dim=-1) for step in steps]))
# [' New', ' York', ' City']
```

`tracer.all()` covers every generated step. Bounded `tracer.iter[:N]` is still
preferable when you have code after the loop — see the `nnsight` skill →
generation.

## Continuous batching

Each invoke is one vLLM request; the engine batches them. Your block runs once per
request, so **a name saved in every invoke comes back as a list** — one entry per
invoke, in order — and a name saved once stays that value. A container bound *and
saved above* the invokes is one object and merges element-wise instead. Under
`n>1` the same rule applies per sampled sequence, and where you hold outputs
rather than variables (async, serve, `generate`) each sequence's values ride
`output.outputs[i].saves`.

Distinct names per invoke work as well, and read as they always did:

<!-- test: skip -->
```python
with model.trace(max_tokens=3) as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        paris = nnsight.save([])
        for _ in tracer.all():
            paris.append(model.samples.item())

    with tracer.invoke("The capital of Japan is"):
        tokyo = nnsight.save([])
        for _ in tracer.all():
            tokyo.append(model.samples.item())

print(model.tokenizer.decode(paris), model.tokenizer.decode(tokyo))
```

For a **dynamic** number of prompts, save one name in every invoke and read the
list:

<!-- test: skip -->
```python
with model.trace(temperature=0.0, max_tokens=1) as tracer:
    for prompt in prompts:
        with tracer.invoke(prompt):
            hidden = model.transformer.h[5].output.save()

len(hidden) == len(prompts)      # one entry per invoke, in order
```

Firing each prompt as its own async trace with `asyncio.gather` also works — the
engine still batches the concurrent requests and each one's saves arrive with its
own output.

## Sampling parameters

Trace-level kwargs fill in whatever an invoke did not name; anything an invoke
names is that invoke's, **including a value that happens to be vLLM's own default**
(`temperature=1.0`, `max_tokens=16`, `n=1`):

<!-- test: skip -->
```python
with model.trace(max_tokens=3) as tracer:
    with tracer.invoke("Hello", temperature=0.0, top_p=1.0):
        greedy = model.samples.save()
    with tracer.invoke("Hello", temperature=1.5, top_p=0.95):
        sampled = model.samples.save()
```

Anything `vllm.SamplingParams` accepts works: `temperature`, `top_p`, `top_k`,
`min_p`, `max_tokens`, `stop`, `stop_token_ids`, `seed`, `repetition_penalty`,
`presence_penalty`, `frequency_penalty`, `logprobs`, `lora_request`.

Per-invoke input must be a **single** prompt: a string, a list of token ids, a
tokenizer's output, or one of vLLM's own prompt dicts (`TokensPrompt`,
`TextPrompt`).

## Graph taps

vLLM's decode throughput comes from replaying CUDA graphs, and a replayed graph runs
no Python — so the eager engine gives up that speed to serve every location. `taps=`
names the locations to record *into* the graph and serve on every replay; everything
else is vanilla vLLM with graphs on.

<!-- test: skip -->
```python
model = VLLM("meta-llama/Llama-3.1-8B", dispatch=True,
             taps=["model.layers.*.output", "model.layers.16.self_attn.o_proj.input"])

with model.trace("Hello", temperature=0.0, max_tokens=16) as tracer:
    hs = nnsight.save([])
    for _ in tracer.iter[:16]:
        model.model.layers[16].output[0][:] += 4 * vector          # in place
        hs.append(sum(model.model.layers[20].output)[-1].clone())  # clone: graph memory
```

- `*` matches one path segment; the `model.` prefix is implied; a tap naming no
  module is refused at construction.
- **Only taps are served.** A read of any other module location errors when the
  request ends, naming the location; `model.logits`, `model.samples` and
  `tracer.result` always work.
- **Edit in place.** A replacement (`layer.output = t`) is copied back into the
  graph's memory and must keep its shape.
- **Clone what you keep.** The value served *is* the graph's memory, rewritten
  next step.
- Passing `enforce_eager` yourself is refused if it contradicts `taps`.

Measured on Llama-3.1-8B (A100, one GPU): plain generate 92 tok/s on vanilla vLLM,
89 with taps, 86 eager; capturing one layer every step 89 with taps, 79 eager. Under
tensor parallelism the gap widens — at `tp=4` taps hold 213 tok/s against the eager
engine's 67 — so declare taps whenever the GPUs outnumber one.

## Tensor parallelism is transparent

<!-- test: skip -->
```python
model = VLLM("meta-llama/Llama-3.1-8B", tensor_parallel_size=4, dispatch=True)

with model.trace("Hello", temperature=0.0):
    hidden = model.model.layers[16].output.save()      # full, unsharded tensor
```

nnsight gathers a column-parallel output or a row-parallel input into the whole
tensor before your code reads it (all-reduces a deferred partial sum), and re-splits
what you write, so every rank runs identical intervention code against the complete
tensor. Two caveats: a *fused* projection (`qkv_proj`, `gate_up_proj`) gathers in
rank order — `[q₀ k₀ v₀ | q₁ k₁ v₁]` — so slice it by head rather than by `[:q_size]`;
and every rank runs your block, so no rank-dependent control flow, and sample greedily
or seeded. Decode-context parallelism on MLA models (`decode_context_parallel_size`)
is handled too.

## Async streaming

<!-- test: skip -->
```python
import asyncio
from nnsight.modeling.vllm import VLLM

model = VLLM("gpt2", mode="async", gpu_memory_utilization=0.1, dispatch=True)

async def main():
    with model.trace("The Eiffel Tower is in", max_tokens=10, temperature=0.0) as tracer:
        for _ in tracer.all():
            model.transformer.h[8].output[:] = 0

    async for output in tracer.backend:          # an attribute, not a call
        print(output.outputs[0].text)

asyncio.run(main())
```

`tracer.backend` yields `RequestOutput`s as the engine produces them. Use
`asyncio.gather` over several traces for concurrent requests.

## Not supported on the vLLM path

- **Gradients / `tensor.backward()`** — no backward through the engine
- **`model.scan()`** — shape inference is not available
- **`.source` on fused kernels** — vLLM's fused CUDA ops have no Python source
- **Diffusion and multimodal** — the integration is text-only
- **`tracer.barrier(n)`** — each invoke is its own request, scheduled independently, so the blocks never meet; it raises rather than hanging
- **Pipeline parallelism and speculative decoding** — shard with `tensor_parallel_size` instead; do not pass `pipeline_parallel_size > 1` or `speculative_config`

If an experiment needs gradients or source tracing, run it on `TransformersModel`
and move only the throughput-bound part to vLLM.

## Editing the engine

A trace rides one request, so a sweep re-sends the same block per prompt and only
requests that *are* traces get touched. `model.edit()` sends it once; every
request the engine runs afterwards gets its own copy — including ones submitted by
something that never heard of nnsight.

<!-- test: skip -->
```python
model = VLLM("meta-llama/Llama-3.1-8B", dispatch=True, enable_prefix_caching=False)

with model.edit() as (tracer, edit):
    hidden = model.model.layers[16].output[0].save()

outputs = model.generate(prompts, max_tokens=5)   # plain requests, not traces
outputs[3].saves["hidden"]                        # on the output that produced it

edit.clear()
```

The block is written like a trace body, but belongs to no request — there is **no
`tracer.invoke(...)`**. The tracer is bound so `tracer.all()` can follow a request
across its generated tokens; without it the block sees only the prefill. Values
arrive on `output.saves` and are dropped as they go, so nothing accumulates; for a
traced request, reach them through `tracer.result.saves`, which carries the edit's
values only — your trace's own come back as your variables. (An output from
`model.generate(...)` is a different object: it carries both, the trace's winning a
name collision, with the trace's own also on `output.nnsight_saves`.)

`model.clear_edits()` drops every edit still installed (`await model.aclear_edits()` on an async engine). `model.edit(serve=url)`
installs one on an nnsight-serve engine from a GPU-less client.

**`enable_prefix_caching=False` is required.** A cached token is served without a
forward pass, so nothing runs for it and the block sees fewer rows than the prompt
has, silently. A trace forces its own recompute; an edit rides requests it did not create
and cannot.

On `mode="async"`, installing the block has to be awaited from inside the loop —
use `async with model.edit()` and `await edit.aclear()`. A plain `with`
there raises rather than silently not installing it.

Editing is worth it for sweeps (one serialization instead of one per prompt:
1024 prompts capturing one layer went 2.04 s traced → 1.43 s edited, against
0.87 s for bare vLLM) and for instrumenting traffic you do not control. Keep
tracing for one-off experiments and when you want values pushed back into your own
variables.

## vLLM versions

Works on 0.16–0.27. nnsight asks vLLM for the model runner it instruments
(`VLLM_USE_V2_MODEL_RUNNER=0`) when building the engine, since 0.27 defaults to a
second runner for non-MoE models; setting that to `1` yourself is refused, and the
worker refuses any other runner rather than coming up uninstrumented. Graph taps need
a vLLM with breakable CUDA graphs (`vllm.compilation.breakable_cudagraph`). MoE expert
outputs need no gathering on 0.27 — that release moved the all-reduce inside the
layer, so the value is already whole (verified: both ranks return the identical
tensor); through 0.26 nnsight all-reduces the partial itself.

Two engine settings nnsight chooses for you: **chunked prefill is off** unless you
pass `enable_chunked_prefill=True` — a block must see its prompt whole, so a prompt
that does not fit a step's token budget waits a step instead of being split (and
if you turn chunking on, a request whose prompt got chunked comes back with an
error rather than a slice); and a **traced request skips the prefix cache** so its
prompt is recomputed. Tracing needs no `VLLM_ALLOW_INSECURE_SERIALIZATION`.

## Choosing between vLLM and TransformersModel

| Situation | Use |
|---|---|
| one prompt, or a small sweep | `TransformersModel` — simpler, full feature set |
| gradients, attribution, `.source`, `scan` | `TransformersModel` |
| hundreds of concurrent prompts | `VLLM` |
| model larger than one GPU, locally | `VLLM` with `tensor_parallel_size` |
| model larger than your whole machine | NDIF — see the `nnsight-remote` skill |
| token-by-token streaming to a client | `VLLM` with `mode="async"` |
| instrumenting requests you did not write | `VLLM` with `model.edit()` |

## Related skills

- `nnsight` — the intervention API that carries over unchanged
- `nnsight-remote` — the other way to run models you cannot host
- `nnsight-debugging` — errors; note that vLLM worker exceptions are re-raised at the client as `RuntimeError` carrying the original traceback
