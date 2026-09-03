---
name: vllm
description: Run nnsight interventions inside the vLLM inference engine — continuous batching, tensor parallelism, CUDA-graph taps, async streaming, an engine-wide edit() and nnsight-serve. Use when an experiment needs throughput or many concurrent prompts, a model sharded across GPUs, streamed tokens with interventions, or a served engine that other clients hit. Covers what a block sees on vLLM (flat [tokens, hidden] rows, the (hidden, residual) layer-output pair, live buffers), model.logits/model.samples, one prompt per invoke, passing values between invokes with two traces, sweeps with model.edit(), taps, and what is not supported.
---

# vLLM

`VLLM` runs your block inside vLLM's worker process: the intervention code is
serialized onto each request, run against the real modules, and the saved values
shipped back. You get PagedAttention, continuous batching, tensor parallelism and
CUDA-graph replay (with `taps=`), and the same trace syntax.

Blocks in this skill run under the test suite on a GPU with vLLM 0.27.1 (they are
skipped elsewhere). Model: `HuggingFaceTB/SmolLM2-135M-Instruct`, a Llama-style
trunk, so what it shows about layer outputs holds for Llama, Qwen and Mistral.

## What breaks first: four things a vLLM block sees differently

<!-- test: gpu setup -->
```python
import torch
import nnsight
from nnsight.modeling.vllm import VLLM

model = VLLM("HuggingFaceTB/SmolLM2-135M-Instruct", dispatch=True,
             gpu_memory_utilization=0.2, max_model_len=1024, enable_prefix_caching=False)

with model.trace("The Eiffel Tower is in the city of", temperature=0.0, max_tokens=1):
    out = model.model.layers[12].output
    shapes = nnsight.save((type(out).__name__, tuple(out[0].shape), tuple(out[1].shape)))
    norms = nnsight.save((out[0].norm(dim=-1).mean().item(), out[1].norm(dim=-1).mean().item()))
    resid = (out[0] + out[1]).clone().save()      # the residual stream after layer 12
    alias = out[0].save()                         # NOT cloned: a live buffer
    logits = model.logits.save()
    samples = model.samples.save()

print(shapes, [round(n, 1) for n in norms])
# ('tuple', (9, 576), (9, 576)) [...]   -- no batch axis; 9 prompt tokens
print(tuple(logits.shape), tuple(samples.shape), model.tokenizer.decode(samples.item()))
# (1, 49152) (1, 1) ' Paris'
assert shapes[1] == shapes[2] and shapes[1][0] == 9
assert not torch.equal(alias, resid) and alias.shape == resid.shape
```

1. **No batch axis.** A served value is `[tokens, hidden]` for *your* request's
   rows: the prefill serves every prompt token, each decode step serves one row.
   The last position is `[-1]`, never `[:, -1, :]`.
2. **A decoder layer's `.output` is a pair `(hidden, residual)`.** vLLM fuses the
   residual add into the next layer's norm, so `output[0]` is this layer's
   sub-block output and `output[1]` the residual stream entering it; the residual
   stream *after* the layer is `out[0] + out[1]`. Patching `output[0]` alone
   changes almost nothing. Writing either element steers (the next norm adds them).
3. **Clone what you keep.** A served value is the model's live buffer, and the next
   layer's fused kernel rewrites it after your block returns — the un-cloned
   `alias` above comes back holding later data. Reduce or `.clone()` before saving.
4. **Where tensors live.** A tensor referenced from outside the block (a steering
   vector) travels with the block; move it onto the served value with
   `v.to(h.device, h.dtype)`. Saved tensors come back on the worker's device unless
   you `.cpu()` them inside. `model.device` on an undispatched client is `meta`.

`model.logits` is `[1, vocab]` for the step, `model.samples` is `[1, 1]` for one
sequence. Greedy decoding is `temperature=0.0`; vLLM's default is `1.0`.

## Loading and the constructor

`VLLM(repo_id, *, mode="sync"|"async", dispatch=False, tensor_parallel_size=1,
gpu_memory_utilization=0.9, taps=(), **vllm_kwargs)`. `dispatch=False` builds a
meta tree only. `mode` is fixed at construction. Every other keyword goes to
vLLM's engine args (`max_model_len`, `enable_prefix_caching`, `seed`, ...). The
engine runs eagerly, serving every location, unless you declare `taps=` — see
[graph taps](references/graph-taps.md).

Scripts need an `if __name__ == "__main__":` guard. Dispatching initializes CUDA in
your process, so vLLM starts its EngineCore with `spawn` (it sets
`VLLM_WORKER_MULTIPROC_METHOD` itself, logging `Reasons: CUDA is initialized`), spawn
re-imports the main module, and a `VLLM(...)` at the top level of that module dies
with `An attempt has been made to start a new process before the current process has
finished its bootstrapping phase`. One GPU and `mode="sync"` included; notebooks are
exempt. `enable_prefix_caching=False` whenever you will `model.edit()`.

## Per-step values and generation

Sampling kwargs go on `trace(...)` / `invoke(...)`, not the constructor. Step 0
under `tracer.iter` is the prefill; step *k* serves the *k*-th generated token,
and `model.samples` there is `result.outputs[0].token_ids[k]`. Python locals
persist across steps, so a block can carry state. Order within a step is forward
order: a write to a layer *below* one you already read this step parks until that
layer's next visit — and on the last step there is none, so the block never
finishes and nothing after the loop runs. Put upstream writes (from last step's
state) at the top of the loop body, downstream reads after.

<!-- test: gpu -->
```python
with model.trace("The capital of France is", temperature=0.0, max_tokens=6, ignore_eos=True) as tracer:
    steps = nnsight.save([])
    triggered = False
    for step in tracer.iter[:6]:
        if triggered:                                                  # state from earlier steps
            model.model.layers[8].output[0][-1] += 0.0                  # upstream write FIRST
        score = (model.model.layers[20].output[1][-1] ** 2).sum()      # then downstream reads
        if step == 2:
            model.samples = torch.full_like(model.samples, 3_000)       # force this step's token
        if score.item() > 0 and step >= 3:
            triggered = True
        steps.append((step, triggered, model.samples.item()))
    result = tracer.result.save()          # served at collect time: must be the LAST read

print(result.outputs[0].token_ids, steps[2])
assert result.outputs[0].token_ids[2] == 3_000 and steps[2] == (2, False, 3_000)
assert steps[3][1] is True
```

- `tracer.iter[:N]` is bounded so code after the loop (`tracer.result.save()`)
  runs — but only if all `N` steps happen, and `max_tokens` is a cap, not a
  promise: an EOS, a `stop` string, `stop_token_ids` or a `tracer.stop()` all end
  the request early. A loop the run cannot supply — bounded or open — is cut
  short there with a warning: what it saved comes home, the statements after the
  loop never run, and the warning is emitted in the EngineCore subprocess, so
  `warnings.catch_warnings` in your code never sees it. The result looks complete
  while holding fewer steps than the bound, so check the `len()` of what you
  collected. Hold the run to the count with `ignore_eos=True` or `min_tokens=N`
  (vLLM's spelling; the warning text names it, alongside transformers'
  `min_new_tokens=`), or loop with `tracer.all()` and put the trailing statements
  in a separate `tracer.invoke()` — `tracer.result` cannot be read after the
  `with` block.
- **`tracer.result` must be the last read.** It is the finished `RequestOutput`,
  served after every module, `logits` and `samples` visit; a read after it raises
  `OutOfOrderError` naming that later value.
- Assigning `model.samples` forces the token the engine continues from;
  assigning `model.logits` changes what the sampler sees.

## One prompt per invoke; collecting across invokes

Each `tracer.invoke(prompt)` is one vLLM request; the engine batches them. A name
saved in every invoke comes back as a **list** in invoke order; a name saved once
stays a value. A container saved *above* the invokes is copied per request:
assigning into pre-sized slots merges back, **`append` does not** — you get one
element.

<!-- test: gpu -->
```python
prompts = ["The capital of France is", "The capital of Japan is", "Two plus two is"]

with model.trace(temperature=0.0, max_tokens=1) as tracer:
    slots = nnsight.save([None] * len(prompts))
    appended = nnsight.save([])
    for i, p in enumerate(prompts):
        with tracer.invoke(p):
            out = model.model.layers[20].output
            last = (out[0] + out[1])[-1].clone().save()        # one name, every invoke -> list
            slots[i] = last.norm().item()
            appended.append(i)
            nxt = model.samples.item().save()

print([model.tokenizer.decode(t) for t in nxt], len(last), slots, appended)
assert len(last) == 3 and all(s is not None for s in slots) and len(appended) == 1
```

A list of prompts in one invoke is rejected. Async traces fired with
`asyncio.gather` batch the same way, each carrying its own saves.

### Passing values between invokes: two traces

Invokes are separate requests with separate scopes: a value one invoke reads is
**not** visible to a sibling (`NameError`), `tracer.barrier` raises, and a
`session()` does not bridge traces. Save in one trace, reference in the next —
the saved tensors ship with the second block. Activation patching:

<!-- test: gpu -->
```python
CLEAN, CORRUPT, L, POS = "The Eiffel Tower is in the city of", "The Colosseum is in the city of", 12, slice(1, 4)
paris = model.tokenizer(" Paris")["input_ids"][0]

with model.trace(CLEAN, temperature=0.0, max_tokens=1):
    donor = nnsight.save(tuple(o.clone() for o in model.model.layers[L].output))

with model.trace(temperature=0.0, max_tokens=1) as tracer:
    with tracer.invoke(CORRUPT):
        base = model.logits.softmax(-1)[0, paris].item().save()
    with tracer.invoke(CORRUPT):
        for served, saved in zip(model.model.layers[L].output, donor):     # both elements
            served[POS] = saved[POS].to(served.device)
        patched = model.logits.softmax(-1)[0, paris].item().save()

print(f"P(Paris) corrupt {base:.3f} -> patched {patched:.3f}")
assert patched > base
```

Many invokes still batch: a layer sweep is one trace with one patched invoke per
layer. Index rows as `hs[POS]` (no batch axis) and write *both* elements.

Writing *into* the served value, as above, always fits. A whole-value replacement
(`layer.output = t`) is spliced back into the batch the model is running, so it
has to keep the rows the block owns — and nothing checks that for you. A donor
captured at a different prompt length is the usual way to get a short one; spliced
in as given, it hands the next kernels a slab of the wrong height, and the
mismatch can land as a device-side assert that takes the engine and every request
in it, not just yours. Slice the donor to the rows you are writing
(`served[POS] = donor[POS]`) rather than handing back a shorter tensor.

## Sampling parameters and `n > 1`

Trace-level kwargs fill in what an invoke did not name; anything an invoke names
wins, including vLLM's own defaults (`temperature=1.0`, `max_tokens=16`).
Anything `vllm.SamplingParams` accepts works (`top_p`, `top_k`, `seed`, `stop`,
`logprobs`, `lora_request`, ...). `n=k` fans one prompt into `k` sequences: the
block runs once per sequence, saved names come back as a list per sequence, and a
saved container comes back as a list of `k` containers matching
`result.outputs[i]`.

<!-- test: gpu -->
```python
with model.trace("My favourite colour is", temperature=0.9, seed=0, n=3, max_tokens=4, ignore_eos=True) as tracer:
    caps = nnsight.save([])
    for _ in tracer.iter[:4]:
        caps.append(model.model.layers[8].output[1][-1].clone())
    result = tracer.result.save()

print([o.text for o in result.outputs])
assert len(caps) == 3 and len(caps[1]) == 4 and len(result.outputs) == 3
```

## Sweeps: `model.edit()` and the per-invoke cost

A trace carries its block on the request it rides. `model.edit()` sends the block
once and it runs for every request the engine serves afterwards — including plain
`model.generate(...)` calls and other clients of a served engine — with each
request's values on that request's output. There is no `tracer.invoke` inside an
edit; `tracer.all()` makes it follow every generated step.

<!-- test: gpu -->
```python
import time
sweep = [f"Fact number {i} about the ocean is that it" for i in range(200)]
layer = model.model.layers[20]                     # bind the envoy OUTSIDE the trace

with model.edit() as (tracer, edit):
    out = layer.output
    feat = (out[0] + out[1]).mean(0).cpu().save()

t = time.perf_counter()
outputs = model.generate(sweep, max_tokens=1, temperature=0.0)      # plain requests, block runs
edited = time.perf_counter() - t
feats = torch.stack([o.saves["feat"] for o in outputs])
texts = [o.outputs[0].text for o in outputs]
edit.clear()

t = time.perf_counter()
with model.trace(temperature=0.0, max_tokens=1) as tracer:
    for p in sweep:
        with tracer.invoke(p):
            feat = (layer.output[0] + layer.output[1]).mean(0).cpu().save()
traced = time.perf_counter() - t

t = time.perf_counter()
with model.trace(temperature=0.0, max_tokens=1) as tracer:
    for p in sweep:
        with tracer.invoke(p):
            feat = (layer.output[0] + layer.output[1]).mean(0).cpu().save()
            nxt = model.samples.save()                               # references `model`
traced_with_model = time.perf_counter() - t

print(f"edit {edited:.2f}s  traced {traced:.2f}s  traced+model.samples {traced_with_model:.2f}s")
assert feats.shape == (200, 576) and len(texts) == 200
assert torch.allclose(feats, torch.stack(feat), rtol=2e-2, atol=0.5)   # bf16, batched differently
assert getattr(outputs[0], "saves", None) is not None
```

### Named edits, chosen per request

`model.edit(name=...)` tags an edit; `edits=[...]` on `trace`/`invoke`/plain
`generate` (and over serve) picks which installed edits run: the named ones
listed **plus every unnamed edit**. No `edits=` runs them all; `edits=[]` the
unnamed only. An unknown name is refused (`ValueError` locally; the request's
error over serve).

<!-- test: gpu -->
```python
with model.edit(name="probe") as (tracer, probe):
    score = layer.output[1][-1].norm().item().save()
with model.edit() as (tracer, always):
    seen = nnsight.save(True)

both = model.generate("Fact:", max_tokens=1, temperature=0.0)[0]
only_unnamed = model.generate("Fact:", max_tokens=1, temperature=0.0, edits=[])[0]
with model.trace("Fact:", max_tokens=1, temperature=0.0, edits=["probe"]) as tracer:
    result = tracer.result.save()
print(sorted(both.saves), sorted(only_unnamed.saves), sorted(result.saves))
assert "score" in both.saves and "score" not in only_unnamed.saves and "score" in result.saves
try:
    model.generate("Fact:", max_tokens=1, edits=["nope"])
except ValueError as e:
    print("refused:", str(e)[:40])
model.clear_edits()
```

- Measured on Qwen3-8B, 500 prompts: **0.5 s edited** (bare vLLM 0.45 s), 0.8 s
  traced with the layer envoy bound outside — and **5.2 s traced** once the block
  also reads `model.logits`, `model.samples` or `tracer.result`. A reference to
  `model` inside a block ships the model with every invoke (~9 ms each), and those
  properties cannot be bound outside a trace. In a sweep, take the text from the
  edit's outputs.
- `model.model.layers[i]` written *inside* the block is the same trap; bind the
  layer envoy before the trace.
- Prefix caching must be off: a cached token gets no forward pass, so an edit
  sees fewer rows than the prompt has, silently. A trace forces its own recompute.
- After `edit.clear()` an output has no `.saves` attribute at all.
- On `mode="async"`: `async with model.edit()` and `await edit.aclear()`.
- `tracer.result.saves` on a traced request carries the edit's values only; the
  trace's own come back as your variables.

## Graph taps, serving, parallelism, architectures

- **[Graph taps](references/graph-taps.md)** — `taps=[...]` keeps CUDA-graph replay
  and serves the named locations from it: 96% of vanilla vLLM on one GPU against the
  eager engine's 85%, and 91% against 20% at tp=8. Only taps are served, edits land
  in place, clone what you keep, hybrid trunks pin decode-only graphs.
- **[Serving](references/serving.md)** — `nnsight-serve`, GPU-less clients with
  `trace(..., serve=url)` and `edit(serve=url)`, the CLI's forwarded flags, and
  what the server is not (an OpenAI endpoint).
- **[Tensor parallelism, MoE, hybrid trunks](references/parallel-and-architectures.md)**
  — what is gathered and what is a shard, the logit lens via `logits_processor`,
  router logits as a `(logits, bias)` pair, `linear_attn` vs `self_attn` layers.

## Not supported on the vLLM path

- **Gradients / `backward()`** and **`.source` inside a fused kernel** — the kernel's
  inputs and outputs are locations, its interior is not Python.
- **`model.scan()`** — it propagates shapes by running the model's forward under a
  fake-tensor mode, and the forward is in the worker under `torch.inference_mode`.
  It raises `NotImplementedError: scan is unavailable on vLLM: ... Trace a prompt
  and read the shapes off the activations it serves.`
- **Image/video inputs** — vision-language checkpoints load and trace on text; the
  decoder is at `model.language_model.model.layers`.
- **Cross-invoke values and `tracer.barrier`** — two traces (above).
- **Pipeline parallelism and speculative decoding** — shard with
  `tensor_parallel_size`. A `pipeline_parallel_size=2` engine builds and then fails
  every trace with `RuntimeError: UnknownPersistentIdError:
  Module:model.model.layers.12.self_attn`, naming a layer on the other stage.
- **`model.lm_head(h)`** — raises; the unembed is
  `model.logits_processor(model.lm_head, model.model.norm(h))`.

Errors raised inside the worker come back as `RuntimeError` carrying the original
type and an "Intervention traceback" pointing at your line — so catch `RuntimeError`
and match on the message, not the class. Warnings do not come back: they are emitted
by vLLM's EngineCore subprocess, so `warnings.catch_warnings()` around a trace records
nothing — a `tracer.iter` loop that outran the run is cut short with only that
engine-side warning, and your process sees a short result and nothing else. That is
the one behaviour that differs from the local path; errors are identical. Nor does anything that fails while the engine builds, including a bad `taps=`
entry: the caller gets `RuntimeError: Engine core initialization failed. See root cause
above.` and the real message is in the `(EngineCore pid=...)` lines above it.

## Choosing

| Situation | Use |
|---|---|
| one prompt, gradients, attribution, `.source`, `scan` | `TransformersModel` |
| hundreds of prompts, or a served engine | `VLLM`; sweeps through `model.edit()` |
| model larger than one GPU, locally | `VLLM` with `tensor_parallel_size` (declare `taps`) |
| model larger than your machine | NDIF — the `nnsight-remote` skill |
| token-by-token streaming | `VLLM` with `mode="async"`: `async for output in tracer.backend` |

## Related skills

- `nnsight` — the intervention API that carries over unchanged
- `tensor-parallel` — the `transformers` + `torchrun` alternative for sharding
- `nnsight-debugging` — reading the re-raised worker errors
