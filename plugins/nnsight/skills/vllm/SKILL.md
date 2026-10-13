---
name: vllm
description: Run nnsight interventions inside the vLLM inference engine — continuous batching, tensor parallelism, CUDA-graph taps, async streaming, engine-wide edit() and nnsight-serve. Use for throughput, many concurrent prompts, streamed tokens, or a served engine; for a sharded model that needs gradients, .source or the full TransformersModel API, use the tensor-parallel skill instead. Covers flat [tokens, hidden] rows, the (hidden, residual) layer pair, live buffers, model.logits/model.samples, one prompt per invoke, model.edit() sweeps, taps, and why .skip() can kill the engine.
---

# vLLM

`VLLM` runs your block inside vLLM's worker process: the intervention code is
serialized onto each request, run against the real modules, and the saved values
shipped back. You get PagedAttention, continuous batching, tensor parallelism and
CUDA-graph replay (with `taps=`), and the same trace syntax.

nnsight targets **vLLM 0.27.1 and its V1 engine**. `pip install nnsight[vllm]`
does not cap the version, so a newer vLLM can be installed and then fail at
`import` with `ModuleNotFoundError` — pin `vllm==0.27.1`. nnsight sets
`VLLM_USE_V2_MODEL_RUNNER=0` itself and refuses an engine started with it set to
`1` (`NotImplementedError`: that runner is not instrumented).

The examples use `HuggingFaceTB/SmolLM2-135M-Instruct`, a Llama-style trunk, so
what they show about layer outputs holds for Llama, Qwen and Mistral.

## What breaks first: four things a vLLM block sees differently

<!-- test: gpu setup requires vllm -->
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
   residual add into the next layer's norm, so `output[0]` is this layer's **MLP
   output** and `output[1]` the residual stream **after this layer's attention** —
   not the stream entering the layer, which is `layer.inputs[0][1] +
   layer.inputs[0][2]`. The stream *after* the layer is `out[0] + out[1]`.
   Patching `output[0]` alone changes almost nothing. Writing either element
   steers (the next norm adds them).
3. **Clone what you keep.** A served value is the model's live buffer, rewritten by
   the next fused kernel after your block returns — the un-cloned `alias` above
   comes back holding later data. Reduce or `.clone()` before saving.
   (`NNSIGHT_VLLM_CLONE_READS=1`, set before `VLLM(...)`, serves every read as a
   copy instead; in-place edits then never reach the model, so write by
   assignment.) Parameters need the same care:
   `layer.weight.save()` fails with `Can't get local object
   'Interleaver.instrument.<locals>.forward'` (a vLLM parameter carries a loader
   bound to its instrumented layer); save `weight.detach().clone()`.
4. **Where tensors live.** A tensor referenced from outside the block (a steering
   vector) travels with the block; move it onto the served value with
   `v.to(h.device, h.dtype)`. Saved tensors come back on the worker's device unless
   you `.cpu()` them inside. `model.device` on an undispatched client is `meta`.

`model.logits` is `[1, vocab]` for the step, `model.samples` `[1, 1]`. Greedy is
`temperature=0.0`; vLLM's default is `1.0`.

## Loading and the constructor

`VLLM(repo_id, *, mode="sync"|"async", dispatch=False, tensor_parallel_size=1,
gpu_memory_utilization=0.9, taps=(), **vllm_kwargs)`. `dispatch=False` builds a
meta tree only; every other keyword goes to vLLM's engine args. The engine runs
eagerly, serving every location, unless you declare
[`taps=`](references/graph-taps.md).

**A script needs an `if __name__ == "__main__":` guard** around `VLLM(...)`, at
any `tensor_parallel_size` (notebooks are exempt). Without it the engine's spawned
process re-imports your module and dies with `An attempt has been made to start a
new process before the current process has finished its bootstrapping phase`.
Pass `enable_prefix_caching=False` whenever you will `model.edit()`.

## Per-step values and generation

Sampling kwargs go on `trace(...)` / `invoke(...)`, not the constructor. Step 0
under `tracer.iter` is the prefill; step *k* serves the *k*-th generated token.
Python locals persist across steps. Within a step, order is forward order: a write
to a layer *below* one already read this step parks until that layer's next visit
— on the last step there is none, and nothing after the loop runs. Put upstream
writes at the top of the loop body, downstream reads after.

<!-- test: gpu requires vllm -->
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

assert result.outputs[0].token_ids[2] == 3_000 and steps[2] == (2, False, 3_000)
assert steps[3][1] is True
```

- **`max_tokens` is a cap, not a promise.** An EOS, a `stop` string or
  `tracer.stop()` ends the request early, and a `tracer.iter[:N]` loop the run
  cannot supply is cut short: what it saved comes home, the statements after the
  loop never run, and the only notice is a warning you never see (see "Errors and
  warnings"). Check the `len()` of what you collected, and hold the run to the
  count with `ignore_eos=True` or `min_tokens=N`.
- **`tracer.result` must be the last read** — it is served after every module,
  `logits` and `samples` visit; a read after it raises `OutOfOrderError`.
- Assigning `model.samples` forces the token the engine continues from;
  assigning `model.logits` changes what the sampler sees.

## One prompt per invoke; collecting across invokes

Each `tracer.invoke(prompt)` is one vLLM request; the engine batches them. A name
saved in every invoke comes back as a **list** in invoke order. A container saved
*above* the invokes is copied per request: assigning into pre-sized slots merges
back, **`append` does not** — you get one element.

<!-- test: gpu requires vllm -->
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

assert len(last) == 3 and all(s is not None for s in slots) and len(appended) == 1
```

A list of prompts in one invoke is rejected. Async traces fired with
`asyncio.gather` batch the same way, each carrying its own saves.

### Passing values between invokes: two traces

Invokes are separate requests: a value one reads is **not** visible to a sibling
(`NameError`), `tracer.barrier` raises, and `session()` does not bridge traces.
Save in one trace and reference it in the next. Activation patching:

<!-- test: gpu requires vllm -->
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

assert patched > base
```

Writing *into* the served value, as above, always fits. A whole-value replacement
(`layer.output = t`) must keep the rows the block owns, and nothing checks: a donor
from a different prompt length can land as a device-side assert that kills the
engine and every request in it. Slice the donor to the rows you write.

## Sampling parameters and `n > 1`

Trace-level kwargs fill in what an invoke did not name; anything an invoke names
wins, including vLLM's own defaults (`temperature=1.0`, `max_tokens=16`).
Anything `vllm.SamplingParams` accepts works (`top_p`, `top_k`, `seed`, `stop`,
`logprobs`, `lora_request`, ...). `n=k` fans one prompt into `k` sequences: the
block runs once per sequence, saved names come back as a list per sequence, and a
saved container comes back as a list of `k` containers matching
`result.outputs[i]`.

<!-- test: gpu requires vllm -->
```python
with model.trace("My favourite colour is", temperature=0.9, seed=0, n=3, max_tokens=4, ignore_eos=True) as tracer:
    caps = nnsight.save([])
    for _ in tracer.iter[:4]:
        caps.append(model.model.layers[8].output[1][-1].clone())
    result = tracer.result.save()

assert len(caps) == 3 and len(caps[1]) == 4 and len(result.outputs) == 3
```

## Sweeps: `model.edit()` and the per-invoke cost

`model.edit()` sends the block once and it runs for every request the engine
serves afterwards — plain `model.generate(...)` calls and other clients of a served
engine included — with each request's values on that request's `.saves`. No
`tracer.invoke` inside an edit; `tracer.all()` makes it follow every step.

<!-- test: gpu requires vllm -->
```python
sweep = [f"Fact number {i} about the ocean is that it" for i in range(200)]
layer = model.model.layers[20]                     # bind the envoy OUTSIDE the block

with model.edit() as (tracer, edit):
    out = layer.output
    feat = (out[0] + out[1]).mean(0).cpu().save()

outputs = model.generate(sweep, max_tokens=1, temperature=0.0)      # plain requests, block runs
feats = torch.stack([o.saves["feat"] for o in outputs])
texts = [o.outputs[0].text for o in outputs]
edit.clear()

assert feats.shape == (200, 576) and len(texts) == 200
```

The same sweep as one trace with one invoke per prompt ships the block per
invoke, and a block that references `model` inside it (`model.logits`,
`model.samples`, `tracer.result`, `model.model.layers[i]`) ships the model too,
~9 ms per invoke. On Qwen3-8B, 500 prompts: 0.5 s edited (bare vLLM 0.45 s), 0.8 s
traced with the envoy bound outside, 5.2 s once the block reads `model.samples`.
Bind layer envoys outside the block, and take the text from the edit's outputs.

### Named edits, chosen per request

`model.edit(name=...)` tags an edit; `edits=[...]` on `trace`/`invoke`/`generate`
runs the named ones listed **plus every unnamed edit**. No `edits=` runs them all;
`edits=[]` the unnamed only; an unknown name raises `ValueError`.

<!-- test: gpu requires vllm -->
```python
with model.edit(name="probe") as (tracer, probe):
    score = layer.output[1][-1].norm().item().save()
with model.edit() as (tracer, always):
    seen = nnsight.save(True)

both = model.generate("Fact:", max_tokens=1, temperature=0.0)[0]
only_unnamed = model.generate("Fact:", max_tokens=1, temperature=0.0, edits=[])[0]
with model.trace("Fact:", max_tokens=1, temperature=0.0, edits=["probe"]) as tracer:
    result = tracer.result.save()
assert "score" in both.saves and "score" not in only_unnamed.saves and "score" in result.saves
model.clear_edits()
```

Prefix caching must be off for an edit: a cached token gets no forward pass, so
the edit silently sees fewer rows than the prompt has (a trace forces its own
recompute). After `edit.clear()` an output has no `.saves` attribute at all.
`tracer.result.saves` on a traced request carries the edit's values only.

## Graph taps, serving, parallelism, architectures

- **[Graph taps](references/graph-taps.md)** — `taps=[...]` keeps CUDA-graph replay
  and serves only the named locations, at 91–96% of plain vLLM's throughput;
  edits land in place, clone what you keep.
- **[Serving](references/serving.md)** — `nnsight-serve`, GPU-less clients with
  `trace(..., serve=url)` and `edit(serve=url)`, the async engine, and what the
  server is not (an OpenAI endpoint).
- **[Tensor parallelism, MoE, hybrid trunks](references/parallel-and-architectures.md)**
  — what is gathered and what is a shard, the logit lens via `logits_processor`
  (`model.lm_head(h)` raises), router logits as a `(logits, bias)` pair,
  `linear_attn` vs `self_attn` layers, vision-language checkpoints.

## `.skip()` can take the whole engine down

A skip must replace **every row of the step**, and a step's rows are whatever the
scheduler batched — other invokes, other clients' requests. When the replacement
does not tile them, vLLM treats the error as fatal: `EngineDeadError`, every
in-flight request lost. **Treat `.skip()` as unsupported once more than one
request can be in flight**, including other traffic on a shared engine.

On one prompt with nothing else running, note that **a decoder layer's `.input` is
the positions tensor** (`forward(positions, hidden_states, residual)`), so
`module.skip(module.input)` kills the engine. The pass-through is:

<!-- test: gpu slow requires vllm -->
```python
with model.trace("The capital of France is", temperature=0.0, max_tokens=4, ignore_eos=True) as tracer:
    args, _ = model.model.layers[10].inputs
    model.model.layers[10].skip((args[1], args[2]))       # hidden, residual
    result = tracer.result.save()
assert len(result.outputs[0].token_ids) == 4
```

To ablate a layer without that risk, write its output instead.

## Not supported on the vLLM path

- **Gradients / `backward()`** and **`.source` inside a fused kernel** — the kernel's
  inputs and outputs are locations, its interior is not Python.
- **`model.scan()`** — raises `NotImplementedError: scan is unavailable on vLLM:
  ... Trace a prompt and read the shapes off the activations it serves.`
- **Cross-invoke values and `tracer.barrier`** — two traces (above).
- **Pipeline parallelism and speculative decoding** — shard with
  `tensor_parallel_size`. A `pipeline_parallel_size=2` engine builds and then fails
  every trace with `RuntimeError: UnknownPersistentIdError:
  Module:model.model.layers.12.self_attn`, naming a layer on the other stage.

## Errors and warnings

Errors raised inside the worker come back as `RuntimeError` carrying the original
type and an "Intervention traceback" — match on the message, not the class.
**Warnings do not come back**: they are emitted in vLLM's EngineCore subprocess,
so `warnings.catch_warnings()` records nothing. A failure while the engine builds
(a bad `taps=` entry included) arrives as `RuntimeError: Engine core initialization
failed`; the real message is in the `(EngineCore pid=...)` log lines above it.

## Choosing

| Situation | Use |
|---|---|
| one prompt, gradients, attribution, `.source`, `scan` | `TransformersModel` |
| hundreds of prompts, or a served engine | `VLLM`; sweeps through `model.edit()` |
| model larger than one GPU, for throughput | `VLLM` with `tensor_parallel_size` (declare `taps`) |
| model larger than one GPU, with gradients / `.source` / the full API | the `tensor-parallel` skill |
| model larger than your machine | NDIF — the `remote` skill |
| token-by-token streaming | `VLLM` with `mode="async"`: `async for output in tracer.backend` |

The intervention API itself is the `nnsight` skill's; reading the re-raised worker
errors is the `debugging` skill's.
