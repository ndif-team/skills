# Caching Activations, and Scanning for Shapes

Two tools that remove most boilerplate from an interpretability script:
`tracer.cache()` grabs many modules at once, `model.scan()` tells you shapes
without running the model.

<!-- test: setup -->
```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)
prompt = "The Eiffel Tower is in the city of"
```

## One pass, many modules

The rule: **structure traces by which input you are running, not by which
activation you want.** Everything from one input comes out of one forward pass.

The explicit form — a loop inside the trace, saving into one container:

```python
with model.trace(prompt):
    resid = nnsight.save([])
    for block in model.transformer.h:
        resid.append(block.output[0, -1])          # last position, each layer

print(len(resid), resid[0].shape)                  # 12 torch.Size([768])
```

The anti-pattern is opening one trace per layer — that is N forward passes for no
reason, and N network round-trips when remote.

## tracer.cache()

For "give me everything", `tracer.cache()` records modules without naming them:

```python
with model.trace(prompt) as tracer:
    cache = tracer.cache()

print(cache["model.transformer.h.0"].output.shape)     # by path
print(cache.transformer.h[0].output.shape)             # or by navigation
print(len(cache.keys()))
```

Restrict it to what you need — caching every module of a large model is a lot of
memory:

```python
with model.trace(prompt) as tracer:
    cache = tracer.cache(modules=[model.transformer.h[0], model.transformer.h[5], "model.lm_head"])

print(sorted(cache.keys()))
```

Storage options — captured tensors land on CPU, detached, by default:

```python
with model.trace(prompt) as tracer:
    cache = tracer.cache(
        modules=[model.transformer.h[-1]],
        device=torch.device("cpu"),   # None keeps them where they are
        dtype=torch.float32,          # optional cast
        detach=True,
        include_output=True,
        include_inputs=True,          # .inputs is None unless you ask
    )

entry = cache["model.transformer.h.11"]
print(entry.output.shape, entry.input.shape)
```

### What the cache returns depends on how many times a module ran

One visit gives the value; several visits give a **list**:

```python
with model.generate(prompt, max_new_tokens=3) as tracer:
    cache = tracer.cache(modules=[model.transformer.h[-1]])

entry = cache["model.transformer.h.11"]
print(len(entry), isinstance(entry.output, list))     # 3 True
print(entry.output[0].shape, entry.output[1].shape)   # prefill, then one token
```

This makes the cache the easiest way to collect activations across generation
steps — no per-step `.save()` calls.

### Cache rules

| Rule | Why it bites |
|---|---|
| Call `tracer.cache()` **first**, right after opening the trace | Only modules reached *after* the call are captured |
| It must be called inside a trace | It attaches to the running worker |
| It observes **post-intervention** values | Cache + edit in the same trace records the edited value |
| A cache opened inside an invoke sees that invoke's rows only | Not the full batch |
| Multiple visits → list | Don't assume a tensor when generating |

```python
with model.trace(prompt) as tracer:
    cache = tracer.cache(modules=[model.transformer.h[0]])
    model.transformer.h[0].output[:] = 0

print(cache["model.transformer.h.0"].output.abs().sum().item())   # 0.0 — post-intervention
```

## model.scan() — shapes without compute

`scan` runs the model under fake tensors: shapes and dtypes are real, data is not,
no kernels run, and **the model is never dispatched**. Use it to check indexing
before spending a forward pass — or before downloading weights at all.

```python
meta = TransformersModel("openai-community/gpt2")     # no dispatch=True
print(meta.dispatched)                                # False

with meta.scan(prompt):
    hidden_size = nnsight.save(meta.transformer.h[0].output.shape[-1])
    n_layers = nnsight.save(len(meta.transformer.h))
    resid_shape = nnsight.save(tuple(meta.transformer.h[-1].output.shape))

print(hidden_size, n_layers, resid_shape)             # 768 12 (1, 10, 768)
print(meta.dispatched)                                # still False
```

Scan is a tracing context like any other: `.save()` is still required, module
access is still in forward order, and `tracer.invoke` / `tracer.cache` still work.

**You cannot branch on fake data.** Shapes are fine; values are not:

<!-- test: expect-error GuardOnDataDependentSymNode -->
```python
with meta.scan(prompt):
    if meta.transformer.h[0].output.mean() > 0:       # data-dependent — raises
        pass
```

Use scan for shape questions and a real trace for value questions. Some ops have
no fake/meta kernel and will raise under scan even though they run fine for real —
move those out of the scan block.

## Choosing between them

| You want | Use |
|---|---|
| One value from one module | `module.output.save()` |
| The same value from many modules | a loop inside one trace, saving a list |
| Everything, unknown names, or across generation steps | `tracer.cache()` |
| Shapes, no compute, possibly no weights | `model.scan()` |

## Related

- [access-and-modify.md](access-and-modify.md)
- [batching.md](batching.md) — many inputs in the same pass
- [execution-model.md](execution-model.md) — saving containers vs elements
