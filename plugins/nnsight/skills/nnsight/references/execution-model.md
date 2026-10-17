# Execution Model

Why nnsight code blocks, deadlocks, or returns nothing. Read this once — most
confusing nnsight behavior follows from it.

## The body of a trace does not run where you wrote it

`with model.trace(...)` does not execute its body inline. On `__enter__`, nnsight
parses the block's **source**, compiles it, and raises an internal exception to
skip past it. On `__exit__` it runs the compiled block *interleaved with the
model's forward pass*, then copies the saved names back into your frame.

So values must be `.save()`d to escape (the block ran in a different frame),
`.output` reads park until the model produces that tensor, and the block's source
must be available — a bare REPL where `inspect.getsource` fails cannot trace;
files, IPython and notebooks are fine.

## Interleaving: your code and the model take turns

The block runs in a **greenlet** (a cooperative worker, not a thread) called a
*Mediator*. Every module's `forward` is a controller that hands the module's input
and output to the interleaver. When your code asks for a value the model has not
produced yet, the worker parks; when the model reaches that location, the worker
wakes with the real tensor, runs until its next request, and hands control back.

So inside a trace you hold **real `torch.Tensor`s**, not proxies or futures:

<!-- test: setup -->
```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

with model.trace("The Eiffel Tower is in the city of"):
    hidden = model.transformer.h[5].output
    assert isinstance(hidden, torch.Tensor)      # a real tensor, right now
    assert hidden.shape[-1] == 768
    peak = hidden.abs().max().item()             # .item() works — it is not a proxy
    mean = hidden.mean().save()

print(round(mean.item(), 4))
```

There is no `.value` to unwrap and no proxy type. A variable that does not behave
like a tensor genuinely is not one (e.g. a module returned a tuple).

## The ordering rule

Within one invoke, you must touch modules in **forward-pass order**. The worker
parks on each request; asking for layer 8 and then layer 2 means the model has
already run past layer 2 when you ask, and nothing will ever wake you.

<!-- test: expect-error OutOfOrderError -->
```python
with model.trace("hello"):
    late = model.transformer.h[8].output
    early = model.transformer.h[2].output      # model already ran past h[2]
```

This surfaces as `OutOfOrderError` at the end of the run, not as a hang:

```
OutOfOrderError: 'model.transformer.h.2.output.i0' was requested but the model
already ran past it
```

Fixes, in order of preference:

| Situation | Fix |
|---|---|
| You just wrote the reads in the wrong order | Reorder them into forward order |
| You need one prompt's layer 8 and another's layer 2 | Put each in its own `tracer.invoke(...)` — workers are independent |
| You need "everything", order unknown | `tracer.cache()` (see [caching-and-scan.md](caching-and-scan.md)) |
| You need a value from a later module *before* an earlier one, same input | Two traces, or cache the first pass |

`tracer.result` is the last thing anything can ask for — it is served after the
forward returns. Reading it before `model.output` puts `model.output` out of
order.

Order is per-module-execution, not per-line: inside one block, `h[0].attn` comes
before `h[0].mlp` comes before `h[0].output` comes before `h[1]`.

## `.save()` semantics

`.save()` marks a value to be pushed back to your frame when the outermost trace
exits. The push-back returns the block's *locals*, filtered to the marked ones —
which is why both rules in SKILL.md hold:

- **Bind what you save.** A bare `model.output.logits.save()` on its own line marks
  a value with no name to return it under, so it silently never arrives.
- **Save the container, not the elements.** A list is returned only if the list
  itself is marked. Appending `x.save()` into an unsaved list happens to work
  locally (you mutated a list in your own frame) and returns nothing remotely.

`nnsight.save(x)` and `x.save()` are the same; the method form is mounted onto
every object by a C extension, so prefer `nnsight.save(...)` for plain builtins and
for code that must not depend on that mount. Either raises `ValueError` outside a
trace — there is nothing to return the value from.

Only the **outermost** context filters on saves. An inner trace (inside a
`model.session()`) or a `tensor.backward()` block pushes everything up to its
parent, so `.save()` is needed only at the outer boundary — see
[control-flow.md](control-flow.md#modelsession--several-traces-one-scope).

Trace-body code outside any `tracer.invoke` runs **before** the invoke workers, so
it can never read a name an invoke binds; accumulate into a saved container from
inside the invokes instead — see
[batching.md](batching.md#sharing-values-across-invokes). The body is ordinary
Python against real values; for **remote** execution it is serialized and shipped,
so locals from your own files need `nnsight.register(...)` (the `remote` skill).

## Mental checklist when something misbehaves

| Symptom | Almost always |
|---|---|
| `NameError` / `UnboundLocalError` after the block | forgot `.save()` (`NameError` in a script, `UnboundLocalError` in a function) |
| Saved list is empty | saved the elements, not the container |
| `OutOfOrderError` | modules touched out of forward order |
| `ValueError: Cannot access ... outside of interleaving` | `.output` read outside a trace body |
| Everything after a loop vanished | unbounded `tracer.iter[:]` / `tracer.all()` — see [generation.md](generation.md) |
| `NameError` on a cross-invoke value | the reader has not parked past the binder — park it, or `tracer.barrier(n)`; see [batching.md](batching.md#sharing-values-across-invokes) |
| `NameError` in trace-body code outside the invokes | that code runs before every invoke; write into a `nnsight.save({})` from inside the invokes |

Full error catalogue: the `debugging` skill.
