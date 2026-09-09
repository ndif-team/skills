# Execution Model

Why nnsight code blocks, deadlocks, or returns nothing. Read this once — most
confusing nnsight behavior follows from it.

## The body of a trace does not run where you wrote it

<!-- test: skip -->
```python
with model.trace("hello"):
    h = model.transformer.h[5].output
    print("this does not print here")
```

`with model.trace(...)` does not execute its body inline. On `__enter__`, nnsight
parses the block's **source**, compiles it, and raises an internal exception to
skip past it. On `__exit__` it runs the compiled block *interleaved with the
model's forward pass*, then copies the saved names back into your frame.

Three consequences you feel immediately:

1. **Values must be `.save()`d to escape.** Ordinary assignment inside the block
   is invisible afterwards — the block ran in a different frame.
2. **`.output` reads block.** Reading `model.transformer.h[5].output` parks your
   code until the model actually reaches layer 5 and produces that tensor.
3. **The block's source must be available.** Code typed into a bare REPL where
   `inspect.getsource` fails cannot be traced. Files, IPython, and notebooks are
   fine.

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

There is no `.value` to unwrap and no proxy type. If a variable inside a trace
does not behave like a tensor, it is because it genuinely is not one (e.g. a
module returned a tuple).

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
exits. Two rules cover every mistake:

**1. Bind what you save.** The push-back returns the block's *locals*, filtered to
the marked ones. A bare `model.output.logits.save()` on its own line marks a value
with no name to return it under, so it silently never arrives.

```python
with model.trace("hello"):
    logits = model.output.logits.save()          # right — bound to a name
```

**2. Save the container, not the elements.** When collecting across layers or
steps, mark the list itself and append raw values.

```python
with model.trace("The Eiffel Tower is in the city of"):
    per_layer = nnsight.save([])                 # the list is what comes back
    for block in model.transformer.h:
        per_layer.append(block.output[0, -1].mean())

print(len(per_layer))                            # 12
```

Appending `x.save()` into an unsaved list happens to work locally (you are
mutating a list in your own frame) and silently returns nothing remotely. Save the
container.

Two spellings, same thing: `nnsight.save(x)` and `x.save()`. The method form is
mounted onto every Python object by a C extension; prefer `nnsight.save(...)` for
plain builtins (lists, dicts, ints) and for code that must not depend on that
mount.

**`.save()` outside a trace raises.** There is no trace to return the value from:

<!-- test: expect-error ValueError -->
```python
import nnsight
buffer = nnsight.save([])          # no trace running
```

## Saving vs. escaping in nested contexts

Only the **outermost** trace filters on saves. Inner traces (a trace inside a
`model.session()`, or a `tensor.backward()` block inside a trace) push everything
up to their parent, so you only need `.save()` at the outer boundary.

```python
with model.session() as session:
    with model.trace("The Eiffel Tower is in"):
        a = model.transformer.h[-1].output[0, -1]     # no .save() needed here
    with model.trace("The Colosseum is in"):
        b = model.transformer.h[-1].output[0, -1]
        similarity = torch.nn.functional.cosine_similarity(a, b, dim=0).save()

print(round(similarity.item(), 3))
```

## What runs where

Inside a trace body you can use ordinary Python — `if`, `for`, function calls,
list comprehensions — and it runs in the worker greenlet against real values.
The cost is that anything you touch must be reachable there: for **remote**
execution the body is serialized and shipped, so locals from your file need
`nnsight.register(...)`. See the `nnsight-remote` skill.

## Mental checklist when something misbehaves

| Symptom | Almost always |
|---|---|
| `NameError` / `UnboundLocalError` after the block | forgot `.save()` (`NameError` in a script, `UnboundLocalError` in a function) |
| Saved list is empty | saved the elements, not the container |
| `OutOfOrderError` | modules touched out of forward order |
| `ValueError: Cannot access ... outside of interleaving` | `.output` read outside a trace body |
| Everything after a loop vanished | unbounded `tracer.iter[:]` / `tracer.all()` — see [generation.md](generation.md) |
| `NameError` on a cross-invoke value | needs `tracer.barrier(n)` — see [batching.md](batching.md) |

Full error catalogue: the `nnsight-debugging` skill.
