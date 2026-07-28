# Batching: Many Inputs in One Pass

One `model.trace()` is one forward pass. Putting several inputs through it is the
single biggest speedup available in nnsight — and remotely it is also the
difference between one request and N.

<!-- test: setup -->
```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)
```

## Two ways to batch

**A list input** — one invoke, several rows, same intervention for all:

```python
with model.trace(["The Eiffel Tower is in", "The Colosseum is in"]):
    logits = model.output.logits.save()

print(logits.shape)      # [2, seq, vocab]
```

**Separate invokes** — several rows, *different* interventions per row:

```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        clean = model.output.logits[:, -1].save()

    with tracer.invoke("The Eiffel Tower is in"):
        model.transformer.h[5].output[:, -1, :] = 0      # only this row
        ablated = model.output.logits[:, -1].save()

print(clean.shape, ablated.shape)                        # both [1, vocab]
print(torch.equal(clean, ablated))                       # False
```

Inside an invoke, **index as if that input were alone**. `[:, -1, :]` means "this
invoke's rows, last position" — nnsight maps it onto the real batch. Do not try to
compute global batch offsets yourself.

## The sweep pattern

Any experiment of the form "apply the same intervention with one varying
parameter" collapses into a single forward pass: loop *inside* the trace, one
invoke per variant, no input on `model.trace()`.

```python
prompt = "The Eiffel Tower is in the city of"
paris = model.tokenizer(" Paris").input_ids[0]

with model.trace() as tracer:
    scores = nnsight.save([])
    for layer in range(len(model.transformer.h)):
        with tracer.invoke(prompt):
            model.transformer.h[layer].output[:, -1, :] = 0
            scores.append(model.output.logits[0, -1, paris])

print(len(scores))            # 12 — one forward pass, 12 ablations
```

The effective batch grows with the number of invokes, so chunk the loop into
groups of traces if you run out of memory:

```python
def sweep(layers):
    with model.trace() as tracer:
        out = nnsight.save([])
        for layer in layers:
            with tracer.invoke(prompt):
                model.transformer.h[layer].output[:, -1, :] = 0
                out.append(model.output.logits[0, -1, paris])
    return out

chunked = [value for start in range(0, 12, 6) for value in sweep(range(start, start + 6))]
print(len(chunked))          # 12
```

## Empty invokes see the whole batch

`tracer.invoke()` with no input adds no rows and is not scoped — it observes every
row contributed by the other invokes. Use it for batch-wide reductions.

```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        a = model.output.logits[:, -1].save()
    with tracer.invoke(["Rome", "Berlin"]):
        b = model.output.logits[:, -1].save()
    with tracer.invoke():
        whole = model.output.logits[:, -1].save()

print(a.shape[0], b.shape[0], whole.shape[0])       # 1 2 3
print(torch.equal(torch.cat([a, b]), whole))        # True
```

## Mixing input formats

Strings, lists of strings, token-id lists, and tensors can all be mixed across
invokes; nnsight left-pads them into one forward pass.

```python
ids = model.tokenizer("Madison Square Garden is in").input_ids

with model.trace() as tracer:
    with tracer.invoke(ids):
        pass
    with tracer.invoke(["a b c", "d e"]):
        pass
    with tracer.invoke():
        total = nnsight.save(model.output.logits.shape[0])

print(total)      # 3
```

Tokenizer kwargs go on the invoke and apply to that invoke only:
`tracer.invoke("word " * 50, truncation=True, max_length=8)`.

## Sharing values across invokes

Invokes share the enclosing scope, so a name bound in one is visible in a later
one — **provided the reader has already run past the point where it was bound.**
All invoke workers start together, so "later in the file" does not mean "later in
time".

```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        source = model.transformer.h[5].output[:, -1]

    with tracer.invoke("The Colosseum is in"):
        model.transformer.h[6].output          # park past h[5] first
        transferred = (source.norm()).save()

print(round(transferred.item(), 2))
```

Read it too early and you get `NameError` — the producing worker has not run yet.

## Barriers: handing a value across the *same* module

When two invokes read and write the **same** location, scope-sharing is not enough
— nothing orders the reader before the writer. `tracer.barrier(n)` is that
ordering: `n` blocks call it, everyone waits, the last one through releases all.

```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)

    with tracer.invoke("The Eiffel Tower is in the city of"):
        embeddings = model.transformer.wte.output
        barrier()                                    # embeddings have been read
        donor = model.output.logits[:, -1].argmax().save()

    with tracer.invoke("_ _ _ _ _ _ _ _"):
        barrier()                                    # wait for the read
        model.transformer.wte.output = embeddings
        receiver = model.output.logits[:, -1].argmax().save()

print(model.tokenizer.decode(donor), "|", model.tokenizer.decode(receiver))
```

Both prompts now predict the same token: the second one is running on the first
one's embeddings.

Rules: `n` must equal the number of invokes that actually call `barrier()`; it is
called (`barrier()`), not entered; it is reusable — each round waits for its own
`n` arrivals.

## What batching cannot do

| Constraint | Detail |
|---|---|
| Base `NNsight` cannot batch input invokes | Two input invokes raise `NotImplementedError`. Empty invokes always work. Implement `_batch_size`/`_batch` to add support. |
| Invokes cannot nest | Opening an invoke while the model runs raises `Cannot invoke while the model is already running.` |
| `.skip()` must cover every row | If one invoke skips a module, all of them must — a shared forward cannot run for a subset of rows. |
| A trace with no input needs an invoke | `with model.trace():` and nothing else is a `ValueError`. |
| One invoke is not "narrowed" | A lone invoke *is* the whole batch. |

## Related

- [execution-model.md](execution-model.md) — why worker order is not source order
- [access-and-modify.md](access-and-modify.md)
- [control-flow.md](control-flow.md) — `skip`, `stop`, `session`
