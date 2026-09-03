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
    scores = nnsight.save({})
    for layer in range(len(model.transformer.h)):
        with tracer.invoke(prompt):
            model.transformer.h[layer].output[:, -1, :] = 0
            scores[layer] = model.output.logits[0, -1, paris]

assert len(scores) == 12                      # one forward pass, 12 ablations
assert sorted(scores) == list(range(12))
```

**Key the results, do not append them.** Workers resume in the order the model
reaches what each asked for, so a `nnsight.save([])` filled from several invokes
comes back in model-reached order, and zipping it against the parameter list
misattributes every row:

```python
with model.trace() as tracer:
    outs = nnsight.save([])
    for i, layer in enumerate([9, 2, 6, 0]):
        with tracer.invoke(f"prompt {i}"):
            outs.append((i, model.transformer.h[layer].output.norm()))

assert [i for i, _ in outs] == [3, 1, 2, 0]   # layers 0, 2, 6, 9 — not [0, 1, 2, 3]
```

A sweep whose invokes all read the *same* module keeps source order by accident,
because those workers park on one location and resume in block order. Vary the
module and the order changes with no error. A dict keyed by the loop variable is
order-proof either way.

Give each invoke its own saved name for the same reason: three invokes that each
run `out = ....save()` leave one value behind, whichever ran last.

The effective batch grows with the number of invokes, so chunk the loop into
groups of traces if you run out of memory:

```python
def sweep(layers):
    with model.trace() as tracer:
        out = nnsight.save({})
        for layer in layers:
            with tracer.invoke(prompt):
                model.transformer.h[layer].output[:, -1, :] = 0
                out[layer] = model.output.logits[0, -1, paris]
    return out

chunked = {}
for start in range(0, 12, 6):
    chunked.update(sweep(range(start, start + 6)))
assert sorted(chunked) == list(range(12))
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

## Positions shift with the batch

`_batch` tokenizes with left padding and pads every invoke out to the batch's
longest input. A one-token prompt batched against a fourteen-token one has
activations of shape `[1, 14, 768]`:

```python
with model.trace() as tracer:
    with tracer.invoke("Hello"):
        short = model.transformer.h[6].output.save()
    with tracer.invoke("The Eiffel Tower is located in the beautiful and historic city of"):
        long = model.transformer.h[6].output.save()

with model.trace("Hello"):
    solo = model.transformer.h[6].output.save()

assert tuple(short.shape) == tuple(long.shape) == (1, 14, 768)
assert tuple(solo.shape) == (1, 1, 768)
assert torch.allclose(short[0, -1], solo[0, -1], atol=1e-3)   # numerics are fine
```

So `[:, -1]` is stable and any absolute index is not: `output[:, 0]` is the first
real token alone and a pad token here, and `output[:, SUBJECT]` only lands on the
subject when every prompt in the batch has the same length. Compute the offset from
the invoke's own token count, or index from the right.

Pad positions are not blank. The attention mask keeps them out of the real
positions, but the model computes something at each one, and those values can carry
a larger norm than the real tokens beside them:

```python
pad_norms = short[0, :-1].norm(dim=-1)
real_norm = short[0, -1].norm()
assert float(pad_norms.max()) > float(real_norm)       # 3118.5 vs 3029.8
```

A max, mean, or top-k over the sequence axis picks them up unless you mask them
out. The same padding rides along in a batched `generate`: `tracer.result` rows are
as long as the longest input plus the new tokens, and a short row decodes with
`<|endoftext|>` in front of its prompt.

## Sharing values across invokes

Invokes share the enclosing scope, so a name bound in one is visible in another.
Whether it is bound *yet* is not about which module each invoke touches. Each
worker runs until it asks for a value the model has not produced, which parks it at
a location; a parked worker resumes when the model reaches that location. So a name
bound in invoke A is readable in invoke B once B has parked at a location the model
reaches after A's binding — or at the same location, if A's block is written above
B's.

A read-only consumer buys its park with one extra `.output` access:

```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        source = model.transformer.h[5].output[:, -1]

    with tracer.invoke("The Colosseum is in"):
        model.transformer.h[6].output          # park past h[5] first
        transferred = (source.norm()).save()

assert float(transferred) > 0
```

Drop that line and the read raises `NameError` — the producing worker has not run.
Source order of the blocks does not matter; the binder can be written second, as
long as the model reaches its location first.

## Barriers: ordering a handoff the reader cannot park past

An assignment evaluates its right-hand side before the attribute access that parks
the worker, so a consumer whose first statement is a write has parked zero times
when it reads the donor — whichever module it writes to. That case, and any case
where the consumer must act at or before the producer's location, needs
`tracer.barrier(n)`: `n` blocks call it, everyone waits, the last one through
releases all.

```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)

    with tracer.invoke("Madison Square Garden is in the city of"):
        embeddings = model.transformer.wte.output
        barrier()                                    # embeddings have been read
        donor = model.output.logits[:, -1].argmax().save()

    with tracer.invoke("_ _ _ _ _ _ _ _"):
        barrier()                                    # wait for the read
        model.transformer.wte.output = embeddings
        receiver = model.output.logits[:, -1].argmax().save()

assert donor.item() == receiver.item()               # both ' New'
```

`wte` is the first module, so the receiver has nowhere earlier to park: a barrier is
the only option. Make it the default for anything that writes — park-past depends on
where two lines sit relative to each other, and inserting one line above the read
turns it into a `NameError`.

Rules: `n` must equal the number of blocks that actually call `barrier()`; it is
called (`barrier()`), not entered; it is reusable, each round waiting for its own
`n` arrivals. Counting too high ends the run with `ValueError: A barrier was never
reached by every block it waits for`, not a hang. Counting too low releases the
round early and the block it let through raises `NameError` on the value it was
waiting for, which names a variable and points nowhere near the barrier.

Use a donor name that exists nowhere outside the invokes. Each block reads its own
copy of the surrounding scope first, so an outer `donor` shadows the one the
producer binds, and nothing errors.

## What batching cannot do

| Constraint | Detail |
|---|---|
| Base `NNsight` cannot batch input invokes | Two input invokes raise `NotImplementedError`. Empty invokes always work. Implement `_batch_size`/`_batch` to add support. |
| Invokes cannot nest | Opening an invoke while the model runs raises `Cannot invoke while the model is already running.` |
| `.skip()` must cover every row | If one invoke skips a module, all of them must — a shared forward cannot run for a subset of rows. |
| A trace with no input needs an invoke | `with model.trace():` and nothing else is a `ValueError`. |
| One invoke is not "narrowed" | A lone invoke *is* the whole batch, so its write may change the leading dim and widen the run. |
| A batched write must keep its rows | With two or more invokes, a replacement is spliced back into the combined batch as given — nothing checks its height. One with the wrong leading dim builds a batch that is no longer the model's; the mismatch surfaces in a later module, or not at all. |
| `tracer.stop()` is not per-invoke | It halts the shared forward, so a sibling parked on a later location dies with `OutOfOrderError`. |
| Direct input and invokes don't mix | `with model.trace(x)` plus `tracer.invoke(y)` raises `Cannot invoke while the model is already running.` |

## Related

- [execution-model.md](execution-model.md) — why worker order is not source order
- [access-and-modify.md](access-and-modify.md)
- [control-flow.md](control-flow.md) — `skip`, `stop`, `session`
