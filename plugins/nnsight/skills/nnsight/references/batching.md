# Batching: Many Inputs in One Pass

One `model.trace()` is one forward pass. Putting several inputs through it is the
biggest speedup available — and remotely, the difference between one request and N.

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

Inside an invoke, **index as if that input were alone**: `[:, -1, :]` means "this
invoke's rows, last position". Never compute global batch offsets yourself.

## The sweep pattern

"The same intervention with one varying parameter" is one forward pass: loop
*inside* the trace, one invoke per variant, no input on `model.trace()`.

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
reaches what each asked for, so a list filled from several invokes comes back in
model-reached order:

```python
with model.trace() as tracer:
    outs = nnsight.save([])
    for i, layer in enumerate([9, 2, 6, 0]):
        with tracer.invoke(f"prompt {i}"):
            outs.append((i, model.transformer.h[layer].output.norm()))

assert [i for i, _ in outs] == [3, 1, 2, 0]   # layers 0, 2, 6, 9 — not [0, 1, 2, 3]
```

Invokes that all read the *same* module keep source order by accident; vary the
module and it changes with no error. A dict keyed by the loop variable is
order-proof. For the same reason, give each invoke its own saved name: three
invokes that each run `out = ....save()` leave whichever ran last.

The effective batch grows with the number of invokes; if you run out of memory,
wrap the trace in a function over a slice of the parameters and call it per chunk,
merging the returned dicts.

## Empty invokes see the whole batch

`tracer.invoke()` with no input adds no rows and is not scoped — it observes every
row, for batch-wide reductions.

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

## Which values are scoped to your rows

Scoping is decided by a value's **leading dimension**, not where it came from. A
tensor is narrowed to the invoke when its leading dim is the combined batch size
**or a whole multiple of it** — the multiple covers a model that folds tokens into
the batch axis, as every transformers MoE block does before its router
(`mlp.gate.output` is `(batch*seq, ...)`, and an invoke owns `seq` rows per row it
contributed; `.source` values below the reshape too). A dim that is a multiple by
coincidence (four experts, two invokes) is sliced as well.

Anything else is handed to every invoke **whole** — right for an unbatched value
(a causal mask, rotary `cos`/`sin`), wrong for one batched on an axis nnsight
cannot recognize: a time-major `(seq, batch, hidden)` activation, a vision tower's
`(patches, hidden)`, a list of per-sequence tensors. For those:

- a **read** sees the whole batch rather than this invoke's part of it;
- an **in-place edit** applies to every invoke;
- a **replacement** is dropped — there is nowhere to splice it.

Both writes warn, naming the location, its leading dim and the batch size:

```
UserWarning: An in-place edit to `model.visual.merger.output` applies to every
invoke: its leading dimension (256) is neither the batch size (2) nor a multiple
of it, so nnsight served the whole batch rather than this invoke's rows.
```

A read does not warn — an unbatched value looks the same from its shape. When the
warning fires, run that input in its own trace.

## Forward keywords belong to the batch, not the invoke

Every invoke reaches the model in **one** forward call, so the last invoke to pass
a keyword decides it for every row (disagreement warns). Anything per-row belongs
in the invoke's *input* — pass `{"input_ids": ..., "attention_mask": ...}`
positionally, which collates per row; `attention_mask=` as a keyword replaces the
whole batch's mask. `max_new_tokens` and other generation parameters are batch-wide
too.

## Mixing input formats

Strings, lists of strings, token-id lists and tensors mix freely across invokes;
nnsight left-pads them into one forward pass. Tokenizer kwargs go on the invoke and
apply to it only: `tracer.invoke("word " * 50, truncation=True, max_length=8)`.

## Positions shift with the batch

Every invoke is left-padded to the batch's longest input, so a one-token prompt
batched against a fourteen-token one has activations of shape `[1, 14, 768]`:

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

That tolerance holds in float32. In bfloat16 — what NDIF serves — padded and solo
logits differ by up to a bf16 rounding step, enough to flip an argmax whose top two
logits nearly tie. Compare margins, not single tokens, across batch shapes.

So `[:, -1]` is stable and any absolute index is not: `output[:, 0]` is a pad token
here, and `output[:, SUBJECT]` lands on the subject only when every prompt has the
same length. Index from the right, or offset by the invoke's own token count.

Pad positions are not blank — the model computes something at each, and it can
carry a larger norm than the real tokens beside them:

```python
pad_norms = short[0, :-1].norm(dim=-1)
first_real = short[0, -1].norm()      # "Hello" is one token, so this is the first-token outlier
assert float(pad_norms.max()) > float(first_real) > 1000 > float(long[0, -1].norm())   # 3118.5, 3029.8, ~93
```

A max, mean, or top-k over the sequence axis picks them up unless you mask them —
in a per-invoke `tracer.cache()` too, which is padded the same way. A batched
`generate` keeps the padding in `tracer.result`. Take the mask the forward ran with
from `model.inputs`, first thing in the trace:

```python
prompts = ["Hello", "The Eiffel Tower is located in the city of"]
with model.trace(prompts):
    args, kwargs = model.inputs
    mask = kwargs["attention_mask"].save()
    hidden = model.transformer.h[6].output.save()
m = mask[..., None].to(hidden.dtype)
masked_mean = (hidden * m).sum(1) / m.sum(1)          # per prompt, real tokens only

with model.trace(prompts[0]):
    alone = model.transformer.h[6].output.mean(1).save()
assert (masked_mean[0] - alone[0]).norm() / alone[0].norm() < 1e-3   # unmasked: off by ~71 per dim
```

A tokenizer that adds a BOS (Llama, Gemma) counts it in `m.sum(1)`. And the first
real token is itself an outlier on GPT-2 — norm ~3030 against ~93 — so a max over
tokens needs it excluded as well as the pads (the `sae-and-dictionary-learning`
skill shows the mask).

## Sharing values across invokes

Invokes share the enclosing scope, but whether a name is bound *yet* depends on
parking. Each worker runs until it asks for a value the model has not produced, and
resumes when the model reaches it. A name bound in invoke A is readable in invoke B
once B has parked at a location the model reaches after A's binding (or at the same
location, if A's block is written above B's).

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

**Trace-body code outside the invokes runs before any of them.** It never parks,
so it cannot see a name an invoke binds:

<!-- test: expect-error NameError -->
```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        eiffel_norm = model.transformer.h[0].output.norm()
    doubled = (eiffel_norm * 2).save()       # runs first: NameError
```

Combine across invokes by creating a saved container up front — the container
itself must be saved, or it never comes back — and writing into it from inside each
invoke. Both homes for arithmetic are valid: reduce *inside* the invoke to shrink
what is transferred (the norm below), and combine across invokes *after* the block,
on the saved values:

```python
with model.trace() as tracer:
    norms = nnsight.save({})
    for name in ["The Eiffel Tower is in", "The Colosseum is in"]:
        with tracer.invoke(name):
            norms[name] = model.transformer.h[0].output[:, -1].norm()

assert len(norms) == 2
```

## Barriers: ordering a handoff the reader cannot park past

An assignment evaluates its right-hand side before the access that parks, so a
consumer whose first statement is a write reads the donor having parked zero times.
That, and any consumer that must act at or before the producer's location, needs
`tracer.barrier(n)`: `n` blocks call it, and the last arrival releases all.

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

`wte` is the first module, so the receiver has nowhere earlier to park. Make a
barrier the default for anything that writes — park-past depends on where two lines
sit, and inserting one line above the read turns it into a `NameError`.

Rules: `n` must equal the number of blocks that call `barrier()`; it is called, not
entered; it is reusable, each round waiting for its own `n` arrivals. Too high ends
the run with `ValueError: A barrier was never reached by every block it waits for`;
too low releases early, and the block let through raises `NameError` on the value it
was waiting for. Use a donor name that exists nowhere outside the invokes — an outer
`donor` shadows the one the producer binds, and nothing errors.

## What batching cannot do

| Constraint | Detail |
|---|---|
| Base `NNsight` cannot batch input invokes | Two input invokes raise `NotImplementedError`. Empty invokes always work. Implement `_batch_size`/`_batch` to add support. |
| Invokes cannot nest | Opening an invoke while the model runs raises `Cannot invoke while the model is already running.` |
| `.skip()` must cover every row | If one invoke skips a module, all of them must — a shared forward cannot run for a subset of rows. |
| A trace with no input needs an invoke | `with model.trace():` and nothing else is a `ValueError`. |
| One invoke is not "narrowed" | A lone invoke *is* the whole batch, so its write may change the leading dim and widen the run. |
| A batched write must keep its rows | With two or more invokes a replacement is spliced back as given — nothing checks its height; a wrong leading dim surfaces in a later module, or not at all. |
| `tracer.stop()` is not per-invoke | It halts the shared forward, so a sibling parked on a later location dies with `OutOfOrderError`. |
| Direct input and invokes don't mix | `with model.trace(x)` plus `tracer.invoke(y)` raises `Cannot invoke while the model is already running.` |

## Related

- [execution-model.md](execution-model.md) — why worker order is not source order
- [access-and-modify.md](access-and-modify.md)
- [control-flow.md](control-flow.md) — `skip`, `stop`, `session`
