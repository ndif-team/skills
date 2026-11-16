# Batching: Padding, Invokes and Values Shared Between Them

[SKILL.md](../SKILL.md) shows the two ways to batch (a list, or one
`tracer.invoke` per variant) and the sweep pattern. This page is what goes wrong
past that: padding, scoping, and passing a value from one invoke to another.

```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True)
```

## Positions shift with the batch

Every row is left-padded to the batch's longest input, in a list and across
invokes alike. Only `[:, -1]` keeps its meaning. An absolute index
(`[:, 0]`, `[:, len(ids) - 1]`, a subject position) and any mean, max or top-k
over the sequence axis are wrong for every row but the longest:

```python
prompts = ["The cat sat", "The Eiffel Tower is located in the city of"]

with model.trace(prompts):
    args, kwargs = model.inputs                    # read it first
    mask = kwargs["attention_mask"].save()         # the mask the forward ran with
    hidden = model.transformer.h[6].output.save()

with model.trace(prompts[0]):
    alone = model.transformer.h[6].output.save()

def off(a, b):                                     # relative error
    return float((a - b).norm() / b.norm())

assert hidden.shape[1] == 11 and alone.shape[1] == 3          # 8 pads on row 0
assert off(hidden[0, -1], alone[0, -1]) < 1e-3                # last token: the same
assert off(hidden[0, 2], alone[0, 2]) > 0.5                   # absolute index: a pad
assert off(hidden[0].mean(0), alone[0].mean(0)) > 0.5         # plain mean: mostly pads

m = mask[..., None].to(hidden.dtype)
masked_mean = (hidden * m).sum(1) / m.sum(1)                  # per row, real tokens only
assert off(masked_mean[0], alone[0].mean(0)) < 1e-3
```

- Index from the right (`[:, -k]`), or offset by the row's own token count.
- **Pad positions are not blank.** The model computes a value at each, and its
  norm can exceed the real tokens' (on GPT-2 a pad position's residual norm is
  about 3,000 against about 90 for an ordinary token), so an unmasked reduction
  is mostly padding.
- A tokenizer that prepends BOS (Llama, Gemma) counts it in the mask. GPT-2's
  first real token is itself a norm outlier; the
  [SAE guide](sae-and-dictionary-learning.md) masks it too.
- Position ids are right under left padding, on rotary and learned-position
  models: a padded row's last-token logits match the same prompt run alone to
  float32 rounding. In bfloat16 they differ by a rounding step, enough to flip an
  argmax between two near-tied logits. Compare margins, not single tokens,
  across batch shapes.
- A `tracer.cache()` opened inside an invoke and the `tracer.result` of a batched
  `generate` keep the padding too.

## Input formats, and what belongs to the batch

Strings, lists of strings, token-id lists, tensors, a tokenizer encoding
(`{"input_ids": ..., "attention_mask": ...}`) and chat-message lists mix freely
across invokes. Tokenizer keywords on an invoke apply to that invoke:
`tracer.invoke(text, truncation=True, max_length=8)`.

All invokes reach the model in **one** forward call, so a forward or generation
keyword (`max_new_tokens`, a keyword `attention_mask=`) is decided once for the
whole batch: the last invoke to pass it wins. Anything per row goes in the
invoke's positional input.

The batch grows with the number of invokes. When memory runs out, put the trace
in a function over a slice of the variants and merge the returned dicts.

## Empty invokes see the whole batch

`tracer.invoke()` with no input adds no rows and is not scoped to any:

```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        a = model.output.logits[:, -1].save()
    with tracer.invoke(["Rome", "Berlin"]):
        b = model.output.logits[:, -1].save()
    with tracer.invoke():
        whole = model.output.logits[:, -1].save()

assert whole.shape[0] == 3 and torch.equal(torch.cat([a, b]), whole)
```

Use it for a reduction or a loss over every row. It sees module values; a name
another invoke binds follows the parking rule
[below](#sharing-values-across-invokes).

## Which values are scoped to an invoke

Scoping goes by a value's **leading dimension**. A tensor is narrowed to the
invoke's rows when its leading dimension is the batch size or a whole multiple of
it (a model that folds tokens into the batch axis, as an MoE router's
`(batch * seq, ...)` input does). Anything else is handed to every invoke whole.
That is right for an unbatched value such as a causal mask, and wrong for one
batched on another axis: a time-major `(seq, batch, hidden)` activation, a vision
tower's `(patches, hidden)`. For those a read sees the whole batch, an in-place
edit applies to every invoke, and a replacement is dropped. Both writes warn
(`... applies to every invoke: its leading dimension (256) is neither the batch
size (2) nor a multiple of it`); the read does not. Run such an input in its own
trace.

## Sharing values across invokes

Invokes share the enclosing scope, but each is a worker that runs until it asks
for a value the model has not produced. A name bound in invoke A exists for
invoke B only once B has parked at a location the model reaches after A's
binding. A reader buys that park with one extra access:

```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        source = model.transformer.h[5].output[:, -1]

    with tracer.invoke("The Colosseum is in"):
        model.transformer.h[6].output          # park past h[5] first
        transferred = source.norm().save()

assert float(transferred) > 0
```

Without that line the read raises `NameError`: the producer has not run. Simpler
whenever it fits: run the donor in its own trace first and patch from the saved
tensor, as the [activation-patching guide](activation-patching.md) does.

## Barriers

A consumer that must act at or before the producer's location cannot park past
it. `tracer.barrier(n)` is a meeting point: `n` blocks call it and the last
arrival releases them all.

```python
with model.trace() as tracer:
    barrier = tracer.barrier(2)

    with tracer.invoke("Madison Square Garden is in the city of"):
        embeddings = model.transformer.wte.output
        barrier()                                    # embeddings is bound
        donor = model.output.logits[:, -1].argmax().save()

    with tracer.invoke("_ _ _ _ _ _ _ _"):
        barrier()                                    # wait for it
        model.transformer.wte.output = embeddings
        receiver = model.output.logits[:, -1].argmax().save()

assert donor.item() == receiver.item()
```

- `n` is the number of blocks that call the barrier. It is called, not entered,
  and can be called again for a second round.
- Too high: `ValueError: A barrier was never reached by every block it waits
  for`. Too low: it releases early and the waiting block raises `NameError` on
  the value it wanted.
- Give the shared name a spelling that exists nowhere outside the invokes. An
  outer variable of the same name shadows the one the producer binds, and
  nothing errors.
- One barrier call per shared site: to patch at several layers in one trace, the
  producer and every consumer call it once per layer.

## What batching cannot do

| Constraint | Detail |
|---|---|
| Plain `NNsight` cannot batch two input invokes | `NotImplementedError: NNsight does not support batching multiple invokes`. One input invoke plus empty invokes works. |
| Invokes cannot nest or mix with a direct input | `Cannot invoke while the model is already running.` Open every invoke at the top level of an input-less `trace()`. |
| `.skip()` must cover every row | Skip the module in every invoke or in none. |
| `tracer.stop()` is not per invoke | It halts the shared forward; a sibling parked later dies with `OutOfOrderError`. |
| A replacement keeps its rows | With two or more invokes a replacement is spliced back as given. A wrong leading dimension surfaces in a later module, or not at all. |
| A processor encoding or `pixel_values` is not batched | `NotImplementedError: Can't batch these inputs; pass text or token ids.` Chat messages with the image embedded do batch: [diffusion-and-multimodal](diffusion-and-multimodal.md). |
