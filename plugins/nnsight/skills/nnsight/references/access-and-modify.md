# Access and Modify Activations

Everything you do to a model's internals goes through three properties on any
module (`Envoy`), read or written **inside a trace body**.

| Property | Is | Assignable |
|---|---|---|
| `module.output` | the module's forward return value | yes |
| `module.input` | its first positional arg (or first kwarg if none) | yes |
| `module.inputs` | `(args, kwargs)` — everything it was called with | yes |

<!-- test: setup -->
```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)
prompt = "The Eiffel Tower is in the city of"
```

```python
with model.trace(prompt):
    resid = model.transformer.h[5].output.save()         # read
    mlp_in = model.transformer.h[6].mlp.input.save()     # read an input
    logits = model.output.logits.save()

print(resid.shape, mlp_in.shape, logits.shape)
```

## Know what a module actually returns

**Do not assume `.output[0]`.** In `transformers` 5, a GPT-2 *block* returns a
plain tensor, while the *attention* submodule returns a tuple. Older nnsight
examples on the internet write `h[i].output[0]` everywhere, which silently indexes
the batch dimension instead of unpacking a tuple.

This is a `transformers` fact, and it moved: in 4.x the same block returned
`(hidden_states,)`, so `.output[0]` was correct there. The answer differs per
version and per architecture
([modules-and-architectures.md](modules-and-architectures.md#tensor-or-tuple) has
the table), so check once at the top of an experiment:

```python
with model.trace(prompt):
    attn_out = model.transformer.h[0].attn.output    # attn runs before the block ends
    block_out = model.transformer.h[0].output
    kinds = nnsight.save({
        "block": type(block_out).__name__,
        "attn": type(attn_out).__name__,
    })

assert kinds == {"block": "Tensor", "attn": "tuple"}
```

Note the order: a block's `attn` and `mlp` return *before* the block does, so read
them first. `model.scan(...)` gets the same answer with no real compute — see
[caching-and-scan.md](caching-and-scan.md).

## In-place vs replacement

Both work, and they differ:

```python
with model.trace(prompt):
    # IN-PLACE: mutate the tensor the model is holding.
    model.transformer.h[0].output[:] = 0
    zeroed = model.transformer.h[0].output.mean().save()

    # REPLACEMENT: hand the model a different object to continue with.
    model.transformer.h[1].output = model.transformer.h[1].output * 2

    logits = model.output.logits.save()

print(zeroed.item())     # 0.0
```

Use in-place (`[:] =`, `[..., idx] =`) for surgical edits to part of a tensor; use
replacement when you are producing a new tensor wholesale. Replacement is also the
only option when the value is not a tensor (a tuple, a dataclass output).

A replacement built from a fresh tensor (`torch.zeros_like(output)`) cuts the
autograd graph there; see [gradients.md](gradients.md#replacements-and-the-graph)
if gradients are involved.

**Mutating a tuple element by assignment fails** — `output[0] = x` is
`tuple.__setitem__`. Either mutate in place through the element, or rebuild:

```python
with model.trace(prompt):
    # in-place through the tuple's first element
    model.transformer.h[0].attn.output[0][:] = 0

    # or rebuild the tuple and assign the whole thing
    attn = model.transformer.h[1].attn.output
    model.transformer.h[1].attn.output = (torch.zeros_like(attn[0]),) + tuple(attn[1:])

    out = model.output.logits.save()
```

## Positions and heads

Activations are `(batch, seq, hidden)`. Positions index `seq`; `-1` is the last
token, where next-token prediction is read (`output[:, -1, :] += v` steers only it).

Heads live inside the hidden dimension: head `i` is the slice
`[..., i * head_dim : (i + 1) * head_dim]` of the attention output projection's
*input*. The attention module's own `.output` is past that projection, so the same
slice there mixes every head:

```python
# read it off the module: equals n_embd // n_head on GPT-2, not on GQA/Gemma-style models
head_dim = model.transformer.h[5].attn._module.head_dim
head = 4

with model.trace(prompt):
    proj_in = model.transformer.h[5].attn.c_proj.input
    proj_in[:, :, head * head_dim : (head + 1) * head_dim] = 0    # ablate one head
    logits = model.output.logits.save()

print(logits.shape)
```

For a reusable, writable per-head view, define an `eproperty` — see
[modules-and-architectures.md](modules-and-architectures.md#adding-your-own-served-value-eproperty-envoys).

## Clone when you need the "before"

Reads after an in-place edit return the edited tensor. Clone to keep a snapshot:

```python
with model.trace(prompt):
    before = model.transformer.h[0].output.clone().save()
    model.transformer.h[0].output[:] = 0
    after = model.transformer.h[0].output.save()

print(before.abs().sum().item() > 0, after.abs().sum().item() == 0)   # True True
```

The same holds for anything you save: a saved tensor aliases the live one, so
clone it if you also write to that location afterwards.

## Tensors you create must land on the model's device

Build new tensors on the activation's device and dtype:

```python
with model.trace(prompt):
    resid = model.transformer.h[6].output
    direction = torch.ones(resid.shape[-1], device=resid.device, dtype=resid.dtype)
    model.transformer.h[6].output[:, -1, :] += 3.0 * direction
    logits = model.output.logits.save()
```

Deriving them from the activation, not `model.device`, holds under `device_map`
sharding and mixed precision.

## Setting inputs

`.input = value` repacks into the underlying `(args, kwargs)` for you
(`h[3].input = h[3].input * 0.5`). For anything beyond the first argument, assign
the whole pair through `.inputs`:

```python
with model.trace(prompt):
    args, kwargs = model.transformer.h[3].inputs
    model.transformer.h[3].inputs = ((args[0] * 0.5, *args[1:]), kwargs)
    keys = nnsight.save(sorted(kwargs.keys()))
    halved = model.output.logits.save()

print(keys, halved.shape)
```

## Applying a module to a value

Calling an envoy inside a trace runs the module with nnsight stood down: it serves
no value and spends no occurrence, so it is not bound by the ordering rule and does
not consume the module's real place in the forward pass. This is how logit lens is
written:

```python
with model.trace(prompt):
    mid = model.transformer.h[5].output
    decoded = model.lm_head(model.transformer.ln_f(mid))    # apply out of order
    top = decoded[0, -1].argmax(dim=-1).save()

print(model.tokenizer.decode(top))
```

Pass `hook=True` when you want the called module's own internals observable (an
attached SAE or adapter) — see
[modules-and-architectures.md](modules-and-architectures.md#attaching-your-own-module-sae-probe-adapter).

## Reading outside a trace

`.output` raises outside interleaving — open a trace for values, or `model.scan(...)`
for shapes:

<!-- test: expect-error ValueError -->
```python
hidden = model.transformer.h[0].output     # no trace running
```

## Name collisions

A child module named like an nnsight property (BERT's `output` submodule) moves to
`.E_output`; see [api-reference.md](api-reference.md#envoy-properties-any-module).

## Related

- [execution-model.md](execution-model.md) — why reads block and what order they must be in
- [batching.md](batching.md) — doing this for several prompts in one pass
- [source-tracing.md](source-tracing.md) — reaching values *inside* a module's forward
- [caching-and-scan.md](caching-and-scan.md) — grabbing many modules at once
