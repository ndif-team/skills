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

This is a `transformers` fact, not an nnsight one, and it moved recently: in
`transformers` 4.x the same block returned `(hidden_states,)`, so `.output[0]` was
correct there. Check the version before porting either way — and check the model,
since the answer differs per architecture.

```python
with model.trace(prompt):
    attn_out = model.transformer.h[0].attn.output    # attn runs before the block ends
    block_out = model.transformer.h[0].output
    kinds = nnsight.save({
        "block": type(block_out).__name__,
        "attn": type(attn_out).__name__,
    })

print(kinds)     # {'block': 'Tensor', 'attn': 'tuple'}
```

Note the order: a block's `attn` and `mlp` produce their outputs *before* the block
itself returns, so they must be read first. Reading `h[0].output` and then
`h[0].attn.output` is an `OutOfOrderError` — see
[execution-model.md](execution-model.md).

When unsure, check once at the top of an experiment:

```python
with model.trace(prompt):
    attn_is_tuple = isinstance(model.transformer.h[0].attn.output, tuple)
    mlp_shape = tuple(model.transformer.h[0].mlp.output.shape)
    block_shape = tuple(model.transformer.h[0].output.shape)
    shape_info = nnsight.save({
        "attn is tuple": attn_is_tuple,
        "mlp": mlp_shape,
        "h[0]": block_shape,
    })

print(shape_info)
```

Or use `model.scan(...)`, which gets shapes with no real compute — see
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
token, which is where next-token prediction is read.

```python
with model.trace(prompt):
    # steer only the final position
    model.transformer.h[6].output[:, -1, :] += 5.0
    logits = model.output.logits.save()

print(model.tokenizer.decode(logits[0, -1].argmax()))
```

For attention, heads live inside the hidden dimension: head `i` of a
`(batch, seq, n_heads * head_dim)` tensor is the slice
`[..., i * head_dim : (i + 1) * head_dim]`. The clean place to cut is the
attention output projection's *input*:

```python
n_heads = model.config.n_head
head_dim = model.config.n_embd // n_heads
head = 4

with model.trace(prompt):
    proj_in = model.transformer.h[5].attn.c_proj.input
    proj_in[:, :, head * head_dim : (head + 1) * head_dim] = 0    # ablate one head
    logits = model.output.logits.save()

print(logits.shape)
```

## Clone when you need the "before"

Reads after an in-place edit return the edited tensor. Clone to keep a snapshot:

```python
with model.trace(prompt):
    before = model.transformer.h[0].output.clone().save()
    model.transformer.h[0].output[:] = 0
    after = model.transformer.h[0].output.save()

print(before.abs().sum().item() > 0, after.abs().sum().item() == 0)   # True True
```

Cloning also matters when a slice you captured is shared across invokes — mutate a
copy and assign it back rather than writing through a view.

## Tensors you create must land on the model's device

Anything you build inside the trace has to match the activation you are combining
it with:

```python
with model.trace(prompt):
    resid = model.transformer.h[6].output
    direction = torch.ones(resid.shape[-1], device=resid.device, dtype=resid.dtype)
    model.transformer.h[6].output[:, -1, :] += 3.0 * direction
    logits = model.output.logits.save()
```

`model.device` gives the model's device outside a trace; deriving `device=` and
`dtype=` from the activation itself (as above) is more robust under `device_map`
sharding and mixed precision.

## Setting inputs

`.input` assignment repacks correctly into the underlying `(args, kwargs)`:

```python
with model.trace(prompt):
    model.transformer.h[3].input = model.transformer.h[3].input * 0.5
    out = model.output.logits.save()
```

For anything beyond the first argument, go through `.inputs`:

```python
with model.trace(prompt):
    args, kwargs = model.transformer.h[3].inputs
    keys = nnsight.save(sorted(kwargs.keys()))

print(keys)
```

## Applying a module to a value

Calling an envoy inside a trace runs its `forward` directly — no hooks, no
ordering constraints. This is how logit lens is written:

```python
with model.trace(prompt):
    mid = model.transformer.h[5].output
    decoded = model.lm_head(model.transformer.ln_f(mid))    # apply out of order
    top = decoded[0, -1].argmax(dim=-1).save()

print(model.tokenizer.decode(top))
```

Pass `hook=True` when the module is one you *attached* to the tree (an SAE, an
adapter) and you want its own internals observable — see
[modules-and-architectures.md](modules-and-architectures.md).

## Reading outside a trace

There is nothing to read outside interleaving; `.output` raises there.

<!-- test: expect-error ValueError -->
```python
hidden = model.transformer.h[0].output     # no trace running
```

Use `model.scan(...)` when you want shapes without running the model, or open a
trace when you want values.

## Name collisions

If a module's own children include a name nnsight uses (BERT's `output`
submodule), the child keeps the name and nnsight's property moves to `.nns_output`
(with a warning at load). Check with `print(model)` if a `.output` returns
something that looks like a module rather than a tensor.

## Related

- [execution-model.md](execution-model.md) — why reads block and what order they must be in
- [batching.md](batching.md) — doing this for several prompts in one pass
- [source-tracing.md](source-tracing.md) — reaching values *inside* a module's forward
- [caching-and-scan.md](caching-and-scan.md) — grabbing many modules at once
