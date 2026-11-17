# Gradients: Many Layers, Edits, Training Through the Model

[SKILL.md](../SKILL.md) has the basic pattern and its four rules: capture in the
forward, then inside `with metric.backward():` read `.grad` on the exact captured
tensors, last layer first.

```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True)
prompt = "The Eiffel Tower is in the city of"
paris = model.tokenizer.encode(" Paris", add_special_tokens=False)[0]
n_layers = len(model.transformer.h)
```

## When a `.grad` read fails

The error names an object id where a module path would be:

```
OutOfOrderError: '139853884760112.grad.i0' was requested but the model already ran past it
```

It has three causes:

- **gradients read in forward order.** The backward pass reaches the last layer
  first, so read last layer first;
- **`.grad` asked of a slice.** `hidden[0].grad` and `block.output[0].grad` are
  new tensors that never receive a gradient. Ask the tensor you captured, and
  slice the gradient;
- **the tensor is not on the metric's path**: a branch computed off to the side,
  or anything upstream of a value replaced by a fresh tensor
  ([below](#replacements-and-the-graph)).

## Every layer in one backward pass

```python
with model.trace(prompt):
    resid = [block.output for block in model.transformer.h]       # forward order
    metric = model.output.logits[0, -1, paris]
    with metric.backward():
        grads = nnsight.save([])
        for layer in reversed(range(n_layers)):                    # reverse order
            grads.append(resid[layer].grad[:, -1].norm())

norms = [float(g) for g in reversed(grads)]                        # back to layer order
assert len(norms) == n_layers and min(norms) > 0
```

To read gradients with no ordering rule, retain them and call plain
`backward()` after the trace. This holds every `.grad` in memory, which the
interleaved block does not:

```python
with model.trace(prompt):
    refs = nnsight.save([])
    for block in model.transformer.h:
        hidden = block.output
        hidden.retain_grad()
        refs.append(hidden)
    loss = model.output.logits[0, -1, paris].save()

loss.backward()
assert all(float(r.grad[:, -1].norm()) > 0 for r in refs)
model.zero_grad(set_to_none=True)
```

## Editing a gradient, and several backward passes

Assigning to `.grad` replaces what flows further back:

```python
with model.trace(prompt):
    early = model.transformer.h[2].output
    late = model.transformer.h[9].output
    logits = model.output.logits

    with logits[0, -1, paris].backward(retain_graph=True):   # every pass but the last
        late.grad = late.grad * 0                            # cut the path at layer 9
        blocked = early.grad.norm().save()
    with logits[0, -1, paris].backward():
        full = early.grad.norm().save()

assert blocked.item() == 0.0 and full.item() > 0
```

In a batched trace each invoke reads the gradient of its own rows. The block
works on any tensor with a live graph, no model needed
(`with loss.backward(): g = x.grad.save()`), and a bare `tensor.backward()` with
no `with` is plain PyTorch.

## Training a tensor through the model

An intervention is ordinary tensor arithmetic, so gradients flow through it to
anything you add. Freeze the model first: otherwise the backward fills `.grad`
on every weight.

```python
model.requires_grad_(False)
direction = torch.zeros(768, device=model.device, requires_grad=True)
optimizer = torch.optim.SGD([direction], lr=1.0)

losses = []
for _ in range(3):
    with model.trace(prompt):
        model.transformer.h[6].output[:, -1, :] += direction
        loss = -model.output.logits[0, -1, paris]
        with loss.backward():
            pass
        value = nnsight.save(loss.item())
    optimizer.step()
    optimizer.zero_grad()
    losses.append(value)

assert losses[0] > losses[-1] and direction.norm() > 0
model.requires_grad_(True)
```

**Replace, do not write through, when the new value is computed from the old.**
`module.output = f(module.output)` and `module.output += delta` differentiate.
`module.output[:] = f(module.output)` raises `RuntimeError: one of the variables
needed for gradient computation has been modified by an inplace operation`
whenever `f`'s backward needs the tensor being overwritten, as a LoRA's
`hidden + lora(hidden)` does.

**Edit at the point you intercept.** An in-place write to a tensor after a later
module was read is a silent no-op on a forward-only trace
([SKILL.md](../SKILL.md), trap 2). With a backward pass the same code raises that
`RuntimeError` instead, naming `version 1; expected version 0`.

## Replacements and the graph

A replacement has to descend from the value it replaces if gradients are to pass
it. `output = torch.zeros_like(output)` hands the model a tensor autograd has
never seen:

- replace a whole block's `.output` and a later `.grad` read upstream fails with
  the object-id `OutOfOrderError` above;
- replace a submodule the residual stream routes around (an MLP, an attention)
  and upstream `.grad` reads **succeed**, silently missing that module's path.
  A detached copy of the same values does this too: the forward is bit-identical
  and the gradient is wrong.

`output[:] = 0` writes through the existing tensor and keeps the graph. So does
any replacement built from the old value (`output * 0`, `output + steering`).

```python
def upstream_grad(replace):
    with model.trace(prompt):
        early = model.transformer.h[2].output
        mlp = model.transformer.h[5].mlp
        mlp.output = replace(mlp.output)
        with model.output.logits[0, -1, paris].backward():
            grad = early.grad.clone().save()
    return grad

kept = upstream_grad(lambda out: out * 1.0)
cut = upstream_grad(lambda out: out.detach().clone())
assert not torch.allclose(kept, cut)          # same forward, different gradient
```

## Where gradients are unavailable

- **Inside `model.generate`.** HuggingFace runs `generate` under
  `torch.no_grad()`, so a backward block raises `NotImplementedError: This tensor
  does not require grad, so a backward session cannot produce gradients`. Drive
  the decoding yourself: one `model.trace` per step over the growing prefix.
- **On a frozen model with nothing trainable in the path.** The same
  `NotImplementedError`. Once you inject a tensor that requires grad, gradients
  exist from that point downstream; `.grad` on an activation upstream of it
  raises `RuntimeError: cannot register a hook on a tensor that doesn't require
  gradient`.
- **Outside the backward block.** `.grad` is `None` there, with only PyTorch's
  non-leaf warning to say so.
- **On an `int8`-quantized model with a large loss**, gradients overflow to NaN:
  [quantization](quantization.md).

Gradient-times-difference attribution over a clean and a corrupt run is the
[attribution-patching guide](attribution-patching.md).
