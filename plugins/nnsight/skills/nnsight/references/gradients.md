# Gradients and Backward Passes

`with tensor.backward():` runs a **real backward pass interleaved with your code**,
the same way `model.trace` interleaves the forward. Inside that block you can read
and replace `.grad` on any tensor you captured during the forward.

<!-- test: setup -->
```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)
prompt = "The Eiffel Tower is in the city of"
paris = model.tokenizer.encode(" Paris")[0]
```

## The shape of every gradient experiment

```python
with model.trace(prompt):
    hidden = model.transformer.h[-1].output       # 1. capture during the forward
    metric = model.output.logits[0, -1, paris]    # 2. build a scalar metric

    with metric.backward():                       # 3. backward, interleaved
        grad = hidden.grad.clone().save()         # 4. read grads (reverse order)

print(grad.shape)
```

Four rules, and every gradient bug is one of them:

1. **Capture forward tensors before the backward block.** The forward is over once
   autograd runs; reading `.output` inside the backward block raises
   `OutOfOrderError`.
2. **Read `.grad` in reverse-forward order** — last layer first.
3. **Ask for `.grad` on the exact tensor you captured**, not a slice of it.
   `hidden.grad` works; `hidden[0].grad` is a different tensor and raises.
4. **`.grad` lives on tensors, not modules.** There is no `module.output.grad`
   shortcut that skips capturing the tensor first.

No `requires_grad_(True)` is needed for activations — they are non-leaf tensors
already in the graph.

## Per-layer saliency

The canonical layer-importance sweep. Capture every layer's residual in forward
order, then read gradients in reverse:

```python
n_layers = len(model.transformer.h)

with model.trace(prompt):
    resid = [model.transformer.h[i].output for i in range(n_layers)]   # forward order
    metric = model.output.logits[0, -1, paris]

    with metric.backward():
        grads = nnsight.save([])
        for i in reversed(range(n_layers)):                            # reverse order
            grads.append(resid[i].grad[:, -1, :].norm())

for i, g in enumerate(reversed(grads)):
    print(f"layer {i:2d}  ||grad|| = {g.item():.4f}")
```

## Input × gradient

Attribution that weights each activation by its own gradient — one number per
position, the standard saliency map:

```python
with model.trace(prompt):
    embeds = model.transformer.wte.output
    metric = model.output.logits[0, -1, paris]
    with metric.backward():
        attribution = (embeds * embeds.grad).sum(dim=-1).save()

tokens = [model.tokenizer.decode([i]) for i in model.tokenizer(prompt).input_ids]
for token, score in zip(tokens, attribution[0].tolist()):
    print(f"{token!r:20} {score:+.3f}")
```

## Editing gradients mid-backward

Assigning to `.grad` replaces what flows further back — useful for gradient
surgery, blocking paths, or scaling a component's contribution:

```python
with model.trace(prompt):
    early = model.transformer.h[2].output
    late = model.transformer.h[9].output
    metric = model.output.logits[0, -1, paris]

    with metric.backward():
        late.grad = late.grad * 0        # cut the gradient path at layer 9
        blocked = early.grad.norm().save()

print(blocked.item())                    # 0.0 — nothing flows past the cut
```

## Several backward passes

Pass `retain_graph=True` to every backward but the last:

```python
with model.trace(prompt):
    hidden = model.transformer.h[-1].output
    logits = model.output.logits

    with logits[0, -1, paris].backward(retain_graph=True):
        g1 = hidden.grad.clone().save()
    with (logits[0, -1, paris] * 3).backward():
        g2 = hidden.grad.clone().save()

print(torch.allclose(g2, g1 * 3, atol=1e-4))
```

## Gradients per invoke

Each invoke sees the gradient for its own rows, even though they share one forward
and one backward:

```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        pass
    with tracer.invoke("The Great Wall is in"):
        hidden = model.transformer.h[-1].output
        with model.output.logits.sum().backward():
            grad = hidden.grad.norm().save()

print(round(grad.item(), 3))
```

## Standalone backward, no model

The context manager works on any tensor whose graph is alive:

```python
x = torch.tensor([2.0, 3.0], requires_grad=True)
loss = (x * x).sum()
with loss.backward():
    g = x.grad.save()

print(g)          # tensor([4., 6.])
```

A bare `tensor.backward()` with no `with` block is untouched vanilla PyTorch.

## Gradients through an intervention

Interventions are ordinary tensor ops, so gradients flow through them. This is how
you optimize a steering vector or a soft prompt against a frozen model:

```python
direction = torch.zeros(768, requires_grad=True)
optimizer = torch.optim.SGD([direction], lr=1.0)

for _ in range(3):
    with model.trace(prompt):
        model.transformer.h[6].output[:, -1, :] += direction.to(model.device)
        loss = -model.output.logits[0, -1, paris]
        with loss.backward():
            pass
        tracked = nnsight.save(loss.item())
    optimizer.step()
    optimizer.zero_grad()
    print(f"loss {tracked:.3f}  ||v|| {direction.norm().item():.4f}")
```

The model's own parameters are untouched — only `direction` receives updates.

**When gradients must flow through an intervention, replace instead of writing
in place.** `module.output = new_tensor` hands the model a new object and leaves
the autograd graph intact; `module.output[:] = new_tensor` mutates the tensor
autograd recorded and raises:

```
RuntimeError: one of the variables needed for gradient computation has been
modified by an inplace operation
```

In-place is fine for interventions you are not differentiating through — it is
only a problem on the path to a `backward()`.

## Where this leads

- **Attribution patching** — gradients times clean/corrupt activation differences,
  a linear approximation of activation patching that costs 2 passes instead of N.
  See the `attribution-patching` skill.
- **Integrated gradients** — average `input × grad` along a path from a baseline;
  a loop of the pattern above.

## Related

- [access-and-modify.md](access-and-modify.md) — capturing the forward values
- [execution-model.md](execution-model.md) — why ordering matters
