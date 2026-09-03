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
already in the graph. The call is a no-op on them: a captured `.output` is already
`requires_grad=True, is_leaf=False`, and calling it changes nothing. Only a leaf
you build yourself needs it — an integrated-gradients baseline, a steering vector
under an optimizer.

A gradient is readable only while the block is open. Afterwards `t.grad` is `None`
again, with nothing but PyTorch's non-leaf warning to say so, so `.save()` what you
want inside.

### When not to use the block

Rules 2 and 3 exist because the block runs interleaved. If you only want to *read*
gradients, `retain_grad()` plus a plain `loss.backward()` after the trace gets
byte-identical numbers under no ordering rule at all:

```python
with model.trace(prompt):
    refs = nnsight.save([])
    for i in range(len(model.transformer.h)):
        hidden = model.transformer.h[i].output
        hidden.retain_grad()                  # forward order, any order
        refs.append(hidden)
    loss = model.output.logits.sum().save()

loss.backward()                               # outside the trace, plain PyTorch
retained = [r.grad[:, -1, :].norm().item() for r in refs]
assert all(x > 0 for x in retained)
```

It materializes and holds every `.grad`, which the interleaved block does not. Use
`with metric.backward():` when you want to edit a gradient mid-pass, or when you
want only a few of them.

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

norms = [g.item() for g in reversed(grads)]
assert len(norms) == n_layers and min(norms) > 0
for i, g in enumerate(norms):
    print(f"layer {i:2d}  ||grad|| = {g:.4f}")
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

assert blocked.item() == 0.0             # nothing flows past the cut
print(blocked.item())
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

assert torch.allclose(g2, g1 * 3, atol=1e-4)
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

**Edit at the point you intercept.** Form does not matter: at the module you just
read, `module.output = new_tensor`, `module.output[:] = new_tensor` and
`module.output += delta` all differentiate, and the loop above uses the in-place
form on purpose. Position matters. Reading a *later* module's output advances the
forward past the earlier one, and an in-place write after that point lands on a
value the model has already consumed:

```python
with model.trace(prompt):
    hidden = model.transformer.h[6].output
    later = model.transformer.h[8].output    # the forward is now past layer 6
    hidden *= 2                              # too late
    logits_late = model.output.logits.save()

with model.trace(prompt):
    logits_plain = model.output.logits.save()

assert torch.equal(logits_late, logits_plain)   # the late write did nothing
```

On a forward-only trace it is a silent no-op — no error, no effect. Add a backward
and the same code raises instead:

<!-- test: expect-error RuntimeError -->
```python
with model.trace(prompt):
    hidden = model.transformer.h[6].output
    later = model.transformer.h[8].output
    hidden *= 2
    with model.output.logits.sum().backward():
        grad = later.grad.norm().save()
# RuntimeError: one of the variables needed for gradient computation has been
# modified by an inplace operation: [torch.cuda.FloatTensor [1, 10, 768]], which
# is output 0 of Mul, is at version 1; expected version 0 instead.
```

Moving `hidden *= 2` above the `h[8]` read fixes both.

## Where gradients are unavailable

**Inside `model.generate()`.** HuggingFace decorates `GenerationMixin.generate`
with `@torch.no_grad()`, so activations captured in a generation trace come back
with `requires_grad=False` and opening a backward block raises
`NotImplementedError: This tensor does not require grad, so a backward session
cannot produce gradients`. `torch.enable_grad()` around the call does not override
the decorator. Drive the decoding yourself instead — a `model.trace` per step over
the growing prefix, with the backward inside each.

**On a frozen model.** After `model.requires_grad_(False)` the same
`NotImplementedError` appears, because no parameter is in the graph. Inject a
tensor that does require grad — a steering vector, an adapter — and gradients exist
from that point downstream; reading `.grad` on an activation *upstream* of the
injection raises `RuntimeError: cannot register a hook on a tensor that doesn't
require gradient`.

**After the metric's path ends.** Asking for the `.grad` of a tensor autograd never
reaches, such as a branch computed off to the side, raises `OutOfOrderError` at the
end of the run rather than hanging.

## Where this leads

- **Attribution patching** — gradients times clean/corrupt activation differences,
  a linear approximation of activation patching that costs 2 passes instead of N.
  See the `attribution-patching` skill.
- **Integrated gradients** — average `input × grad` along a path from a baseline;
  a loop of the pattern above.

## Related

- [access-and-modify.md](access-and-modify.md) — capturing the forward values
- [execution-model.md](execution-model.md) — why ordering matters
