---
name: attribution-patching
description: Approximate activation patching with gradients — attribution = (clean activation − corrupt activation) · gradient of the metric — so a whole layer×position×head map costs two forward passes and one backward instead of one forward pass per component. Use to screen large models or large component sets before verifying the survivors with real patching, to build circuit-level attribution heatmaps, and for edge attribution patching. Includes the validation step that tells you whether the approximation is trustworthy on your task, and the conditions under which it silently is not.
---

# Attribution Patching

Activation patching costs one forward pass per component. For every (layer,
position, head) of a real model that is intractable. Attribution patching replaces
the measurement with its first-order Taylor approximation:

```
effect of patching component a  ≈  (a_clean − a_corrupt) · ∇_a metric
```

Both terms come from **two forward passes and one backward** — total, for every
component at once. It is an approximation, so the workflow is always: attribute
everything cheaply, then verify the top candidates with real patching.

<!-- test: setup -->
```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

clean = "The Eiffel Tower is in the city of"       # → " Paris"
corrupt = "The Colosseum is in the city of"        # → " Rome"

paris = model.tokenizer.encode(" Paris")[0]
rome = model.tokenizer.encode(" Rome")[0]
n_layers = len(model.transformer.h)
assert len(model.tokenizer(clean).input_ids) == len(model.tokenizer(corrupt).input_ids)
```

## Canonical implementation

```python
# Pass 1 — clean activations. No gradients needed.
with torch.no_grad():
    with model.trace(clean):
        clean_acts = nnsight.save([block.output for block in model.transformer.h])

# Pass 2 — corrupt forward, then backward through the metric.
with model.trace(corrupt):
    refs = [block.output for block in model.transformer.h]          # forward order
    corrupt_acts = nnsight.save([h.detach() for h in refs])

    logits = model.output.logits[0, -1]
    metric = logits[paris] - logits[rome]

    with metric.backward():
        grads = nnsight.save([])
        for layer in reversed(range(n_layers)):                     # reverse order
            grads.append(refs[layer].grad.clone())

grads = grads[::-1]                                                 # back to layer order

attribution = [
    float(((clean_acts[i] - corrupt_acts[i]) * grads[i]).sum())
    for i in range(n_layers)
]

assert len(attribution) == n_layers and max(attribution) > 1.0

for layer, score in enumerate(attribution):
    print(f"layer {layer:2d}  attribution {score:+.4f}")
```

Three ordering rules, all inherited from the `nnsight` execution model:

- capture activations in **forward** order in the corrupt trace
- read `.grad` in **reverse** order inside `with metric.backward():`
- ask for `.grad` on the tensor you captured, not a slice of it

No `requires_grad_()` is needed — activations are already in the graph. Do not wrap
the corrupt pass in `torch.no_grad()`.

## Validate before you trust it

The approximation is only useful if it ranks components the way real patching
does. You already have both halves, so measure the agreement before you believe a
heatmap. Patch one layer at a time — a *slice* of the residual, not all of it —
and correlate the real effect against the attribution restricted to that same
slice:

```python
def validate(positions, name):
    approx = torch.tensor([
        float(((clean_acts[i] - corrupt_acts[i]) * grads[i])[:, positions].sum())
        for i in range(n_layers)
    ])

    # Keyed by layer, not appended: values come back in the order the model
    # reaches the invokes, which is not necessarily the order you wrote them.
    with model.trace() as tracer:
        real = nnsight.save({})
        for layer in range(n_layers):
            with tracer.invoke(corrupt):
                model.transformer.h[layer].output[:, positions] = clean_acts[layer][:, positions]
                patched = model.output.logits[0, -1]
                real[layer] = (patched[paris] - patched[rome]).detach()

    real_effects = torch.tensor([float(real[i]) for i in range(n_layers)])

    # Layers that all return the same metric leave nothing to correlate against.
    assert real_effects.std() > 0.5, f"{name}: degenerate sweep, std {real_effects.std():.2e}"

    centered_a = approx - approx.mean()
    centered_r = real_effects - real_effects.mean()
    r = float((centered_a @ centered_r) / (centered_a.norm() * centered_r.norm()))
    print(f"{name:<20} real-effect spread {real_effects.std():.4f}   Pearson r {r:+.3f}")
    return approx, real_effects, r


SUBJECT = slice(1, 5)        # ' Col', 'os', 'se', 'um' — the corrupt prompt's subject
LAST = slice(-1, None)

approx, real_effects, r_subject = validate(SUBJECT, "subject positions")
_, _, r_last = validate(LAST, "last position")

print(f"top-3 by attribution: {torch.topk(approx, 3).indices.tolist()}")
print(f"top-3 by real patch:  {torch.topk(real_effects, 3).indices.tolist()}")
assert r_last > 0.99, f"last-position linearization no longer near-exact: {r_last:+.3f}"
assert abs(r_subject) < 0.5, f"subject-slice agreement left the weak band: {r_subject:+.3f}"
```

On this prompt pair that prints:

```
subject positions    real-effect spread 1.5652   Pearson r +0.168
last position        real-effect spread 1.4942   Pearson r +0.999
top-3 by attribution: [7, 6, 5]
top-3 by real patch:  [2, 0, 1]
```

Two sizes of intervention, two verdicts. At one position the linearization is
nearly exact and the rankings agree. Across the four subject tokens it falls to
`+0.168`, and the top-3 lists share no layer at all — swapping four positions of
residual at once moves the run well outside the regime a first-order term
describes.

| What is patched | Pearson r vs real patching |
|---|---|
| the last position only | **+0.999** |
| four subject-token positions | +0.168 |

The rule that falls out: **attribution patching is trustworthy for small, local
interventions and decays as the intervention grows.** Patch a position, a head,
or a feature. And run this check on your own task before believing a heatmap; it
costs one extra sweep and it is the difference between a screening tool and a
random number generator.

**Why the sweep asserts a spread.** Patch a layer's *entire* residual output and
the sweep stops measuring anything. Overwriting all of layer L with the clean
activation makes everything after it a deterministic function of a clean state,
so every layer returns exactly the clean metric — `2.4247` at all twelve layers
here, with a spread of `1.2e-05`. Correlating attribution against that vector is
rounding error over rounding error, and it prints a confident-looking number
either way. The assertion is what makes that failure loud instead of publishable.

## Per-position heatmap

Keep the position axis instead of summing it — the same two passes give a
`[layer, position]` map:

```python
heatmap = torch.stack([
    ((clean_acts[i] - corrupt_acts[i]) * grads[i]).sum(dim=-1)[0]
    for i in range(n_layers)
])

tokens = [model.tokenizer.decode([i]) for i in model.tokenizer(corrupt).input_ids]
assert heatmap.shape == (n_layers, len(tokens))

print(f"{'token':<12}" + "".join(f"L{l:<7}" for l in range(0, 12, 3)))
for pos, token in enumerate(tokens):
    row = "".join(f"{heatmap[l, pos]:+.3f} " for l in range(0, 12, 3))
    print(f"{token!r:<12}{row}")
```

## Per-head attribution

Attribute at the attention output projection's input, where heads are still
separate slices — this is the cheap version of a head-level circuit scan:

```python
n_heads = model.config.n_head
head_dim = model.config.n_embd // n_heads

with torch.no_grad():
    with model.trace(clean):
        clean_heads = nnsight.save([
            model.transformer.h[i].attn.c_proj.input for i in range(n_layers)
        ])

with model.trace(corrupt):
    head_refs = [model.transformer.h[i].attn.c_proj.input for i in range(n_layers)]
    corrupt_heads = nnsight.save([h.detach() for h in head_refs])
    logits = model.output.logits[0, -1]
    with (logits[paris] - logits[rome]).backward():
        head_grads = nnsight.save([])
        for i in reversed(range(n_layers)):
            head_grads.append(head_refs[i].grad.clone())

head_grads = head_grads[::-1]

scores = torch.zeros(n_layers, n_heads)
for layer in range(n_layers):
    delta = (clean_heads[layer] - corrupt_heads[layer]) * head_grads[layer]
    for head in range(n_heads):
        lo, hi = head * head_dim, (head + 1) * head_dim
        scores[layer, head] = delta[..., lo:hi].sum()

assert scores.shape == (n_layers, n_heads) and scores.abs().max() > 0

flat = scores.flatten().abs().topk(5).indices
for index in flat.tolist():
    layer, head = divmod(index, n_heads)
    print(f"L{layer}H{head:<2} attribution {scores[layer, head]:+.4f}")
```

144 head attributions from two forward passes. Verify the top few with real head
patching (`activation-patching` skill) before believing any of them.

## Edge attribution

The same trick applied to *connections* rather than components: attribute the
effect of the path from an upstream component to a downstream one by taking the
gradient at the downstream input and the activation difference at the upstream
output. That is the basis of edge attribution patching and automated circuit
discovery — see the `circuit-discovery` skill.

## When the approximation breaks

| Condition | Effect |
|---|---|
| Large intervention (many positions at once, or a whole layer) | agreement decays toward chance — `+0.999` at one position against `+0.168` across four, measured above. Keep interventions local |
| Saturated metric (softmax probability near 0 or 1) | gradients vanish; everything scores ~0. Use a logit difference, not a probability |
| Components whose effect is gated or thresholded | attribution can be near zero for a component that fully controls the output |
| Very deep interactions | error compounds across layers; treat late-layer scores more cautiously |

Two habits that keep it honest: always report *validated* attribution (correlation
against real patching on a subset), and always state that a heatmap is
approximate. Integrated gradients — averaging the gradient along a path from
corrupt to clean rather than taking it at one point — is the standard upgrade when
the linearization is poor, at the cost of N backward passes.

## Cost comparison

| Approach | Passes for L layers × P positions × H heads |
|---|---|
| Activation patching, naive | one forward per component |
| Activation patching, batched into invokes | one forward per layer |
| Attribution patching | 2 forward + 1 backward, total |
| Attribution + verification of top-K | 2 forward + 1 backward + K |

The last row is the recommended workflow.

## Related skills

- `activation-patching` — the ground truth this approximates, and the verification step
- `circuit-discovery` — edge attribution and automated circuit search
- `nnsight` — gradients, ordering rules, batching
- `nnsight-remote` — running the two passes on a model you cannot host
