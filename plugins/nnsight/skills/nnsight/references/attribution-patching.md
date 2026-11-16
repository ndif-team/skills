# Attribution patching

Attribution patching estimates what activation patching would measure, for every
component at once, from a first-order approximation taken at the corrupt run:

```
effect of patching a  ~  (a_clean - a_corrupt) * d metric / d a
```

Two forward passes and one backward pass give every score, where
[activation-patching](activation-patching.md) costs a forward pass per component.
It is a screen: rank with it, check it against real patching on your task as
below, and confirm the top candidates with real patching.

```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True)
n_layers = len(model.transformer.h)

clean = "The Eiffel Tower is in the city of"        # answer " Paris"
corrupt = "The Colosseum is in the city of"         # answer " Rome", same token count
paris = model.tokenizer.encode(" Paris", add_special_tokens=False)[0]
rome = model.tokenizer.encode(" Rome", add_special_tokens=False)[0]

def attribute(site):
    """(clean - corrupt) * gradient at site(block), for every block."""
    with torch.no_grad():
        with model.trace(clean):
            clean_acts = nnsight.save([site(block) for block in model.transformer.h])
    with model.trace(corrupt):                      # not under no_grad: this pass needs its graph
        refs = [site(block) for block in model.transformer.h]
        corrupt_acts = nnsight.save([ref.detach() for ref in refs])
        logits = model.output.logits[0, -1]
        with (logits[paris] - logits[rome]).backward():
            grads = nnsight.save({})
            for layer in reversed(range(n_layers)):
                grads[layer] = refs[layer].grad.clone()
    scores = torch.stack([(clean_acts[i] - corrupt_acts[i]) * grads[i] for i in range(n_layers)])
    return scores[:, 0].cpu(), clean_acts

resid_scores, clean_resid = attribute(lambda block: block.output)   # (layer, position, hidden)
by_position = resid_scores.sum(-1)                                   # the layer x position map
assert tuple(by_position.shape) == (n_layers, 10)
```

Use a logit difference as the metric. With `P(' Paris')` the estimate for the
last position at the last block is six times too small (0.011 against a real
0.067): a probability near 0 or 1 has almost no gradient.

## Validate before reading a map

Run real patching on the same cells and correlate. `real_effect` takes the cells
and a function that writes one patch, and runs them as one batch with an
unpatched baseline:

```python
def real_effect(cells, write):
    """Change in the metric that write(cell) causes: one invoke per cell, one forward pass."""
    with model.trace() as tracer:
        out = nnsight.save({})
        for cell in [None] + cells:                 # None is the unpatched corrupt run
            with tracer.invoke(corrupt):
                if cell is not None:
                    write(cell)
                logits = model.output.logits[0, -1]
                out[cell] = (logits[paris] - logits[rome]).detach()
    return torch.stack([out[cell] - out[None] for cell in cells]).cpu()

def agreement(approx, real):
    """Pearson r between attribution and real patching over the same cells."""
    assert real.max() - real.min() > 1e-3 * real.abs().max(), "degenerate sweep: every cell returns the same metric"
    return torch.corrcoef(torch.stack([approx, real]))[0, 1].item()

def patch_resid(positions):
    def write(layer):
        model.transformer.h[layer].output[:, positions] = clean_resid[layer][:, positions]
    return write

layers = list(range(n_layers))
LAST, SUBJECT = slice(-1, None), slice(1, 5)        # ' of', and the four tokens that differ

for name, positions in (("last position", LAST), ("subject span", SUBJECT)):
    approx = resid_scores[:, positions].sum((1, 2))
    real = real_effect(layers, patch_resid(positions))
    print(f"{name:<14} r {agreement(approx, real):+.3f}   top-3 layers: "
          f"attribution {approx.topk(3).indices.tolist()}  real {real.topk(3).indices.tolist()}")
```

```
last position  r +0.999   top-3 layers: attribution [11, 10, 9]  real [11, 10, 9]
subject span   r +0.168   top-3 layers: attribution [7, 6, 5]  real [2, 0, 1]
```

**Residual scores are right where the two prompts share the token and wrong where
they differ.** At the last position the two runs' activations differ by 4 to 29
percent of their norm and the linearization is nearly exact. At the subject
tokens they differ by 60 to 100 percent, which no first-order term describes: the
two top-3 lists share no layer, and a single subject token does no better (r of
+0.29 at `'um'`, -0.01 at `'os'`). Patch those positions for real.

**The assert in `agreement` is what catches a degenerate sweep.** Patch every
position of a layer (`slice(None)`) and everything downstream is the clean run,
so all twelve layers return the same number. A correlation against that is
rounding error, and without the assert it still prints a coefficient.

## Heads

The same two passes at the attention output projection's input, where head `i` is
a slice ([access-and-modify](access-and-modify.md)):

```python
n_heads = model.config.n_head
head_dim = model.transformer.h[0].attn._module.head_dim

proj_scores, clean_heads = attribute(lambda block: block.attn.c_proj.input)
head_scores = proj_scores.unflatten(-1, (n_heads, head_dim)).sum((1, 3))    # (layer, head)

def patch_head(cell):
    layer, head = cell
    span = slice(head * head_dim, (head + 1) * head_dim)
    model.transformer.h[layer].attn.c_proj.input[:, :, span] = clean_heads[layer][:, :, span]

cells = [(layer, head) for layer in range(n_layers) for head in range(n_heads)]
real_heads = real_effect(cells, patch_head)

print(f"144 heads  r {agreement(head_scores.flatten(), real_heads):+.3f}")
for index in head_scores.flatten().abs().topk(4).indices.tolist():
    layer, head = cells[index]
    print(f"{f'L{layer}H{head}':<7}attribution {head_scores[layer, head]:+.3f}   real {real_heads[index]:+.3f}")
```

```
144 heads  r +0.989
L9H8   attribution +1.240   real +1.431
L8H11  attribution +1.023   real +1.071
L6H4   attribution +0.621   real +0.721
L10H0  attribution +0.477   real +0.513
```

Head scores hold up here: the top four are the four that real patching finds, in
the same order. Rank by absolute value, because a head that pushes against the
answer matters as much as one that pushes for it, and keep the sign when you
confirm. On a model where the full real sweep costs too much, pass
`real_effect` the top-k cells and a random sample of the rest. Report the
validation `r` beside any attribution map.

[circuit-discovery](circuit-discovery.md) builds its head ranking on this block
and extends the same product to connections between components (edge
attribution).
