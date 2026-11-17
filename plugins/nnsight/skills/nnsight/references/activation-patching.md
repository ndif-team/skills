# Activation patching

Copy an activation from a run on one prompt into a run on another and measure how
far the answer moves. **Denoising** copies clean into corrupt and asks whether the
activation is enough to restore the answer. **Noising** copies corrupt into clean
and asks whether it is needed. Say which one a number came from.

```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True)
n_layers = len(model.transformer.h)

clean = "The Eiffel Tower is in the city of"        # answer " Paris"
corrupt = "The Colosseum is in the city of"         # answer " Rome"
paris = model.tokenizer.encode(" Paris", add_special_tokens=False)[0]
rome = model.tokenizer.encode(" Rome", add_special_tokens=False)[0]

tokens = [model.tokenizer.decode([i]) for i in model.tokenizer(corrupt).input_ids]
assert len(model.tokenizer(clean).input_ids) == len(tokens) == 10   # positions must correspond
SUBJECT = slice(1, 5)                               # the tokens that differ
```

The two prompts have the same structure and the same token count, and differ only
in the fact under test.

## The metric

```python
with model.trace() as tracer:
    with tracer.invoke(clean):
        logits = model.output.logits[0, -1]
        clean_diff = (logits[paris] - logits[rome]).save()
        clean_raw = logits[paris].save()
    with tracer.invoke(corrupt):
        logits = model.output.logits[0, -1]
        corrupt_diff = (logits[paris] - logits[rome]).save()
        corrupt_raw = logits[paris].save()

def recovery(diff):
    """Fraction of the corrupt-to-clean gap restored: 0 is the corrupt run, 1 the clean run."""
    return (float(diff) - corrupt_diff.item()) / (clean_diff.item() - corrupt_diff.item())

print(f"logit diff  clean {clean_diff.item():+.3f}  corrupt {corrupt_diff.item():+.3f}")
print(f"raw logit   clean {clean_raw.item():+.3f}  corrupt {corrupt_raw.item():+.3f}")
assert clean_raw < corrupt_raw                      # the raw logit points the wrong way
```

```
logit diff  clean +2.425  corrupt -1.487
raw logit   clean -92.087  corrupt -91.957
```

Score every patch as recovery of the logit difference between the two answers.
**Do not normalize a raw logit.** GPT-2's logits carry an offset shared by every
token, and it moves between prompts: `' Paris'` is 21 times likelier on the clean
prompt and its raw logit is lower there. A raw-logit recovery has the wrong sign
and a denominator of 0.13: the sweep below would read +15.6 at layer 0. Do not
judge by the top token either: three tokens outrank `' Rome'` on the corrupt
prompt.

## The sweep and its two controls

Cache the donor activations, then patch every layer in one trace, one invoke per
layer. The same sweep runs three donors: the clean run, the corrupt run itself,
and a random draw of matching norm.

```python
def residuals(text):
    with model.trace(text):
        resid = nnsight.save([block.output.detach() for block in model.transformer.h])
    return resid

clean_resid, corrupt_resid = residuals(clean), residuals(corrupt)

generator = torch.Generator().manual_seed(0)
random_resid = []
for real in clean_resid:
    draw = torch.randn(real.shape, generator=generator).to(real.device)
    random_resid.append(draw / draw.norm(dim=-1, keepdim=True) * real.norm(dim=-1, keepdim=True))

def sweep(base, donor):
    """Recovery after patching SUBJECT at each layer of `base` from `donor`: one forward pass."""
    with model.trace() as tracer:
        diffs = nnsight.save({})
        for layer in range(n_layers):
            with tracer.invoke(base):
                model.transformer.h[layer].output[:, SUBJECT, :] = donor[layer][:, SUBJECT, :]
                logits = model.output.logits[0, -1]
                diffs[layer] = (logits[paris] - logits[rome]).detach()
    return [recovery(diffs[layer]) for layer in range(n_layers)]

denoise = sweep(corrupt, clean_resid)
self_patch = sweep(corrupt, corrupt_resid)
random_patch = sweep(corrupt, random_resid)

for name, row in (("clean", denoise), ("self", self_patch), ("random", random_patch)):
    print(f"{name:<7}" + " ".join(f"{value:+.2f}" for value in row))
assert max(abs(value) for value in self_patch) < 1e-3, "self-patch is not a no-op"
```

```
clean  +1.00 +1.00 +1.02 +0.96 +0.94 +0.93 +0.86 +0.83 +0.56 +0.12 +0.05 +0.00
self   +0.00 +0.00 +0.00 +0.00 +0.00 +0.00 +0.00 +0.00 +0.00 +0.00 +0.00 +0.00
random +0.37 +0.01 -0.01 +1.30 +0.32 +0.34 +0.41 +0.56 -0.19 +0.09 +0.30 +0.00
```

- **Run the self-patch on every sweep.** The corrupt run's own activations must
  leave the metric where it was. It fails when the cache is stale, the donor is
  indexed at the wrong layer or the positions do not line up.
- **The random donor is the null.** It carries no information and still reaches
  1.30 at layer 3, more than the clean donor recovers at layers 8 to 11. An
  effect inside the null is not an effect.
- **The two ends of the curve are architecture, not evidence.** Patching
  `block.output` at layer L overwrites everything computed at those positions up
  to L. The clean donor gives 1 at the first block because the run is the clean
  run from there on, and every donor gives 0 at the last block because a
  non-final position there cannot reach the last position's logits. A decay from
  1 to 0 is what any residual sweep gives. It shows how late the subject
  positions are still read, not where the fact is; the map below localizes.

The donor can also come from the same forward pass as the patched runs, with a
`tracer.barrier` ordering the handoff ([batching](batching.md)). On NDIF the
caching traces and the sweep go in one session ([remote](remote.md)).

## Layer x position map

Patch one (layer, position) cell per invoke:

```python
n_pos = len(tokens)

with model.trace() as tracer:
    cells = nnsight.save({})
    for layer in range(n_layers):
        for pos in range(n_pos):
            with tracer.invoke(corrupt):
                model.transformer.h[layer].output[:, pos, :] = clean_resid[layer][:, pos, :]
                logits = model.output.logits[0, -1]
                cells[layer, pos] = (logits[paris] - logits[rome]).detach()

grid = torch.tensor([[recovery(cells[layer, pos]) for pos in range(n_pos)] for layer in range(n_layers)])

print(f"{'':<9}" + "".join(f"L{layer:<6}" for layer in range(0, n_layers, 2)))
for pos, token in enumerate(tokens):
    print(f"{token!r:<9}" + "".join(f"{grid[layer, pos]:+.2f}  " for layer in range(0, n_layers, 2)))

assert grid[:5, SUBJECT].max() > 0.4 and grid[10:, SUBJECT].max() < 0.1    # the subject, early
assert grid[:7, -1].abs().max() < 0.05 and grid[10, -1] > 0.8              # the last token, late
assert grid[:, 5:9].abs().max() < 0.05                                     # nothing in between
```

```
         L0     L2     L4     L6     L8     L10
'The'    +0.00  +0.00  +0.00  +0.00  +0.00  +0.00
' Col'   +0.13  +0.06  +0.04  +0.02  +0.01  -0.00
'os'     +0.30  +0.51  +0.43  +0.19  +0.12  +0.02
'se'     +0.00  +0.07  +0.11  +0.12  +0.08  +0.02
'um'     +0.34  +0.41  +0.40  +0.37  +0.23  +0.02
' is'    +0.00  +0.00  +0.01  -0.00  +0.00  +0.00
' in'    +0.00  -0.01  +0.01  +0.00  +0.00  +0.01
' the'   +0.00  -0.00  -0.00  +0.00  +0.00  +0.00
' city'  -0.00  -0.02  -0.01  +0.02  +0.01  +0.01
' of'    +0.00  +0.01  +0.00  +0.02  +0.28  +0.87
```

The map separates two sites that the sweep could not: the subject pieces `'os'`
and `'um'` through layer 6, and the final `' of'`, which does nothing until layer
8 and then carries the answer. All 120 patched runs are one batch here; on a
larger model open one trace per layer. Where the answer becomes readable, as
opposed to carried, is the [logit-lens](logit-lens.md) question.

## Heads and sublayers

Head `i` is a slice of the attention output projection's input
(`attn.c_proj.input` on GPT-2, `self_attn.o_proj.input` on Llama); the convention
is in [access-and-modify](access-and-modify.md).

```python
n_heads = model.config.n_head
head_dim = model.transformer.h[0].attn._module.head_dim

with model.trace(clean):
    clean_heads = nnsight.save([block.attn.c_proj.input.detach() for block in model.transformer.h])

with model.trace() as tracer:
    cells = nnsight.save({})
    for layer in range(n_layers):
        for head in range(n_heads):
            with tracer.invoke(corrupt):
                span = slice(head * head_dim, (head + 1) * head_dim)
                model.transformer.h[layer].attn.c_proj.input[:, :, span] = clean_heads[layer][:, :, span]
                logits = model.output.logits[0, -1]
                cells[layer, head] = (logits[paris] - logits[rome]).detach()

heads = torch.tensor([[recovery(cells[layer, head]) for head in range(n_heads)] for layer in range(n_layers)])
for index in heads.flatten().abs().topk(4).indices.tolist():
    layer, head = divmod(index, n_heads)
    print(f"L{layer}H{head}  {heads[layer, head]:+.3f}")
```

```
L9H8  +0.366
L8H11  +0.274
L6H4  +0.184
L10H0  +0.131
```

Rank by absolute value, so that a head that pushes against the answer is kept.
Give a head the same two controls before reading it as a site. A whole sublayer
is patched the same way, at `block.mlp.output` or `block.attn.output[0]`.

## Noising

The same sweep with donor and recipient exchanged, scored as the fraction of the
gap the patch destroys:

```python
damage = [1 - value for value in sweep(clean, corrupt_resid)]
print(" ".join(f"{value:.2f}" for value in damage))
assert damage[0] > 0.9 and damage[-1] < 0.1
```

```
1.00 1.00 1.01 0.99 0.97 1.01 0.94 0.93 0.68 0.25 0.09 0.00
```

It has the same architectural ends, mirrored, and needs the same two controls
with the prompts exchanged. Noising with a mean or a random activation in place
of a paired run is [ablation](ablation.md).

## From one pair to a claim

- **One pair is an anecdote.** Make the claim over a set of pairs, and aggregate
  recovery as a ratio of means, `(mean patched - mean corrupt) / (mean clean -
  mean corrupt)`. A mean of per-pair ratios is dominated by the pairs whose clean
  and corrupt runs nearly tie.
- **Batching pairs.** An invoke takes a list of corrupt prompts and a donor with
  one row per prompt. Every prompt must then have the same token count: left
  padding moves an absolute slice such as `SUBJECT` ([batching](batching.md)).
- **Order of work.** Sweep the residual stream to find where, then patch
  attention and MLP outputs there to find which component. With every head x
  layer x position to cover, screen with
  [attribution-patching](attribution-patching.md) and confirm the survivors here.
- **No single component patches cleanly.** The variable may lie along a direction
  spread over the residual stream. Distributed alignment search learns that
  direction and swaps only along it: [activation-patching-das](activation-patching-das.md).
- Metric, controls and prompt sets for any causal experiment:
  [interp-experiment-design](interp-experiment-design.md).
