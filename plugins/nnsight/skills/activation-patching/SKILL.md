---
name: activation-patching
description: Establish which components causally carry a behavior by copying activations from one run into another — test specific components on a clean/corrupt prompt pair, by denoising (clean into corrupt) and noising (corrupt into clean), across layers, positions, attention heads, and MLPs. Use to localize where a fact or capability lives, to confirm that a component found by a probe or attention pattern is actually causal, to build layer×position causal maps, and for IOI-style paired-prompt designs. Includes distributed alignment search (learned subspace patching) and the metric and prompt-design mistakes that produce confident wrong answers.
---

# Activation Patching

Patching answers a causal question that observation cannot: **if this activation
had been different, would the answer change?** Take a **clean** prompt that gives
the correct answer and a **corrupt** one with the same structure and a different
answer, copy one activation from one run into the other, and measure the shift.
**Denoising** (clean → corrupt) asks whether the activation *suffices* to restore
the answer; **noising** (corrupt → clean) asks whether it is *necessary*. A
component that is one but not the other is telling you about redundancy.

<!-- test: setup -->
```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)
n_layers = len(model.transformer.h)

clean = "The Eiffel Tower is in the city of"      # → " Paris"
corrupt = "The Colosseum is in the city of"       # 10 tokens, same as clean
# GPT-2 small is near-degenerate here: its top-5 on the corrupt prompt spans 0.3
# logits and " Rome" is only rank 3. Judge by the logit difference below, not by
# the top token, or float noise between runs will flip the argmax.

paris = model.tokenizer.encode(" Paris")[0]
rome = model.tokenizer.encode(" Rome")[0]

# Paired prompts must tokenize to the same length or positions won't correspond.
assert len(model.tokenizer(clean).input_ids) == len(model.tokenizer(corrupt).input_ids)
SUBJECT = slice(1, 5)          # the tokens that differ
```

## The metric: a normalized logit difference

`P(answer)` alone confounds "the right answer got likelier" with "everything got
likelier". The logit difference between the clean and corrupt answers isolates the
effect and is roughly linear. Its two values define the scale every result below
is reported on: **recovery** = (patched − corrupt) / (clean − corrupt), 0 at the
corrupt run and 1 at the clean one.

```python
with model.trace() as tracer:
    with tracer.invoke(clean):
        logits = model.output.logits[0, -1]
        clean_diff = (logits[paris] - logits[rome]).save()
        clean_paris = logits[paris].save()
    with tracer.invoke(corrupt):
        logits = model.output.logits[0, -1]
        corrupt_diff = (logits[paris] - logits[rome]).save()
        corrupt_paris = logits[paris].save()

def recovery(diff):
    """Fraction of the corrupt → clean gap restored: 0 is corrupt, 1 is clean."""
    return (float(diff) - corrupt_diff.item()) / (clean_diff.item() - corrupt_diff.item())

print(f"logit diff    clean {clean_diff.item():+.3f}   corrupt {corrupt_diff.item():+.3f}")
print(f"raw ' Paris'  clean {clean_paris.item():+.3f}  corrupt {corrupt_paris.item():+.3f}")
assert clean_paris.item() < corrupt_paris.item()     # the raw logit points the wrong way
```

```
logit diff    clean +2.425   corrupt -1.487
raw ' Paris'  clean -92.087  corrupt -91.957
```

**Never compute recovery on a raw logit.** GPT-2's logits carry a large offset
shared by every token, and it moves between prompts — the mean logit is `-106.9`
on the clean prompt and `-103.2` on the corrupt one. `" Paris"` is 21× likelier on
the clean prompt, yet its raw logit is *lower* there, so a raw-logit recovery has
the wrong sign and a near-zero denominator that turns every patch into a number
like `+15`. Use a logit difference, a log-probability or a probability, normalized
as above.

## Canonical sweep — two forward passes for the whole layer scan

Cache the clean activations once, then patch every layer in one batched trace —
one forward pass for the cache and one for the whole sweep. Reach for this by
default.

```python
with model.trace(clean):
    clean_resid = nnsight.save([block.output.detach() for block in model.transformer.h])

with model.trace() as tracer:
    scores = nnsight.save([])
    for layer in range(n_layers):
        with tracer.invoke(corrupt):
            model.transformer.h[layer].output[:, SUBJECT, :] = clean_resid[layer][:, SUBJECT, :]
            logits = model.output.logits[0, -1]
            scores.append((logits[paris] - logits[rome]).detach())

print("recovery by layer:", " ".join(f"{recovery(score):+.2f}" for score in scores))
```

```
recovery by layer: +1.00 +1.00 +1.01 +0.96 +0.94 +0.93 +0.86 +0.83 +0.56 +0.12 +0.05 +0.00
```

A monotone decay like this one is what a residual sweep produces whether or not
the information is early, so read it as a **reachability** curve rather than a
localization. Patching `block.output` at layer L overwrites everything the model
computed at those positions up to and including L, so **both ends are fixed by the
architecture**: patch at the embeddings and you reproduce the clean run exactly,
and patch a non-final position at the last block and it cannot reach the last
position's logits at all. Neither end is evidence, and neither is a check that the
sweep is wired correctly — both hold for any donor, garbage included.

## Controls for a patching sweep

Two donors through the same sweep. The corrupt run's own activations must not move
the metric; a matched-norm random draw is the null an effect has to beat:

```python
with model.trace(corrupt):
    corrupt_resid = nnsight.save([block.output.detach() for block in model.transformer.h])

generator = torch.Generator().manual_seed(0)
random_donor = []
for layer in range(n_layers):
    real = clean_resid[layer]
    draw = torch.randn(real.shape, generator=generator).to(real.device)
    random_donor.append(draw / draw.norm(dim=-1, keepdim=True) * real.norm(dim=-1, keepdim=True))

def sweep(donor):
    with model.trace() as tracer:
        out = nnsight.save([])
        for layer in range(n_layers):
            with tracer.invoke(corrupt):
                model.transformer.h[layer].output[:, SUBJECT, :] = donor[layer][:, SUBJECT, :]
                logits = model.output.logits[0, -1]
                out.append((logits[paris] - logits[rome]).detach())
    return [recovery(x) for x in out]

self_patch = sweep(corrupt_resid)
random_patch = sweep(random_donor)

assert max(abs(v) for v in self_patch) < 1e-3, "self-patch is not a no-op"
print("self:  ", " ".join(f"{v:+.2f}" for v in self_patch))
print("random:", " ".join(f"{v:+.2f}" for v in random_patch))
```

```
self:   +0.00 +0.00 +0.00 +0.00 +0.00 +0.00 +0.00 +0.00 +0.00 +0.00 +0.00 +0.00
random: +0.37 +0.01 -0.01 +1.30 +0.32 +0.34 +0.41 +0.56 -0.19 +0.09 +0.30 +0.00
```

**self** is the control worth running every time. It fails loudly when the cache
is stale, the donor is indexed at the wrong layer, or the positions do not line
up, which the sweep's two ends never will.

**random** carries no information and still swings recovery to `1.3` at layer 3,
further than the real effect at layers 8–11. An effect smaller than the null is
not an effect.

The sweep is a first look at depth. The object that localizes is the
layer × position map below.

## The same sweep in one trace, with a barrier

When the donor has to come from the same forward pass — one request on a remote
model, or no room to hold a cache — put the clean run, one corrupt invoke per
layer, and an unpatched baseline in a single trace. The clean invoke binds each
layer's donor and then calls the barrier; every patching invoke calls the barrier
once per layer too, so all `n_layers + 1` participants meet once per layer and no
read happens before its write. The baseline touches nothing shared and stays out:

```python
with model.trace() as tracer:
    barrier = tracer.barrier(n_layers + 1)
    one_trace = nnsight.save({})

    with tracer.invoke(clean):
        donor = {}
        for layer in range(n_layers):
            donor[layer] = model.transformer.h[layer].output[:, SUBJECT, :]
            barrier()                               # donor[layer] is bound

    for target in range(n_layers):
        with tracer.invoke(corrupt):
            for layer in range(n_layers):
                barrier()                           # wait for the clean read
                if layer == target:
                    model.transformer.h[layer].output[:, SUBJECT, :] = donor[layer]
            logits = model.output.logits[0, -1]
            one_trace[target] = (logits[paris] - logits[rome]).detach()

    with tracer.invoke(corrupt):
        logits = model.output.logits[0, -1]
        baseline = (logits[paris] - logits[rome]).save()

assert abs(baseline.item() - corrupt_diff.item()) < 1e-3
assert all(abs(one_trace[layer].item() - scores[layer].item()) < 1e-3 for layer in range(n_layers))
```

It matches the cached sweep to every printed digit. Drop the barrier, or call it
only at the layer an invoke patches, and that invoke reads `donor[layer]` before
the clean invoke has bound it — a `KeyError`. See the `nnsight` skill → batching.

## Layer × position map

This is the object that localizes, and it is barely more expensive than the sweep:
patch one (layer, position) cell at a time, one forward pass per layer.

```python
n_pos = len(model.tokenizer(clean).input_ids)
tokens = [model.tokenizer.decode([i]) for i in model.tokenizer(corrupt).input_ids]

grid = []
for layer in range(0, n_layers, 2):
    with model.trace() as tracer:
        row = nnsight.save([])
        for pos in range(n_pos):
            with tracer.invoke(corrupt):
                model.transformer.h[layer].output[:, pos, :] = clean_resid[layer][:, pos, :]
                logits = model.output.logits[0, -1]
                row.append((logits[paris] - logits[rome]).detach())
    grid.append([recovery(x) for x in row])

print(f"{'position':<12}" + "".join(f"L{l:<6}" for l in range(0, n_layers, 2)))
for pos, token in enumerate(tokens):
    print(f"{token!r:<12}" + "".join(f"{grid[i][pos]:+.2f}  " for i in range(len(grid))))

early_subject = max(grid[i][pos] for i in range(3) for pos in range(1, 5))    # L0-L4
late_subject = max(grid[i][pos] for i in range(4, 6) for pos in range(1, 5))  # L8-L10
assert early_subject > 0.4 and late_subject < 0.25   # subject matters early, not late
suffix_cells = [grid[i][pos] for i in range(len(grid)) for pos in range(5, 9)]
assert all(abs(v) < 0.04 for v in suffix_cells)      # ' is'..' city' flat everywhere
assert all(abs(grid[i][-1]) < 0.04 for i in range(4))  # ' of' does nothing through L6
assert grid[-1][-1] > 0.8                            # then carries the answer
```

```
position    L0     L2     L4     L6     L8     L10
'The'       +0.00  +0.00  +0.00  +0.00  +0.00  +0.00
' Col'      +0.13  +0.06  +0.04  +0.02  +0.01  -0.00
'os'        +0.30  +0.51  +0.43  +0.19  +0.12  +0.02
'se'        +0.00  +0.07  +0.11  +0.12  +0.08  +0.02
'um'        +0.34  +0.41  +0.40  +0.37  +0.23  +0.02
' is'       +0.00  +0.00  +0.01  -0.00  +0.00  +0.00
' in'       +0.00  -0.01  +0.01  +0.00  +0.00  +0.01
' the'      +0.00  -0.00  -0.00  +0.00  +0.00  +0.00
' city'     -0.00  -0.02  -0.01  +0.02  +0.01  +0.01
' of'       +0.00  +0.01  +0.00  +0.02  +0.28  +0.87
```

Two sites, and the map separates them where the sweep could not: subject pieces
`'os'` and `'um'` move the metric at layers 0–4 and fade by layer 8, and the final
`' of'` position does nothing until layer 8 and then carries most of the answer.
The suffix positions are flat everywhere.

Combine with the `logit-lens` skill: the lens says where the answer becomes
*readable*, patching says where it is *carried*.

## Patching attention heads

Heads are slices of the attention output's hidden dimension. Patch at the output
projection's input, which is the per-head result before it is summed:

```python
n_heads = model.config.n_head
head_dim = model.transformer.h[0].attn._module.head_dim   # equals n_embd // n_head on GPT-2, not on GQA/Gemma models
LAYER = 5

with model.trace(clean):
    clean_heads = model.transformer.h[LAYER].attn.c_proj.input.detach().save()

with model.trace() as tracer:
    head_scores = nnsight.save([])
    for head in range(n_heads):
        with tracer.invoke(corrupt):
            lo, hi = head * head_dim, (head + 1) * head_dim
            model.transformer.h[LAYER].attn.c_proj.input[:, :, lo:hi] = clean_heads[:, :, lo:hi]
            logits = model.output.logits[0, -1]
            head_scores.append((logits[paris] - logits[rome]).detach())

head_recovery = [recovery(score) for score in head_scores]
for start in (0, 6):
    print("   ".join(f"L{LAYER}H{h} {head_recovery[h]:+.3f}" for h in range(start, start + 6)))
assert all(abs(r) < 0.05 for r in head_recovery)    # every head sits at the corrupt baseline
```

```
L5H0 +0.002   L5H1 +0.000   L5H2 +0.005   L5H3 -0.001   L5H4 -0.004   L5H5 -0.002
L5H6 +0.039   L5H7 -0.005   L5H8 -0.002   L5H9 +0.011   L5H10 +0.016   L5H11 -0.020
```

All twelve sit within 4% of the corrupt baseline: no single head at layer 5
carries this fact, and the largest deviation (`L5H6`) is far inside the
random-donor null. A flat column is a result. Sweep layers before reading anything
into one.

Patch MLPs the same way through `model.transformer.h[layer].mlp.output`.

## Noising: the necessity direction

Flip the donor and the recipient — corrupt activations into the clean run, scored
as the fraction of the clean → corrupt gap the patch causes:

```python
with model.trace() as tracer:
    necessity = nnsight.save([])
    for layer in range(0, n_layers, 3):
        with tracer.invoke(clean):
            model.transformer.h[layer].output[:, SUBJECT, :] = corrupt_resid[layer][:, SUBJECT, :]
            logits = model.output.logits[0, -1]
            necessity.append((logits[paris] - logits[rome]).detach())

for i, layer in enumerate(range(0, n_layers, 3)):
    print(f"layer {layer:2d}  damage {1 - recovery(necessity[i]):.3f}")
```

```
layer  0  damage 0.997
layer  3  damage 0.993
layer  6  damage 0.943
layer  9  damage 0.250
```

A large damage means that activation was necessary — and this direction inherits
the same reachability floor as the denoising sweep, mirrored: an early-layer
noising run has the whole model left to propagate the damage, a late-layer one
does not. Give it the same controls, with the roles of clean and corrupt swapped.

Noising with *random* or *mean* activations instead of a paired corrupt run is
ablation — see the `ablation` skill.

## Distributed alignment search

When no single component patches cleanly because the variable lives in a
direction spread across the residual stream, learn that direction and patch only
along it. DAS is easy to over-read — an optimizer finds directions that push the
output whether or not the model uses them — so the training code and the three
nulls it needs are in [references/das.md](references/das.md).

## Designing the experiment

**Prompt pairs.** Same token length (the setup's assert), same structure,
differing only in the fact under test. In a batched sweep every invoke's
activations are padded to the batch's longest prompt, so an absolute position
slice like `slice(1, 5)` shifts with the batch's contents while `[:, -1, :]` does
not. Results appended from the same module in every invoke still come back in
invoke order on a mixed-length batch; it is the position slices that break. See
the `nnsight` skill on invokes and batching.

**Which activation.** The residual stream (`block.output`) tells you *where*
information is; attention output tells you *what moved it*; MLP output tells you
*what computed it*. Sweep the residual first, then drill in.

**Direction of copy.** Denoising and noising answer different questions. State
which one a number came from.

**Controls that can fail.** The self-patch and the random donor above, on every
sweep you report.

**One pair is an anecdote.** Layer indices move across paraphrases. Average over a
set of pairs before making a claim about "the model". When the component count
gets large (every head × layer × position), screen with the
`attribution-patching` skill and confirm the survivors here.

## Related skills

- `nnsight` — invokes, barriers, module paths
- `attribution-patching` — the same map for a fraction of the compute
- `causal-tracing` — the corrupt-by-noise variant and ROME's protocol
- `ablation` — removing a component rather than swapping it
- `logit-lens` — where the answer becomes readable
