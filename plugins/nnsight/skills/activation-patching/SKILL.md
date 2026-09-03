---
name: activation-patching
description: Establish which components causally carry a behavior by copying activations from one run into another — denoising (clean into corrupt) and noising (corrupt into clean), across layers, positions, attention heads, and MLPs. Use to localize where a fact or capability lives, to confirm that a component found by a probe or attention pattern is actually causal, to build layer×position causal maps, and for IOI-style paired-prompt designs. Includes distributed alignment search (learned subspace patching) and the metric and prompt-design mistakes that produce confident wrong answers.
---

# Activation Patching

Patching answers a causal question that observation cannot: **if this activation
had been different, would the answer change?**

You need two prompts that differ in the fact under test:

- **clean** — produces the correct answer
- **corrupt** — same structure, different answer

Copy one activation from one run into the other and measure the output shift.

- **Denoising** (clean → corrupt): does this activation *suffice* to restore the
  right answer? Localizes where the information is carried.
- **Noising** (corrupt → clean): is this activation *necessary*? Localizes what
  breaks the behavior.

Both are worth running; a component that is sufficient but not necessary (or the
reverse) is telling you about redundancy.

<!-- test: setup -->
```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

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

## The metric: use a logit difference

`P(answer)` alone confounds "the right answer got likelier" with "everything got
likelier". The logit difference between the clean and corrupt answers isolates the
effect and is roughly linear:

```python
with model.trace() as tracer:
    with tracer.invoke(clean):
        logits = model.output.logits[0, -1]
        clean_diff = (logits[paris] - logits[rome]).save()
    with tracer.invoke(corrupt):
        logits = model.output.logits[0, -1]
        corrupt_diff = (logits[paris] - logits[rome]).save()

print(f"clean {clean_diff.item():+.3f}   corrupt {corrupt_diff.item():+.3f}")
```

These two numbers are the endpoints of your scale: patching should move the corrupt
run from `corrupt_diff` toward `clean_diff`. Report the fraction of the gap
recovered, not the raw logit.

## Canonical sweep — two forward passes for the whole layer scan

Cache the clean activations once, then patch every layer in a single batched
trace. This is the pattern to reach for by default: it is one forward pass for the
cache and one for the entire sweep, instead of one pass per layer.

```python
with model.trace(clean):
    clean_resid = nnsight.save([block.output.detach() for block in model.transformer.h])

with model.trace() as tracer:
    scores = nnsight.save([])
    for layer in range(len(model.transformer.h)):
        with tracer.invoke(corrupt):
            model.transformer.h[layer].output[:, SUBJECT, :] = clean_resid[layer][:, SUBJECT, :]
            logits = model.output.logits[0, -1]
            scores.append(logits[paris] - logits[rome])

for layer, score in enumerate(scores):
    bar = "#" * max(0, int((score.item() + 2) * 6))
    print(f"layer {layer:2d}  {score.item():+.3f}  {bar}")
```

```
layer  0  +2.414  ##########################
layer  1  +2.412  ##########################
layer  2  +2.484  ##########################
layer  3  +2.260  #########################
layer  4  +2.176  #########################
layer  5  +2.163  ########################
layer  6  +1.866  #######################
layer  7  +1.755  ######################
layer  8  +0.721  ################
layer  9  -1.016  #####
layer 10  -1.298  ####
layer 11  -1.487  ###
```

A monotone decay like this one is what a residual sweep produces whether or not
the information is early, so read it as a **reachability** curve rather than a
localization. Patching `block.output` at layer L overwrites everything the model
computed at those positions up to and including L, so the two ends are fixed by
the architecture: patch at the embeddings and you reproduce the clean run exactly
(`+2.424683` against a clean `+2.424683`), and patch at the last block and the
non-final positions cannot reach the last position's logits at all. Layer 0's
`+2.414` is one block away from the first identity and layer 11's `-1.487` is the
second one.

The next section measures both ends and gives the sweep the controls that make
the middle of the curve mean something.

## Controls for a patching sweep

Three donors, one sweep function. The first must not move the metric, the second
carries no subject information and should not behave like the first, and the
third is the null:

```python
SUFFIX = slice(5, 10)          # ' is',' in',' the',' city',' of' — identical in both prompts

with model.trace(corrupt):
    corrupt_resid = nnsight.save([block.output.detach() for block in model.transformer.h])

generator = torch.Generator().manual_seed(0)
random_donor = []
for layer in range(len(model.transformer.h)):
    real = clean_resid[layer]
    draw = torch.randn(real.shape, generator=generator).to(real.device)
    random_donor.append(draw / draw.norm(dim=-1, keepdim=True) * real.norm(dim=-1, keepdim=True))

def sweep(positions, donor):
    with model.trace() as tracer:
        out = nnsight.save([])
        for layer in range(len(model.transformer.h)):
            with tracer.invoke(corrupt):
                model.transformer.h[layer].output[:, positions, :] = donor[layer][:, positions, :]
                logits = model.output.logits[0, -1]
                out.append((logits[paris] - logits[rome]).detach())
    return [float(x) for x in out]

subject = sweep(SUBJECT, clean_resid)
self_patch = sweep(SUBJECT, corrupt_resid)
suffix = sweep(SUFFIX, clean_resid)
random_patch = sweep(SUBJECT, random_donor)

assert max(abs(v - corrupt_diff.item()) for v in self_patch) < 1e-3, "self-patch is not a no-op"

print("layer  subject   suffix     self    random")
for layer in range(len(model.transformer.h)):
    print(f"{layer:5d}  {subject[layer]:+7.3f}  {suffix[layer]:+7.3f}  "
          f"{self_patch[layer]:+7.3f}  {random_patch[layer]:+7.3f}")
```

```
layer  subject   suffix     self    random
    0   +2.414   -1.477   -1.487   -0.023
    1   +2.412   -1.491   -1.487   -1.457
    2   +2.484   -1.540   -1.487   -1.536
    3   +2.260   -1.461   -1.487   +3.585
    4   +2.176   -1.386   -1.487   -0.228
    5   +2.163   -1.509   -1.487   -0.168
    6   +1.866   -1.265   -1.487   +0.125
    7   +1.755   -1.213   -1.487   +0.699
    8   +0.721   -0.232   -1.487   -2.237
    9   -1.016   +1.445   -1.487   -1.149
   10   -1.298   +2.078   -1.487   -0.297
   11   -1.487   +2.425   -1.487   -1.487
```

**self** is the control worth running every time: donate the corrupt run's own
activations and the metric must not move. It holds to `4.6e-05` here, and it fails
loudly when the cache is stale, the donor is indexed at the wrong layer, or the
positions do not line up. Prefer it to a check that the *final* layer changes
nothing — that one passes for any donor at all, garbage included, because a
non-final position at the last block cannot reach the last position's logits.

**suffix** patches five tokens that are character-identical in both prompts and so
carry no subject content, and it produces the exact complement of the subject
curve, reaching the full clean `+2.425` at layer 11. At every layer one of the two
columns is near clean and the other near corrupt. Together they say the residual
stream carries the answer left to right — which is true of any prompt and any
fact.

**random** is a matched-norm draw with no information in it, and it swings the
metric by up to `+3.6` at layer 3, further than the real effect at layers 8–11.
An effect smaller than the null is not an effect.

The sweep is still worth running as a first look at depth. The object that
localizes is the map below.

## Same-trace patching with a barrier

When you can't cache first — the activation must come from the same forward pass,
or you are minimizing remote memory — both invokes touch the same module, so you
need `tracer.barrier(2)` to order the read before the write:

```python
LAYER = 2

with model.trace() as tracer:
    barrier = tracer.barrier(2)

    with tracer.invoke(clean):
        donor = model.transformer.h[LAYER].output
        barrier()                                   # donor is read

    with tracer.invoke(corrupt):
        barrier()                                   # wait for it
        model.transformer.h[LAYER].output[:, SUBJECT, :] = donor[:, SUBJECT, :]
        logits = model.output.logits[0, -1]
        patched = (logits[paris] - logits[rome]).save()

print(f"patched logit diff {patched.item():+.3f}")
```

Without the barrier the second invoke reads `donor` before the first has bound it
— `NameError`. See the `nnsight` skill → batching.

## Layer × position map

This is the object that localizes, and it is barely more expensive than the sweep:
patch one (layer, position) cell at a time, one forward pass per layer.

```python
n_pos = len(model.tokenizer(clean).input_ids)
tokens = [model.tokenizer.decode([i]) for i in model.tokenizer(corrupt).input_ids]

grid = []
for layer in range(0, 12, 2):
    with model.trace() as tracer:
        row = nnsight.save([])
        for pos in range(n_pos):
            with tracer.invoke(corrupt):
                model.transformer.h[layer].output[:, pos, :] = clean_resid[layer][:, pos, :]
                logits = model.output.logits[0, -1]
                row.append((logits[paris] - logits[rome]).detach())
    grid.append([float(x) for x in row])

print(f"{'position':<12}" + "".join(f"L{l:<6}" for l in range(0, 12, 2)))
for pos, token in enumerate(tokens):
    cells = "".join(f"{grid[i][pos]:+.2f} " for i in range(len(grid)))
    print(f"{token!r:<12}{cells}")

baseline = corrupt_diff.item()
early_subject = max(grid[i][pos] for i in range(3) for pos in range(1, 5))    # L0-L4
late_subject = max(grid[i][pos] for i in range(4, 6) for pos in range(1, 5))  # L8-L10
assert early_subject > 0.0 > baseline    # subject recovery crosses zero at layers 0-4...
assert late_subject < 0.0                # ...and no subject cell does from layer 8 on
suffix_cells = [grid[i][pos] for i in range(len(grid)) for pos in range(5, 9)]
assert all(abs(v - baseline) < 0.15 for v in suffix_cells)        # ' is'..' city' flat everywhere
assert all(abs(grid[i][-1] - baseline) < 0.15 for i in range(4))  # ' of' does nothing through L6
assert grid[-1][-1] > 0.0                                         # then carries the answer
```

```
position    L0     L2     L4     L6     L8     L10
'The'       -1.49 -1.49 -1.49 -1.49 -1.49 -1.49
' Col'      -1.00 -1.24 -1.33 -1.42 -1.46 -1.50
'os'        -0.31 +0.49 +0.18 -0.76 -1.01 -1.42
'se'        -1.48 -1.22 -1.05 -1.02 -1.18 -1.42
'um'        -0.14 +0.12 +0.08 -0.02 -0.59 -1.41
' is'       -1.48 -1.47 -1.46 -1.50 -1.48 -1.47
' in'       -1.48 -1.53 -1.44 -1.47 -1.48 -1.43
' the'      -1.48 -1.50 -1.49 -1.47 -1.49 -1.47
' city'     -1.50 -1.56 -1.52 -1.42 -1.46 -1.45
' of'       -1.48 -1.44 -1.47 -1.42 -0.39 +1.93
```

Two sites, and the map separates them where the sweep could not: subject pieces
`'os'` and `'um'` move the metric at layers 0–4 and stop mattering by layer 8, and
the final `' of'` position does nothing until layer 8 and then carries the whole
answer. The suffix positions that dominated the sweep's late end (`' is'` through
`' city'`) are flat at baseline everywhere, which is the reading the sweep alone
cannot give you.

Combine with the `logit-lens` skill: the lens says where the answer becomes
*readable*, patching says where it is *carried*.

## Patching attention heads

Heads are slices of the attention output's hidden dimension. Patch at the output
projection's input, which is the per-head result before it is summed:

```python
n_heads = model.config.n_head
head_dim = model.config.n_embd // n_heads
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
            head_scores.append(logits[paris] - logits[rome])

for head, score in enumerate(head_scores):
    print(f"L{LAYER}H{head:<2} {score.item():+.3f}")
```

```
L5H0  -1.479    L5H4  -1.503    L5H8  -1.496
L5H1  -1.487    L5H5  -1.494    L5H9  -1.445
L5H2  -1.466    L5H6  -1.335    L5H10 -1.426
L5H3  -1.490    L5H7  -1.509    L5H11 -1.566
```

All twelve sit at the corrupt baseline of `-1.487`: no single head at layer 5
carries this fact, and the largest deviation (`L5H6`, `+0.15`) is well inside the
random-donor null from the controls section. A flat column is a result. Sweep
layers before reading anything into one.

Patch MLPs the same way through `model.transformer.h[layer].mlp.output`.

## Noising: the necessity direction

Flip the donor and the recipient — corrupt activations into the clean run:

```python
with model.trace(corrupt):
    corrupt_resid = nnsight.save([block.output.detach() for block in model.transformer.h])

with model.trace() as tracer:
    necessity = nnsight.save([])
    for layer in range(0, 12, 3):
        with tracer.invoke(clean):
            model.transformer.h[layer].output[:, SUBJECT, :] = corrupt_resid[layer][:, SUBJECT, :]
            logits = model.output.logits[0, -1]
            necessity.append(logits[paris] - logits[rome])

for i, layer in enumerate(range(0, 12, 3)):
    print(f"layer {layer:2d}  clean run degraded to {necessity[i].item():+.3f}")
```

```
layer  0  clean run degraded to -1.477
layer  3  clean run degraded to -1.461
layer  6  clean run degraded to -1.265
layer  9  clean run degraded to +1.445
```

A large drop means that activation was necessary — and this direction inherits the
same reachability floor as the denoising sweep, mirrored: an early-layer noising
run has the whole model left to propagate the damage, a late-layer one does not.
Give it the same controls, with the roles of clean and corrupt swapped.

Noising with *random* or *mean* activations instead of a paired corrupt run is
ablation — see the `ablation` skill.

## Distributed alignment search (learned subspaces)

Sometimes no single component patches cleanly because the variable is encoded in a
**direction** spread across the residual stream. DAS learns that subspace: patch
only the component of the activation along a learned direction, and optimize the
direction to make the transfer work.

```python
LAYER = 6
with model.trace(clean):
    donor_acts = model.transformer.h[LAYER].output.detach().save()

generator = torch.Generator().manual_seed(0)
direction = torch.nn.Parameter(torch.randn(768, generator=generator).to(model.device) * 0.02)
optimizer = torch.optim.Adam([direction], lr=0.05)

for step in range(20):
    with model.trace(corrupt):
        unit = direction / direction.norm()
        acts = model.transformer.h[LAYER].output
        delta = donor_acts[:, SUBJECT, :] - acts[:, SUBJECT, :]
        acts[:, SUBJECT, :] = acts[:, SUBJECT, :] + (delta @ unit).unsqueeze(-1) * unit
        logits = model.output.logits[0, -1]
        loss = -(logits[paris] - logits[rome])
        with loss.backward():
            pass
        tracked = nnsight.save(loss.item())
    optimizer.step()
    optimizer.zero_grad()
    if step % 5 == 0:
        print(f"step {step:2d}  logit diff {-tracked:+.3f}")
```

```
step  0  logit diff -1.491
step  5  logit diff +31.455
step 10  logit diff +53.149
step 15  logit diff +62.827
```

**Read that result skeptically — it is the standard criticism of DAS.** A learned
direction drove the logit difference to +63 when the *real* clean run only
achieves `+2.425`. The optimizer found a direction that pushes the output,
not necessarily the one the model uses. Constrain and validate:

- hold out prompts the direction was not trained on, and report transfer there
- keep the intervention on the same scale as the natural activation difference
- compare against a random direction trained the same way — if random does nearly
  as well, the subspace claim is empty
- prefer the smallest subspace dimension that works
- **train the same subspace toward an arbitrary target and see whether that
  succeeds too.** On Llama-3.2-1B, one rotated dimension at layer 8 trained to emit
  " Brazil" from Rio's activation reaches `P=1.00`; the same dimension, the same
  donor, trained to emit " Japan" reaches `P=0.999`, and trained from a donor
  prompt with no city in it at all it still reaches " Brazil" at `P=1.00`. A
  subspace that can be pointed at any answer is not carrying the fact — see the
  [DAS tutorial](https://nnsight.net/notebooks/tutorials/causal_mediation_analysis/DAS/),
  which runs these three nulls.

Used with those controls, DAS answers questions single-component patching cannot:
whether a variable exists as a linear feature at all, and how many dimensions it
occupies.

## Designing the experiment

**Prompt pairs.** Same token length, same structure, differing only in the fact
under test. Unequal lengths mean position `i` is not the same thing in both runs
and every result is garbage. Check with an assert, as in the setup above. The
constraint is stronger than it looks in a batched sweep: every invoke's
activations are padded to the batch's longest prompt, so an absolute index shifts
with the batch's contents while `[:, -1, :]` does not. Every sweep in this skill
batches invokes of one prompt pair at one length, which is what makes
`slice(1, 5)` and the appended result order safe here; neither survives a mixed
length batch. See the `nnsight` skill on invokes and batching.

**Which activation.** The residual stream (`block.output`) tells you *where*
information is; attention output tells you *what moved it*; MLP output tells you
*what computed it*. Sweep the residual first, then drill in.

**Direction of copy.** Denoising and noising answer different questions. State
which one a number came from.

**Controls that catch wiring bugs.** Run at least one that can actually fail:
patching a layer with the recipient run's own activations must leave the metric
where it was, and a matched-norm random donor must move it less than the effect
you are claiming. Two checks that look like controls and are not: "patching the
final layer's residual fully determines the output" and "patching a non-final
position at the final layer changes nothing" are both true of any donor, since
they only restate which states reach the last position's logits.

**One pair is an anecdote.** Layer indices move across paraphrases. Average over a
set of pairs before making a claim about "the model".

## Cost

Patching is `O(components)` forward passes if written naively. Batching a sweep
into invokes makes it `O(1)` passes per layer; caching the clean run makes the
donor free. When the component count gets large (every head × every layer ×
every position), switch to gradient approximation — see the
`attribution-patching` skill — and confirm the top candidates with real patching.

## Related skills

- `nnsight` — invokes, barriers, module paths
- `attribution-patching` — the same map for a fraction of the compute
- `causal-tracing` — the corrupt-by-noise variant and ROME's protocol
- `ablation` — removing a component rather than swapping it
- `logit-lens` — where the answer becomes readable
