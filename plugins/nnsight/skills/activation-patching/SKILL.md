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
layer  4  +2.176  #########################
layer  8  +0.721  ################
layer 10  -1.299  ####
layer 11  -1.487  ###
```

Read: patching the subject tokens in **early** layers restores the clean answer;
by layer 9 it does nothing, because the city information has already been moved
out of those positions. Baseline (unpatched corrupt) is `-1.487` — layer 11
patching recovers exactly nothing, which is the sanity check that the sweep is
wired correctly.

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

The standard causal trace: patch one (layer, position) cell at a time. Every cell
in the grid runs in one forward pass per layer.

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
```

The bright cells are where and when the information matters. Combine with the
`logit-lens` skill: the lens says where the answer becomes *readable*, patching
says where it is *carried*.

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

A large drop means that activation was necessary. Noising with *random* or *mean*
activations instead of a paired corrupt run is ablation — see the `ablation` skill.

## Distributed alignment search (learned subspaces)

Sometimes no single component patches cleanly because the variable is encoded in a
**direction** spread across the residual stream. DAS learns that subspace: patch
only the component of the activation along a learned direction, and optimize the
direction to make the transfer work.

```python
LAYER = 6
with model.trace(clean):
    donor_acts = model.transformer.h[LAYER].output.detach().save()

direction = torch.nn.Parameter(torch.randn(768, device=model.device) * 0.02)
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
step  0  logit diff -1.441
step  5  logit diff +21.274
step 15  logit diff +61.548
```

**Read that result skeptically — it is the standard criticism of DAS.** A learned
direction drove the logit difference to +61 when the *real* clean run only
achieves a few logits. The optimizer found a direction that pushes the output,
not necessarily the one the model uses. Constrain and validate:

- hold out prompts the direction was not trained on, and report transfer there
- keep the intervention on the same scale as the natural activation difference
- compare against a random direction trained the same way — if random does nearly
  as well, the subspace claim is empty
- prefer the smallest subspace dimension that works

Used with those controls, DAS answers questions single-component patching cannot:
whether a variable exists as a linear feature at all, and how many dimensions it
occupies.

## Designing the experiment

**Prompt pairs.** Same token length, same structure, differing only in the fact
under test. Unequal lengths mean position `i` is not the same thing in both runs
and every result is garbage. Check with an assert, as in the setup above.

**Which activation.** The residual stream (`block.output`) tells you *where*
information is; attention output tells you *what moved it*; MLP output tells you
*what computed it*. Sweep the residual first, then drill in.

**Direction of copy.** Denoising and noising answer different questions. State
which one a number came from.

**Controls that catch wiring bugs.** Patching a layer with its own activations must
be a no-op; patching the final layer's residual must fully determine the output;
the unpatched corrupt baseline must reproduce the corrupt answer. Run at least one
of these every time.

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
