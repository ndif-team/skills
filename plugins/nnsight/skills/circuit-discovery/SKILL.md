---
name: circuit-discovery
description: Find the subgraph of components a model uses for a task and prove it is the right one — task and metric design, per-head attribution to rank candidates, edge attribution and path patching for connections, greedy pruning, and the faithfulness/completeness/minimality tests that separate a circuit from a list of high-scoring components. Use for IOI-style circuit analysis, for automating what activation patching does one component at a time, and whenever a claim of the form "these components implement this behavior" needs evidence.
---

# Circuit Discovery

A circuit is a claim: *these components, connected this way, are what the model
uses for this task.* Producing a ranked list of components is the easy part;
showing the list is the circuit is the work.

The pipeline:

1. **Task** — a prompt distribution with a clean/corrupt contrast
2. **Metric** — a scalar that separates them
3. **Rank** — attribute every component cheaply
4. **Verify** — real interventions on the top candidates
5. **Connect** — which component feeds which
6. **Validate** — faithfulness, completeness, minimality

<!-- test: setup -->
```python
import random
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)
n_layers = len(model.transformer.h)
n_heads = model.config.n_head
head_dim = model.config.n_embd // n_heads
```

## 1–2. Task and metric

The running example is IOI (indirect object identification): "When Mary and John
went to the store, **John** gave a drink to ___" → " Mary". The corrupt variant
swaps which name is repeated, flipping the answer.

Use a **distribution**, not one prompt — a circuit found on a single example is
overfit to that example.

```python
names = [("Mary", "John"), ("Alice", "Bob"), ("Sarah", "Tom"), ("Emma", "Jack")]
clean = [f"When {a} and {b} went to the store, {b} gave a drink to" for a, b in names]
corrupt = [f"When {a} and {b} went to the store, {a} gave a drink to" for a, b in names]

io_ids = torch.tensor([model.tokenizer.encode(" " + a)[0] for a, _ in names], device=model.device)
s_ids = torch.tensor([model.tokenizer.encode(" " + b)[0] for _, b in names], device=model.device)
rows = torch.arange(len(names), device=model.device)

def logit_diff(logits):
    """Mean (indirect object − subject) logit difference at the last position."""
    return (logits[rows, -1, io_ids] - logits[rows, -1, s_ids]).mean()

with model.trace(clean):
    clean_metric = logit_diff(model.output.logits).detach().save()
with model.trace(corrupt):
    corrupt_metric = logit_diff(model.output.logits).detach().save()

print(f"clean {float(clean_metric):+.3f}   corrupt {float(corrupt_metric):+.3f}")
```

```
clean +2.654   corrupt -2.772
```

Those endpoints define the scale: every later number is reported as the fraction
of that gap recovered.

## 3. Rank every head at once

Attribution patching gives all 144 heads from two forward passes and one backward.
Heads live in slices of the attention output projection's input.

```python
with torch.no_grad():
    with model.trace(clean):
        clean_heads = nnsight.save([
            model.transformer.h[i].attn.c_proj.input for i in range(n_layers)
        ])

with model.trace(corrupt):
    refs = [model.transformer.h[i].attn.c_proj.input for i in range(n_layers)]
    corrupt_heads = nnsight.save([h.detach() for h in refs])
    with logit_diff(model.output.logits).backward():
        grads = nnsight.save([])
        for i in reversed(range(n_layers)):
            grads.append(refs[i].grad.clone())
grads = grads[::-1]

scores = torch.zeros(n_layers, n_heads)
for layer in range(n_layers):
    delta = (clean_heads[layer] - corrupt_heads[layer]) * grads[layer]
    for head in range(n_heads):
        scores[layer, head] = delta[..., head * head_dim:(head + 1) * head_dim].sum()

ranked = [(int(i) // n_heads, int(i) % n_heads) for i in scores.flatten().topk(8).indices]
for layer, head in ranked:
    print(f"L{layer}H{head:<2} attribution {scores[layer, head]:+.3f}")
```

```
L8H10 +1.419    L7H9  +1.070
L8H6  +1.349    L9H9  +1.016
L5H5  +1.114    L10H0 +0.613
```

These are the published IOI heads — L9H9 and L10H0 are name movers, L8H6/L8H10 are
S-inhibition heads, L5H5 is an induction head — recovered here from scratch. That
agreement is reassuring but not evidence; the validation below is.

Attribution is a screen, not a result: it is a linear approximation and it fails
for large interventions (see the `attribution-patching` skill). Everything it
ranks must be confirmed with a real intervention.

## 4. Verify — faithfulness with a control

The strongest single test: **mean-ablate every head outside the candidate circuit**
and check that the metric survives. And run the same test on a random set of the
same size, which is the control almost every write-up omits.

```python
with model.trace(clean):
    head_means = nnsight.save([
        model.transformer.h[i].attn.c_proj.input.mean(0, keepdim=True).detach()
        for i in range(n_layers)
    ])

circuit = {(int(i) // n_heads, int(i) % n_heads) for i in scores.flatten().topk(10).indices}

random.seed(0)
control = set()
while len(control) < len(circuit):
    control.add((random.randrange(n_layers), random.randrange(n_heads)))

def keep_only(keep):
    with model.trace(clean):
        for layer in range(n_layers):
            for head in range(n_heads):
                if (layer, head) not in keep:
                    lo, hi = head * head_dim, (head + 1) * head_dim
                    model.transformer.h[layer].attn.c_proj.input[:, :, lo:hi] = \
                        head_means[layer][:, :, lo:hi]
        result = logit_diff(model.output.logits).detach().save()
    return float(result)

span = float(clean_metric) - float(corrupt_metric)
for label, keep in [("circuit", circuit), ("random", control)]:
    value = keep_only(keep)
    print(f"{label:8} keep {len(keep)} heads -> {value:+.3f} "
          f"({(value - float(corrupt_metric)) / span * 100:.0f}% of clean)")
```

```
circuit  keep 10 heads -> +2.574 (99% of clean)
random   keep 10 heads -> +0.747 (65% of clean)
```

Ten heads out of 144 recover 99% of the behavior. **And the control is the reason
this is meaningful**: a random set of ten already recovers 65%, because
mean-ablation is gentle and the model is redundant. Reporting 99% without the 65%
would badly overstate the finding.

## 5. Connect the components

Node importance does not tell you the graph. Two tools:

**Edge attribution** — attribute the *connection* from an upstream component to a
downstream one by pairing the upstream activation difference with the gradient at
the downstream input. The same two passes give every edge, which is what makes
automated circuit discovery tractable:

```python
UPSTREAM, DOWNSTREAM = 5, 9

with model.trace(corrupt):
    upstream_ref = model.transformer.h[UPSTREAM].attn.c_proj.input
    upstream_corrupt = upstream_ref.detach().save()
    downstream_ref = model.transformer.h[DOWNSTREAM].attn.c_proj.input
    with logit_diff(model.output.logits).backward():
        downstream_grad = downstream_ref.grad.clone().save()
        upstream_grad = upstream_ref.grad.clone().save()

for head in range(0, n_heads, 3):
    lo, hi = head * head_dim, (head + 1) * head_dim
    contribution = ((clean_heads[UPSTREAM] - upstream_corrupt)[..., lo:hi]
                    * upstream_grad[..., lo:hi]).sum()
    print(f"L{UPSTREAM}H{head:<2} total downstream effect {float(contribution):+.3f}")
```

**Path patching** — the precise version: patch an upstream component's output but
let it reach only *one* downstream destination, freezing every other path. It
answers "does A affect the output *through* B?" rather than "does A matter at
all", and it is how the published IOI circuit distinguishes name movers from
S-inhibition heads. Implement it by caching the frozen components' values in a
first pass and re-imposing them while the patched value flows.

## 6. Prune to minimality

A ranked top-K is not minimal. Greedy pruning removes members while the metric
holds:

```python
current = set(circuit)
threshold = 0.90 * span + float(corrupt_metric)

for member in sorted(circuit, key=lambda lh: scores[lh[0], lh[1]]):
    trial = current - {member}
    if keep_only(trial) >= threshold:
        current = trial

print(f"pruned circuit ({len(current)} heads): {sorted(current)}")
```

```
pruned circuit (2 heads): [(9, 9), (10, 0)]
```

Ten heads prune to **two** — L9H9 and L10H0, exactly the name-mover heads — while
still holding 90% of the metric. Removing the lowest-scoring member first tends to
find a smaller circuit than removing the highest.

Minimal is not the same as complete: this pair is enough to *produce* the behavior
under mean ablation, but the S-inhibition and induction heads that were pruned are
part of how the model computes which name to move. Report both the minimal set and
the full candidate set, and say which test each survived.

## Validating a circuit claim

| Test | Question | How |
|---|---|---|
| **Faithfulness** | does the circuit alone do the task? | ablate everything outside it; compare to a random-set control |
| **Completeness** | is anything important missing? | ablate the circuit; the metric should collapse to near-corrupt |
| **Minimality** | is every member needed? | remove each member; the metric should drop |

```python
inside = keep_only(circuit)
with model.trace(clean):
    for layer, head in sorted(circuit):          # sorted! see the note below
        lo, hi = head * head_dim, (head + 1) * head_dim
        model.transformer.h[layer].attn.c_proj.input[:, :, lo:hi] = head_means[layer][:, :, lo:hi]
    knocked_out = logit_diff(model.output.logits).detach().save()

print(f"circuit only:     {inside:+.3f}")
print(f"circuit removed:  {float(knocked_out):+.3f}   (clean {float(clean_metric):+.3f})")
```

**Sort your component sets before iterating them.** A circuit is naturally a
`set`, and iterating a set yields an arbitrary layer order, which violates
nnsight's forward-order rule and raises `OutOfOrderError` — intermittently, since
set order depends on the contents. Every loop that touches modules from a
collection needs `sorted(...)`. (The `keep_only` helper above is safe because it
loops over `range(n_layers)`.)

```
circuit only:     +2.574
circuit removed:  +1.885   (clean +2.654)
```

Read that second line carefully: mean-ablating all ten circuit heads leaves the
metric at +1.885, nowhere near the corrupt baseline of −2.772. **The circuit is
sufficient but not complete** — the model still does most of the task without it,
because other components take over. That is the honest state of this analysis, and
it is exactly what the completeness test exists to reveal.

Completeness is the test people skip, and it is the one that catches a circuit
that is *sufficient* but not the mechanism the model actually uses. When it fails,
the usual causes are redundancy (backup name-mover heads are a documented feature
of IOI) and too gentle an ablation — retry with resample ablation from the corrupt
distribution before concluding anything.

## Practical guidance

**Choose the corrupt distribution carefully.** It defines what "not doing the
task" means. Corrupting the names tests name selection; corrupting the sentence
structure tests something else entirely, and you will get a different circuit.

**Ablation choice changes the circuit.** Mean ablation is gentle, zero ablation is
brutal, resample ablation is the strictest in-distribution control. State which
you used — results are not comparable across choices (see the `ablation` skill).

**Positions matter as much as components.** A head can be in the circuit at one
token position and irrelevant at another. Full circuit work is over
(component, position) pairs.

**Scale honestly.** Attribution over every (head, position) edge of a large model
is a big object; report what you searched and what you pruned rather than
implying exhaustiveness.

**Redundancy is real.** The 65% control above is the reminder: models route around
damage, so "the metric survived" is weak evidence unless the control is worse.

## Related skills

- `attribution-patching` — the ranking step and its failure modes
- `activation-patching` — verifying individual candidates
- `ablation` — ablation choices and what each one means
- `attention-analysis` — what the discovered heads actually do
- `interp-experiment-design` — metric choice and controls in general
