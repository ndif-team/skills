---
name: circuit-discovery
description: Find the subgraph of components a model uses for a task and prove it is the right one — the deliverable is a validated set of components plus the evidence. Covers task and metric design, per-head attribution to rank candidates, edge attribution and path patching for connections, greedy pruning, and the faithfulness/completeness/minimality tests that separate a circuit from a list of high-scoring components. Use for IOI-style circuit analysis, for automating what activation patching does one component at a time, and whenever a claim of the form "these components implement this behavior" needs evidence.
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
head_dim = model.transformer.h[0].attn._module.head_dim   # equals n_embd // n_head on GPT-2, not on GQA/Gemma models
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
L8H10 attribution +1.419
L8H6  attribution +1.349
L5H5  attribution +1.114
L7H9  attribution +1.070
L9H9  attribution +1.016
L10H0 attribution +0.613
L7H3  attribution +0.588
L3H0  attribution +0.499
```

These are the published IOI heads — L9H9 and L10H0 are name movers, L8H6/L8H10 are
S-inhibition heads, L5H5 is an induction head — recovered here from scratch. That
agreement is reassuring but not evidence; the validation below is.

Attribution is a screen, not a result: it is a linear approximation and it fails
for large interventions (see the `attribution-patching` skill). Everything it
ranks must be confirmed with a real intervention.

## 4. Verify — faithfulness against a floor that the intervention can reach

The strongest single test: **mean-ablate every head outside the candidate circuit**
and check that the metric survives. Two things decide whether the resulting number
means anything, and the second one is the one write-ups omit: a control circuit of
the same size, and a **floor** — what the metric reads when the circuit is empty.

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

# a circuit that is wrong on purpose: ten heads from the first three layers, which
# are too early to be doing name selection at all
random.seed(1)
wrong = set()
while len(wrong) < len(circuit):
    wrong.add((random.randrange(0, 3), random.randrange(n_heads)))

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

floor = keep_only(set())
span = float(clean_metric) - floor

print(f"{'empty circuit (the floor)':28} -> {floor:+.3f}")
for label, keep in [("top-10 by attribution", circuit), ("random 10", control),
                    ("10 heads from layers 0-2", wrong)]:
    value = keep_only(keep)
    print(f"{label:28} -> {value:+.3f}   {(value - floor) / span * 100:6.1f}% of clean")

assert (keep_only(circuit) - floor) / span > 0.9      # the circuit carries the task
assert (keep_only(wrong) - floor) / span < 0.1        # a wrong circuit does not
```

```
empty circuit (the floor)    -> +0.785
top-10 by attribution        -> +2.574     95.7% of clean
random 10                    -> +0.747     -2.0% of clean
10 heads from layers 0-2     -> +0.783     -0.1% of clean
```

**Scale against the floor, not against the corrupt metric.** Mean-ablating heads
cannot reach `−2.772`: the MLPs and the residual stream are untouched and they
alone hold the metric at `+0.785`. Divide by `clean − corrupt` instead of
`clean − floor` and every one of these numbers inflates by the same 66 points —
the empty circuit scores 66%, the random control 65%, and the *worst* ten heads by
attribution score 85%. On that scale nothing can fail, which is why the test needs
the two lines above it: a wrong circuit scores `−0.1%`, and a test that a wrong
answer passes is not a test.

The floor is one extra `keep_only(set())` call. Run it before reporting any
ablation-based score.

## 5. Connect the components

Node importance does not tell you the graph. Two tools:

**Edge attribution** — attribute the *connection* from an upstream component to a
downstream one: (clean − corrupt) of what the upstream component **writes** into
the residual stream, dotted with the gradient of the metric at the downstream
component's **input**, taken through that component only. The gradient on the
residual stream itself is the wrong one — it sums every path out of that point,
so pairing a write with it gives back the node attribution of section 3.

Every component reads the residual stream through its own LayerNorm. Hand each
norm a private copy of its input, and the gradient on that copy flows only through
the one reader. One corrupt forward and one backward then give every edge:

```python
readers = []                                   # forward order: L0 attn, L0 mlp, ..., final norm
for i, block in enumerate(model.transformer.h):
    readers += [(f"L{i} attn", block.ln_1), (f"L{i} mlp", block.ln_2)]
readers.append(("logits", model.transformer.ln_f))

with model.trace(corrupt):
    copies = []
    for _, norm in readers:
        copy = norm.input.clone()
        norm.input = copy                      # this reader alone now consumes `copy`
        copies.append(copy)
    with logit_diff(model.output.logits).backward():
        reader_grads = nnsight.save([])
        for copy in reversed(copies):
            reader_grads.append(copy.grad.clone())
reader_grads = reader_grads[::-1]

def head_write(layer, head, z):
    """What one head adds to the residual stream: its slice of z through its rows of c_proj."""
    lo, hi = head * head_dim, (head + 1) * head_dim
    return z[..., lo:hi] @ model.transformer.h[layer].attn.c_proj.weight[lo:hi]

def edges_from(layer, head):
    """Attribution of every edge out of one head, keyed by downstream reader."""
    with torch.no_grad():
        delta = head_write(layer, head, clean_heads[layer]) - head_write(layer, head, corrupt_heads[layer])
        return {name: float((delta * reader_grads[k]).sum())
                for k, (name, _) in enumerate(readers) if k > 2 * layer}   # readers after the head

for layer, head in [(5, 5), (8, 6)]:
    edges = edges_from(layer, head)
    top = sorted(edges.items(), key=lambda kv: -abs(kv[1]))[:3]
    print(f"L{layer}H{head} -> " + "   ".join(f"{name} {value:+.3f}" for name, value in top))
    # the edges out of a head partition its node attribution exactly
    assert abs(sum(edges.values()) - float(scores[layer, head])) < 1e-3

random.seed(0)
null = []
for _ in range(100):
    layer, head = random.randrange(n_layers), random.randrange(n_heads)
    null.append(abs(random.choice(list(edges_from(layer, head).values()))))
null_median = sorted(null)[len(null) // 2]
print(f"random edge, median |attribution| {null_median:.3f}")
assert edges_from(5, 5)["L8 attn"] > 20 * null_median
```

```
L5H5 -> L8 attn +1.140   L7 attn +0.530   L6 attn -0.240
L8H6 -> L10 attn +0.712   L9 attn +0.399   L11 attn +0.219
random edge, median |attribution| 0.005
```

The induction head L5H5 sends its effect almost entirely into layer 8's attention,
where the S-inhibition heads L8H6 and L8H10 sit, and L8H6 sends its effect into the
name-mover layers 9 and 10 — the published IOI wiring, as edges rather than as a
ranked list. The assert inside the loop is the check to keep: a head's outgoing
edges must sum to its node attribution, because the gradient on the residual
stream is the sum of the gradients through each reader. Read the gradients off the
shared residual instead of private copies and every reader gets the total, so the
edges overcount and the identity fails.

Here a downstream node is a whole layer's attention. Splitting it into heads means
separate query, key and value inputs per head, which GPT-2's fused `c_attn` does
not expose as separate modules; treat the layer as the node unless you need
head-level destinations.
Edge scores are attributions, so they inherit every limit in the
`attribution-patching` skill; verify the edges you report with path patching.

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
threshold = 0.90 * span + floor

for member in sorted(circuit, key=lambda lh: scores[lh[0], lh[1]]):
    trial = current - {member}
    if keep_only(trial) >= threshold:
        current = trial

pruned_value = keep_only(current)
print(f"pruned circuit ({len(current)} heads): {sorted(current)} -> {pruned_value:+.3f} "
      f"({(pruned_value - floor) / span * 100:.1f}% of clean)")
assert pruned_value >= threshold

# the two name movers alone clear 90% of clean − corrupt, but not 90% of clean − floor
name_movers = keep_only({(9, 9), (10, 0)})
assert name_movers >= 0.90 * (float(clean_metric) - float(corrupt_metric)) + float(corrupt_metric)
assert name_movers < threshold
```

```
pruned circuit (3 heads): [(7, 9), (9, 9), (10, 0)] -> +2.523 (93.0% of clean)
```

Ten heads prune to **three** — the name movers L9H9 and L10H0 plus the
S-inhibition head L7H9 — while still holding 90% of the metric measured from the
floor. Removing the lowest-scoring member first tends to find a smaller circuit
than removing the highest. Prune against the floor: on the corrupt-metric scale
the two name movers alone clear 90% (the asserts above), so the same loop with that
threshold drops L7H9 as well.

Minimal is not the same as complete: these three heads are enough to *produce* the behavior
under mean ablation, but the S-inhibition and induction heads that were pruned are
part of how the model computes which name to move. Report both the minimal set and
the full candidate set, and say which test each survived.

## Validating a circuit claim

| Test | Question | How |
|---|---|---|
| **Faithfulness** | does the circuit alone do the task? | ablate everything outside it; compare to a random-set control and to the empty circuit |
| **Completeness** | is anything important missing? | ablate the circuit; the metric should collapse toward the floor |
| **Minimality** | is every member needed? | remove each member; the metric should drop |
| **Falsifiability** | can this test fail? | run it on a circuit you know is wrong; it must score badly |

```python
inside = keep_only(circuit)
with model.trace(clean):
    for layer, head in sorted(circuit):          # sorted! see the note below
        lo, hi = head * head_dim, (head + 1) * head_dim
        model.transformer.h[layer].attn.c_proj.input[:, :, lo:hi] = head_means[layer][:, :, lo:hi]
    knocked_out = logit_diff(model.output.logits).detach().save()

print(f"circuit only:     {inside:+.3f}")
print(f"circuit removed:  {float(knocked_out):+.3f}   "
      f"(clean {float(clean_metric):+.3f}, floor {floor:+.3f})")
```

**Sort your component sets before iterating them.** A circuit is naturally a
`set`, and iterating a set yields an arbitrary layer order, which violates
nnsight's forward-order rule and raises `OutOfOrderError` — intermittently, since
set order depends on the contents. Every loop that touches modules from a
collection needs `sorted(...)`. (The `keep_only` helper above is safe because it
loops over `range(n_layers)`.)

```
circuit only:     +2.574
circuit removed:  +1.885   (clean +2.654, floor +0.785)
```

Read that second line against the floor. Removing all ten circuit heads costs
`2.654 − 1.885 = 0.769` of the `1.869` an ablation can take away — **41% of the
achievable damage**. So the circuit is sufficient (96%) and far from complete: the
model still does most of the task without it, because other components take over.
That is the honest state of this analysis, and it is exactly what the completeness
test exists to reveal.

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

**Redundancy is real.** The completeness result above is the reminder: ten heads
that carry 96% of the task cost only 41% of it when removed, because models route
around damage. "The metric survived" is weak evidence unless the control is worse.

## Related skills

- `attribution-patching` — the ranking step and its failure modes
- `activation-patching` — verifying individual candidates
- `ablation` — ablation choices and what each one means
- `attention-analysis` — what the discovered heads actually do
- `interp-experiment-design` — metric choice and controls in general
