# Circuit discovery

A circuit is a claim: this set of components, connected this way, is what the
model uses for the task. A ranked list of components is where it starts. The
claim rests on four tests, and each has a number below:

| Test | Question | Step |
|---|---|---|
| faithfulness | does the set alone keep the metric, on a scale whose zero the ablation can reach? | [3](#3-faithfulness-against-a-floor) |
| a test that can fail | do a random set and a deliberately wrong set of the same size score near zero? | [3](#3-faithfulness-against-a-floor) |
| minimality | is every member needed? | [5](#5-prune) |
| completeness | does removing the set lose the metric? | [6](#6-completeness) |

```python
import random
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True)
n_layers = len(model.transformer.h)
n_heads = model.config.n_head
head_dim = model.transformer.h[0].attn._module.head_dim
every_head = {(layer, head) for layer in range(n_layers) for head in range(n_heads)}

def heads(layer):
    """Every head of a layer side by side: head h is columns h * head_dim to (h + 1) * head_dim."""
    return model.transformer.h[layer].attn.c_proj.input
```

The head convention is in [access-and-modify](access-and-modify.md).

## 1. Task and metric

A set of prompts, not one, each with a corrupt twin that flips the answer. Here
it is indirect object identification: "When Mary and John went to the store,
John gave a drink to" is completed with " Mary", and the corrupt twin repeats
the other name.

```python
names = [("Mary", "John"), ("Alice", "Bob"), ("Sarah", "Tom"), ("Emma", "Jack")]
clean = [f"When {a} and {b} went to the store, {b} gave a drink to" for a, b in names]
corrupt = [f"When {a} and {b} went to the store, {a} gave a drink to" for a, b in names]
assert len({len(model.tokenizer(text).input_ids) for text in clean + corrupt}) == 1   # positions align

def token(name):
    return model.tokenizer.encode(" " + name, add_special_tokens=False)[0]

io_ids = torch.tensor([token(a) for a, _ in names], device=model.device)
s_ids = torch.tensor([token(b) for _, b in names], device=model.device)
rows = torch.arange(len(names), device=model.device)

def logit_diff(logits):
    """Mean logit difference, indirect object minus subject, at the last position."""
    return (logits[rows, -1, io_ids] - logits[rows, -1, s_ids]).mean()

with torch.no_grad(), model.trace() as tracer:
    with tracer.invoke(clean):
        clean_metric = logit_diff(model.output.logits).save()
    with tracer.invoke(corrupt):
        corrupt_metric = logit_diff(model.output.logits).save()

clean_metric, corrupt_metric = float(clean_metric), float(corrupt_metric)
print(f"clean {clean_metric:+.3f}   corrupt {corrupt_metric:+.3f}")
```

```
clean +2.654   corrupt -2.772
```

## 2. Rank every head

[Attribution patching](attribution-patching.md) scores all 144 heads from two
forward passes and one backward: (clean − corrupt) activation times the
gradient of the metric.

```python
with torch.no_grad(), model.trace(clean):
    clean_heads = nnsight.save([heads(layer) for layer in range(n_layers)])

with model.trace(corrupt):
    live = [heads(layer) for layer in range(n_layers)]
    corrupt_heads = nnsight.save([z.detach() for z in live])
    with logit_diff(model.output.logits).backward():
        grads = nnsight.save([z.grad.clone() for z in reversed(live)])
grads = grads[::-1]

scores = torch.stack([
    ((clean_heads[layer] - corrupt_heads[layer]) * grads[layer]).unflatten(-1, (n_heads, head_dim)).sum((0, 1, 3))
    for layer in range(n_layers)
])                                                             # (layers, heads)

for i in scores.flatten().abs().topk(8).indices.tolist():
    layer, head = divmod(i, n_heads)
    print(f"{f'L{layer}H{head}':<7}{scores[layer, head]:+.3f}")
```

```
L8H10  +1.419
L8H6   +1.349
L10H7  -1.310
L5H5   +1.114
L11H10 -1.113
L7H9   +1.070
L9H9   +1.016
L10H0  +0.613
```

Rank by `|score|`. A signed ranking hides the heads that push against the
answer: L10H7 and L11H10 are third and fifth here. The score is a linear
approximation, so it only chooses candidates for the interventions below.

## 3. Faithfulness, against a floor

Mean-ablate every head outside the candidate set and see whether the metric
survives. Run the same test on a random set and on a set that is wrong on
purpose.

```python
# a prompt's replacement: the mean over the OTHER clean prompts, per position
head_means = [(z.sum(0, keepdim=True) - z) / (len(clean) - 1) for z in clean_heads]

def ablate(heads_out, source):
    """The metric on the clean prompts with each head of `heads_out` replaced by its slice of `source`."""
    with torch.no_grad(), model.trace(clean):
        for layer, head in sorted(heads_out):                  # sorted: see the note below
            lo, hi = head * head_dim, (head + 1) * head_dim
            heads(layer)[:, :, lo:hi] = source[layer][:, :, lo:hi]
        result = logit_diff(model.output.logits).save()
    return float(result)

def keep_only(keep):
    return ablate(every_head - set(keep), head_means)

circuit = {divmod(i, n_heads) for i in scores.flatten().topk(10).indices.tolist()}   # signed: the ten that carry the answer
random.seed(0)
control = set(random.sample(sorted(every_head), len(circuit)))
wrong = set(random.sample(sorted(h for h in every_head if h[0] < 3), len(circuit)))  # layers 0-2 only

floor = keep_only(set())                                       # the empty circuit
span = clean_metric - floor
print(f"empty circuit {floor:+.3f}, already {(floor - corrupt_metric) / (clean_metric - corrupt_metric):.0%} of the way from corrupt to clean")

faithfulness = {}
for label, keep in [("top 10 by attribution", circuit), ("random 10", control), ("10 from layers 0-2", wrong)]:
    faithfulness[label] = (keep_only(keep) - floor) / span
    print(f"{label:<24}{faithfulness[label]:7.1%} of clean - floor")

assert faithfulness["top 10 by attribution"] > 0.8
assert abs(faithfulness["random 10"]) < 0.1 and abs(faithfulness["10 from layers 0-2"]) < 0.1
```

```
empty circuit +0.069, already 52% of the way from corrupt to clean
top 10 by attribution     83.4% of clean - floor
random 10                  2.8% of clean - floor
10 from layers 0-2        -0.2% of clean - floor
```

**Scale against the floor, not against the corrupt metric.** A mean of clean
runs is not a corrupt run, so mean-ablating every head stops at +0.069 and no
set of heads can score lower. On the `clean − corrupt` scale that empty circuit
already scores 52%, and no set can fail.

**Leave a prompt out of its own mean.** With four prompts each is a quarter of
the plain batch mean, which lifts the floor to +0.785 and the ten heads' score
to 96%.

**Sort a set of components before looping over it in a trace.** A `set`
iterates in an order that depends on its contents, and a later layer before an
earlier one raises `OutOfOrderError`.

## 4. Edges

A node score does not say which component feeds which. An edge from A to B is
attributed as (clean − corrupt) of what A **writes** into the residual stream,
times the gradient of the metric at B's **input**, taken through B alone. The
gradient on the residual stream itself sums every path out of that point, so it
gives back the node score of step 2.

Every component reads the residual stream through its own LayerNorm. Give each
norm a private copy of its input, and the gradient on the copy is that reader's
alone. One corrupt forward and one backward give every edge:

```python
readers = []                                   # forward order: L0 attn, L0 mlp, ..., the final norm
for i, block in enumerate(model.transformer.h):
    readers += [(f"L{i} attn", block.ln_1), (f"L{i} mlp", block.ln_2)]
readers.append(("logits", model.transformer.ln_f))

with model.trace(corrupt):
    copies = []
    for _, norm in readers:
        copy = norm.input.clone()
        norm.input = copy                      # this reader alone consumes `copy`
        copies.append(copy)
    with logit_diff(model.output.logits).backward():
        reader_grads = nnsight.save([copy.grad.clone() for copy in reversed(copies)])
reader_grads = reader_grads[::-1]

def write_delta(layer, head):
    """Clean minus corrupt of what one head adds to the residual stream (the c_proj bias cancels)."""
    lo, hi = head * head_dim, (head + 1) * head_dim
    with torch.no_grad():
        weight = model.transformer.h[layer].attn.c_proj.weight[lo:hi]    # Conv1D: (in, out)
        return (clean_heads[layer] - corrupt_heads[layer])[..., lo:hi] @ weight

def edges_from(layer, head):
    """Attribution of every edge out of one head, keyed by the reader it goes into."""
    delta = write_delta(layer, head)
    return {name: float((delta * reader_grads[k]).sum())
            for k, (name, _) in enumerate(readers) if k > 2 * layer}     # the readers after the head

for layer, head in [(5, 5), (8, 6)]:
    edges = edges_from(layer, head)
    largest = sorted(edges.items(), key=lambda item: -abs(item[1]))[:3]
    print(f"L{layer}H{head} -> " + "   ".join(f"{name} {value:+.3f}" for name, value in largest))
    assert abs(sum(edges.values()) - float(scores[layer, head])) < 1e-3  # the edges sum to the node score

random.seed(0)
null = [abs(random.choice(list(edges_from(*random.choice(sorted(every_head))).values()))) for _ in range(100)]
null_median = sorted(null)[len(null) // 2]
print(f"a random edge, median |attribution|: {null_median:.3f}")
assert edges_from(5, 5)["L8 attn"] > 20 * null_median
```

```
L5H5 -> L8 attn +1.140   L7 attn +0.530   L6 attn -0.240
L8H6 -> L10 attn +0.712   L9 attn +0.399   L11 attn +0.219
a random edge, median |attribution|: 0.005
```

Keep the sum check. With the gradients read off the shared residual stream
every reader gets the total, the edges overcount, and the identity fails. A
destination here is a whole layer's attention or MLP: GPT-2 computes queries,
keys and values in one module (`c_attn`), so a destination head has no input of
its own to copy.

Confirm an edge you report with a **path patch**: add the head's write delta to
one reader's input in the corrupt run, so every other reader still sees the
corrupt residual stream.

```python
def path_patch(layer, head, norm):
    """Metric change when one head's clean - corrupt write reaches one reader and no other."""
    delta = write_delta(layer, head)
    with torch.no_grad(), model.trace(corrupt):
        norm.input = norm.input + delta        # a new tensor: only this reader sees it
        patched = logit_diff(model.output.logits).save()
    return float(patched) - corrupt_metric

for layer, head, name, norm in [(8, 10, "logits", model.transformer.ln_f),
                                (5, 5, "L8 attn", model.transformer.h[8].ln_1)]:
    attributed, patched = edges_from(layer, head)[name], path_patch(layer, head, norm)
    print(f"L{layer}H{head} -> {name}: attribution {attributed:+.3f}, path patch {patched:+.3f}")
```

```
L8H10 -> logits: attribution +0.752, path patch +0.754
L5H5 -> L8 attn: attribution +1.140, path patch +1.014
```

Report the patched number. Assign a new tensor, as above: `norm.input += delta`
writes into the residual stream that every later reader shares. For L5H5 at its
own layer's MLP (`h[5].ln_2`) that returns +1.210, the effect of patching the
whole head, where the edge alone is +0.035.

## 5. Prune

A top-K is not minimal. Drop members one at a time while the set stays above a
threshold on the floor scale:

```python
def prune(order, keep_fraction=0.80):
    current = set(order)
    for member in order:
        if keep_only(current - {member}) >= floor + keep_fraction * span:
            current = current - {member}
    return current

by_score = sorted(circuit, key=lambda member: float(scores[member]))
for label, order in [("lowest score first", by_score), ("highest score first", by_score[::-1])]:
    pruned = prune(order)
    print(f"{label:<20}{sorted(pruned)}  {(keep_only(pruned) - floor) / span:.1%}")
    assert len(pruned) == 3
```

```
lowest score first  [(7, 9), (9, 9), (10, 0)]  87.4%
highest score first [(8, 10), (9, 9), (10, 0)]  89.1%
```

Ten heads prune to three, and the order decides which three. A pruned set is one
minimal set, enough to produce the behavior under this ablation, and not the
list of everything the model uses. Report it beside the full candidate set.

## 6. Completeness

Remove the set and see how much of what the ablation can take away is gone.
Faithfulness says the set is sufficient. This says whether the model depends on
it:

```python
for label, source in [("mean", head_means), ("resample from corrupt", corrupt_heads)]:
    lowest = ablate(every_head, source)                        # every head removed: all this ablation can take
    lost = {name: (clean_metric - ablate(removed, source)) / (clean_metric - lowest)
            for name, removed in [("circuit", circuit), ("random 10", control)]}
    print(f"{label:<23}lowest {lowest:+.3f}   circuit removed {lost['circuit']:6.1%}   random 10 removed {lost['random 10']:6.1%}")
    assert lost["circuit"] > lost["random 10"] + 0.3
```

```
mean                   lowest +0.069   circuit removed  38.9%   random 10 removed -11.3%
resample from corrupt  lowest -2.772   circuit removed  87.2%   random 10 removed   5.7%
```

Each number is the share of `clean − lowest` that the removal costs. Under mean
ablation the ten heads keep 83% of the metric alone, and removing them costs
only 39%: the other heads still carry most of the task. With the corrupt run's
values in the same ten heads the cost is 87%. A sufficient set is not yet the
mechanism, and the answer depends on the ablation, so state which one each
number is under ([ablation](ablation.md)).

The heads here are ablated at every position. To ask where a head acts, slice
positions as well as heads.
