---
name: ablation
description: Measure what a component contributes by removing it — zero ablation, mean ablation, resample ablation, and noise ablation applied to layers, attention heads, MLPs, neurons, and token positions. Use to test necessity ("does the model still work without this?"), to sweep every component for importance, or to knock out a candidate found by a probe or attention analysis. Covers why zero ablation systematically overstates importance, what to use instead, and how to batch a full sweep into one forward pass per layer.
---

# Ablation

Ablation asks the necessity question: **remove this component — does the behavior
survive?** It is the cheapest causal test available, and the easiest to get wrong,
because *what you replace the component with* determines what the number means.

<!-- test: setup -->
```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

prompt = "The Eiffel Tower is in the city of"
paris = model.tokenizer.encode(" Paris")[0]
n_layers = len(model.transformer.h)

with model.trace(prompt):
    baseline = model.output.logits[0, -1].log_softmax(-1)[paris].detach().save()

print(f"baseline log P(' Paris') = {float(baseline):.3f}")
```

## The four ablations

| Method | Replace with | Asks |
|---|---|---|
| **Zero** | `0` | what if this component wrote nothing? |
| **Mean** | its average over a dataset | what if this component were uninformative? |
| **Resample** | its value on a different input | what if this component saw something else? |
| **Noise** | activation + gaussian noise | how robust is the behavior to perturbation? |

**Zero ablation is the default choice and usually the wrong one.** Zero is not a
neutral value — it is far outside the distribution the rest of the network expects,
so the damage you measure mixes "this component mattered" with "the network was
pushed off-distribution". Mean and resample ablation stay in-distribution and give
the more honest answer. Use zero when you want an upper bound, and say so.

## Layer sweep — one forward pass per condition

Every ablation of a given kind fits in one batched trace:

```python
with model.trace() as tracer:
    zero_scores = nnsight.save([])
    for layer in range(n_layers):
        with tracer.invoke(prompt):
            model.transformer.h[layer].output[:, -1, :] = 0
            zero_scores.append(model.output.logits[0, -1].log_softmax(-1)[paris].detach())

for layer, score in enumerate(zero_scores):
    print(f"layer {layer:2d}  log P = {float(score):+.3f}   drop {float(baseline) - float(score):+.3f}")
```

## Mean ablation

Compute the mean over a corpus first, then substitute it. The mean must come from
a distribution that matches your task — a mean over unrelated text is a different
(and weaker) control.

```python
corpus = [
    "The Colosseum is in the city of",
    "The Statue of Liberty is in the city of",
    "Big Ben is in the city of",
    "The Brandenburg Gate is in the city of",
]

with model.trace(corpus):
    means = nnsight.save([block.output[:, -1, :].mean(0).detach() for block in model.transformer.h])

# the corpus is ragged (10, 9, 7 and 9 tokens), so every row is padded to 10 — and
# GPT-2 pads on the left, which is why `[:, -1, :]` still lands on each prompt's own
# last token. An absolute index like `[:, 4, :]`, or a right-padding tokenizer, would
# average pad activations here instead. Check `tokenizer.padding_side` before
# indexing a ragged batch by position.

with model.trace() as tracer:
    mean_scores = nnsight.save([])
    for layer in range(n_layers):
        with tracer.invoke(prompt):
            model.transformer.h[layer].output[:, -1, :] = means[layer]
            mean_scores.append(model.output.logits[0, -1].log_softmax(-1)[paris].detach())

print(f"{'layer':>6} {'zero':>9} {'mean':>9}")
for layer in range(n_layers):
    print(f"{layer:>6} {float(zero_scores[layer]):>+9.3f} {float(mean_scores[layer]):>+9.3f}")
```

Compare the two columns: where zero ablation reports a large drop and mean
ablation reports little, the component was not carrying task information — you
were measuring distribution shock.

## Resample ablation

Replace the activation with its value on a *different* input. This is the
strictest in-distribution control and the one circuit work generally uses.

```python
with model.trace("The Colosseum is in the city of"):
    donor = nnsight.save([block.output[:, -1, :].detach() for block in model.transformer.h])

with model.trace() as tracer:
    resample_scores = nnsight.save([])
    for layer in range(n_layers):
        with tracer.invoke(prompt):
            model.transformer.h[layer].output[:, -1, :] = donor[layer]
            resample_scores.append(model.output.logits[0, -1].log_softmax(-1)[paris].detach())

for layer in range(0, n_layers, 3):
    print(f"layer {layer:2d}  resample log P = {float(resample_scores[layer]):+.3f}")
```

Resample ablation with a paired prompt *is* activation patching in the noising
direction — see the `activation-patching` skill for the full treatment.

## Attention heads

Heads are contiguous slices of the attention output projection's input:

```python
n_heads = model.config.n_head
head_dim = model.config.n_embd // n_heads
LAYER = 5

with model.trace() as tracer:
    head_scores = nnsight.save([])
    for head in range(n_heads):
        with tracer.invoke(prompt):
            lo, hi = head * head_dim, (head + 1) * head_dim
            model.transformer.h[LAYER].attn.c_proj.input[:, :, lo:hi] = 0
            head_scores.append(model.output.logits[0, -1].log_softmax(-1)[paris].detach())

for head, score in enumerate(head_scores):
    print(f"L{LAYER}H{head:<2} drop {float(baseline) - float(score):+.3f}")
```

Scan all layers by nesting the loops — 144 head ablations on GPT-2 is 12 batched
traces.

## Sublayers, neurons, positions

```python
LAYER = 8

with model.trace() as tracer:
    with tracer.invoke(prompt):                                    # whole MLP
        model.transformer.h[LAYER].mlp.output[:] = 0
        no_mlp = model.output.logits[0, -1].log_softmax(-1)[paris].detach().save()

    with tracer.invoke(prompt):                                    # whole attention
        model.transformer.h[LAYER].attn.output[0][:] = 0
        no_attn = model.output.logits[0, -1].log_softmax(-1)[paris].detach().save()

    with tracer.invoke(prompt):                                    # one neuron
        model.transformer.h[LAYER].mlp.source.self_act_0.output[:, :, 100] = 0
        no_neuron = model.output.logits[0, -1].log_softmax(-1)[paris].detach().save()

    with tracer.invoke(prompt):                                    # one position, all layers
        for layer in range(n_layers):
            model.transformer.h[layer].output[:, 4, :] = 0
        no_position = model.output.logits[0, -1].log_softmax(-1)[paris].detach().save()

for name, value in [("mlp", no_mlp), ("attn", no_attn),
                    ("neuron 100", no_neuron), ("position 4", no_position)]:
    print(f"{name:<12} drop {float(baseline) - float(value):+.3f}")
```

Individual neurons rarely matter much on their own — that is the expected result,
not a bug, and it is the motivation for feature-level analysis (see the
`sae-and-dictionary-learning` skill).

## Skipping instead of zeroing

`module.skip(value)` bypasses a module's forward entirely. Skipping with the
module's own input turns a sublayer into a pass-through — the cleanest form of
"remove this computation" for a residual sublayer, since the residual stream keeps
flowing:

```python
with model.trace(prompt):
    mlp_in = model.transformer.h[8].mlp.input
    model.transformer.h[8].mlp.skip(torch.zeros_like(mlp_in))
    skipped = model.output.logits[0, -1].log_softmax(-1)[paris].detach().save()

print(f"mlp skipped: {float(skipped):+.3f}")
```

It also saves the compute, which matters when ablating many layers of a large
model.

## Interpreting results

**Report the drop against a baseline in the same trace.** Absolute log-probs are
not comparable across prompts.

**Sum of parts ≠ whole.** Ablating each of two components separately can show
little while ablating both destroys the behavior (redundancy), or the reverse
(interaction). If you care about a set, ablate the set.

**Ablation shows necessity, not sufficiency.** Pair it with the denoising
direction of `activation-patching` to get both.

**Small drops on small models are noise.** Establish a null distribution: ablate
random components and see how big the drop is when nothing should happen.

**Say which ablation you used.** "Head 5.3 is important" means nothing without
"under zero ablation" or "under resample ablation from prompts of type X" — the
same head can look critical under one and irrelevant under the other.

## Related skills

- `activation-patching` — the sufficiency direction, and paired-prompt design
- `attribution-patching` — approximate every ablation at once with gradients
- `attention-analysis` — deciding which heads are worth ablating
- `nnsight/docs/patterns/per-head-attention.md` — the three routes to a per-head view, and which one actually ablates a head
- `circuit-discovery` — ablating sets of components to isolate a circuit
- `nnsight` — batching, `skip`, `source` for neuron-level access
