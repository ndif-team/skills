---
name: ablation
description: Measure what a component contributes by removing it — one component at a time, no contrast prompt — with zero, mean, resample or noise ablation of layers, attention heads, MLPs, neurons and token positions. Use to test necessity ("does the model still work without this?"), to sweep every component for importance, or to knock out a candidate found by a probe or attention analysis. Covers why zero ablation overstates importance, what to use instead, and how to run a whole sweep — every head of every layer plus the baseline — as one batched trace.
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
n_heads = model.config.n_head
head_dim = model.transformer.h[0].attn._module.head_dim  # equals n_embd//n_head on GPT-2, not on GQA/Gemma models
```

## The four ablations

| Method | Replace with | Asks |
|---|---|---|
| **Zero** | `0` | what if this component wrote nothing? |
| **Mean** | its average over a dataset | what if this component were uninformative? |
| **Resample** | its value on a different input | what if this component saw something else? |
| **Noise** | activation + gaussian noise | how robust is the behavior to perturbation? |

**Zero ablation is the default choice and usually the wrong one.** Zero is not a
neutral value — it pushes activations off-distribution, so the damage you measure
mixes "this component mattered" with "the network was shocked". Layer-0 heads and
MLPs dominate naive zero sweeps for this reason (zeroing head L0H10 drops P(' Paris')
from 0.070 to 0.0059). Prefer mean or resample ablation for any claim; use zero as
an upper bound, and say so.

## Every head, one trace

Head `h` is the slice `[h*head_dim : (h+1)*head_dim]` of the attention output
projection's **input** — after `c_proj` the hidden dimension no longer decomposes
per head. Put the baseline and every (layer, head) ablation in their own invoke of
one trace, keyed into a saved dict:

```python
def head_sweep(site, value=lambda layer, lo, hi: 0):
    """P(' Paris'): a baseline invoke, then one invoke per (layer, head) with site(layer)[..., lo:hi] replaced."""
    with torch.no_grad(), model.trace() as tracer:
        p = nnsight.save({})
        with tracer.invoke(prompt):                                    # invoke 0: baseline
            p["baseline"] = model.output.logits[0, -1].softmax(-1)[paris]
        for layer in range(n_layers):
            for head in range(n_heads):
                with tracer.invoke(prompt):
                    lo, hi = head * head_dim, (head + 1) * head_dim
                    site(layer)[:, :, lo:hi] = value(layer, lo, hi)
                    p[layer, head] = model.output.logits[0, -1].softmax(-1)[paris]
    return p

def top(p, k=3):                                                       # largest |change| first
    return sorted((key for key in p if key != "baseline"), key=lambda key: -abs(float(p[key] - p["baseline"])))[:k]

c_proj_in = lambda layer: model.transformer.h[layer].attn.c_proj.input
p_paris = head_sweep(c_proj_in)

baseline = float(p_paris["baseline"])
ranked = sorted((float(p), key) for key, p in p_paris.items() if key != "baseline")
print(f"baseline P(' Paris') = {baseline:.3f}")
for p, (layer, head) in ranked[:3] + ranked[-1:]:
    print(f"L{layer}H{head:<2} -> {p:.4f}")

assert abs(baseline - 0.070) < 1e-3
assert abs(float(p_paris[0, 10]) - 0.0059) < 1e-3
assert float(p_paris[0, 0]) > baseline          # ablating L0H0 *raises* P(' Paris')
assert top(p_paris) == [(0, 10), (0, 9), (0, 7)]
```

The 145 invokes batch into one forward pass — an order of magnitude faster than a
loop of 144 single-invoke traces on CPU (0.05–0.12 s against 0.8–3.1 s); the
setup's `dispatch=True` lands on the GPU when one is present, where the gap is ~3x.
An ablation can also *raise* the metric (L0H0 above) — report the signed change,
not its absolute value.

**The shortcut that looks the same and is not.** `attn.output[0]` is *after*
`c_proj`, so slicing it cuts 64 mixed hidden dimensions, not a head — and it
ranks entirely different "heads":

```python
wrong = head_sweep(lambda layer: model.transformer.h[layer].attn.output[0])   # WRONG site
assert top(wrong) == [(0, 4), (0, 5), (0, 2)]                                  # vs L0H10, L0H9, L0H7
```

## Zero, mean and resample on one component

The same sweep shape compares replacement values directly. This one targets each
layer's MLP output at the last position; `attn.output[0]` works the same way:

```python
def mlp_sweep(replacement):
    """P(' Paris') with each layer's MLP output at the last position replaced."""
    with torch.no_grad(), model.trace() as tracer:
        scores = nnsight.save({})
        with tracer.invoke(prompt):
            scores["baseline"] = model.output.logits[0, -1].softmax(-1)[paris]
        for layer in range(n_layers):
            with tracer.invoke(prompt):
                model.transformer.h[layer].mlp.output[:, -1, :] = replacement(layer)
                scores[layer] = model.output.logits[0, -1].softmax(-1)[paris]
    return scores

# mean: over prompts of the same task, not unrelated text
corpus = [
    "The Colosseum is in the city of",
    "The Statue of Liberty is in the city of",
    "Big Ben is in the city of",
    "The Brandenburg Gate is in the city of",
]
with torch.no_grad(), model.trace(corpus):   # ragged batch; GPT-2 pads left, so -1 is each prompt's last token
    means = nnsight.save([block.mlp.output[:, -1, :].mean(0) for block in model.transformer.h])

# resample: the value on a different input
with torch.no_grad(), model.trace("The Colosseum is in the city of"):
    donor = nnsight.save([block.mlp.output[:, -1, :] for block in model.transformer.h])

zero = mlp_sweep(lambda layer: 0)
mean = mlp_sweep(lambda layer: means[layer])
resample = mlp_sweep(lambda layer: donor[layer])

print(f"baseline {float(zero['baseline']):.4f}")
print(f"{'layer':>6} {'zero':>8} {'mean':>8} {'resample':>9}")
for layer in range(n_layers):
    print(f"{layer:>6} {float(zero[layer]):>8.4f} {float(mean[layer]):>8.4f} {float(resample[layer]):>9.4f}")

assert float(zero[0]) < 0.01 and float(mean[0]) > 0.06   # MLP0: zero shocks, mean is harmless
```

Where zero reports a large drop and mean reports little — MLP0 here, 0.0012 against
0.0701 — the component was not carrying task information; you were measuring
distribution shock. Resample with a paired prompt *is* activation patching in the
noising direction; the `activation-patching` skill covers prompt design.

Per head, mean ablation writes the head's `c_proj.input` slice averaged over the
same-task corpus, and reorders the zero-ablation ranking:

```python
with torch.no_grad(), model.trace(corpus):     # every position, pads included; masking them keeps the ranking
    head_means = nnsight.save([block.attn.c_proj.input.mean((0, 1)) for block in model.transformer.h])
p_mean = head_sweep(c_proj_in, lambda layer, lo, hi: head_means[layer][lo:hi])
change = {key: float(p - p_mean["baseline"]) for key, p in p_mean.items() if key != "baseline"}
assert top(p_mean, 1) == [(9, 8)] and abs(change[9, 8] + 0.030) < 2e-3   # L9H8 is now #1
assert abs(change[0, 10] + 0.0095) < 1e-3                                 # L0H10: -0.064 under zero
```

The concrete case for "zero overstates": L0H10 loses 85% of its effect once the
head writes a typical value instead of nothing, and a buried layer-9 head tops it.

Assigning to `model.transformer.h[layer].output` instead replaces the **whole
residual stream** — everything every earlier component wrote — so it asks whether
the information present at that layer is needed downstream, not what block `layer`
contributes. Label it as residual-stream ablation when you use it.

## Sublayers, neurons, positions

```python
LAYER = 8

with torch.no_grad(), model.trace() as tracer:
    with tracer.invoke(prompt):                                    # baseline
        base = model.output.logits[0, -1].softmax(-1)[paris].save()

    with tracer.invoke(prompt):                                    # whole MLP
        model.transformer.h[LAYER].mlp.output[:] = 0
        no_mlp = model.output.logits[0, -1].softmax(-1)[paris].save()

    with tracer.invoke(prompt):                                    # whole attention
        model.transformer.h[LAYER].attn.output[0][:] = 0
        no_attn = model.output.logits[0, -1].softmax(-1)[paris].save()

    with tracer.invoke(prompt):                                    # one neuron
        model.transformer.h[LAYER].mlp.act.output[:, :, 100] = 0
        no_neuron = model.output.logits[0, -1].softmax(-1)[paris].save()

    with tracer.invoke(prompt):                                    # one position, residual stream, all layers
        for layer in range(n_layers):
            model.transformer.h[layer].output[:, 4, :] = 0
        no_position = model.output.logits[0, -1].softmax(-1)[paris].save()

for name, value in [("mlp", no_mlp), ("attn", no_attn),
                    ("neuron 100", no_neuron), ("position 4", no_position)]:
    print(f"{name:<12} change {float(value) - float(base):+.4f}")
```

Individual neurons rarely matter much on their own — that is the expected result,
not a bug, and the motivation for feature-level analysis (the
`sae-and-dictionary-learning` skill).

## Skipping instead of zeroing

`module.skip(value)` bypasses a module's forward and uses `value` as its output.
For a residual sublayer, zero *is* the pass-through value — the block adds the
sublayer's output to the residual stream — so skipping with zeros equals zero
ablation and saves the sublayer's compute:

```python
with torch.no_grad(), model.trace(prompt):
    mlp_in = model.transformer.h[LAYER].mlp.input
    model.transformer.h[LAYER].mlp.skip(torch.zeros_like(mlp_in))
    skipped = model.output.logits[0, -1].softmax(-1)[paris].save()

assert abs(float(skipped) - float(no_mlp)) < 1e-5   # same value; batching adds float noise
```

## Interpreting results

**Report the change against the baseline invoke of the same trace.** Absolute
probabilities are not comparable across prompts.

**Sum of parts ≠ whole.** Two components can each show little while ablating both
destroys the behavior (redundancy), or the reverse (interaction). If you care
about a set, ablate the set.

**Ablation shows necessity, not sufficiency.** Pair it with the denoising
direction of `activation-patching`.

**Small changes on small models are noise.** Ablate random components to see how
big the change is when nothing should happen.

**Say which ablation you used.** "Head 5.3 is important" means nothing without
"under zero ablation" or "under resample ablation from prompts of type X" — the
same head can look critical under one and irrelevant under the other.

## Related skills

- `activation-patching` — the sufficiency direction, and paired-prompt design
- `attribution-patching` — approximate every ablation at once with gradients
- `attention-analysis` — deciding which heads are worth ablating
- `circuit-discovery` — ablating sets of components to isolate a circuit
- `nnsight` — batching, `skip`, `source`
