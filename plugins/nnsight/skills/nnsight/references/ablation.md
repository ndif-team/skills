# Ablation

Replace a component's output and measure what the behavior loses. It needs no
contrast prompt, and what the output is replaced with decides what the number
means:

| Ablation | The output becomes | Asks |
|---|---|---|
| zero | `0` | what if the component wrote nothing? |
| mean | its average over prompts of the same task | what if it carried nothing about this prompt? |
| resample | its value on another prompt | what if it had seen that prompt instead? |
| noise | itself plus gaussian noise | how much perturbation does the behavior survive? |

Zero is not a neutral value. It puts the network off-distribution, so the drop
mixes "this component mattered" with "the network was shocked". Make claims from
mean or resample ablation; the table under
[Mean and resample](#mean-and-resample) is the evidence. Resample from a paired
prompt is the noising direction of [activation-patching](activation-patching.md),
and noise on the subject's embeddings is [causal-tracing](causal-tracing.md).

```python
import statistics
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True)

prompt = "The Eiffel Tower is in the city of"
paris = model.tokenizer.encode(" Paris", add_special_tokens=False)[0]
n_layers = len(model.transformer.h)
n_heads = model.config.n_head
head_dim = model.transformer.h[0].attn._module.head_dim

def p_paris():
    return model.output.logits[0, -1].softmax(-1)[paris]
```

## Every head in one trace

Head `h` is columns `h * head_dim` to `(h + 1) * head_dim` of the output
projection's input, `attn.c_proj.input`
([access-and-modify](access-and-modify.md)). The baseline and one invoke per
(layer, head) make one forward pass:

```python
def head_sweep(site, value=lambda layer, lo, hi: 0):
    """Change in P(' Paris') per (layer, head) when site(layer)[:, :, lo:hi] is replaced."""
    with torch.no_grad(), model.trace() as tracer:
        p = nnsight.save({})
        with tracer.invoke(prompt):                                # the baseline, in the same trace
            p["baseline"] = p_paris()
        for layer in range(n_layers):
            for head in range(n_heads):
                with tracer.invoke(prompt):
                    lo, hi = head * head_dim, (head + 1) * head_dim
                    site(layer)[:, :, lo:hi] = value(layer, lo, hi)
                    p[layer, head] = p_paris()
    baseline = float(p.pop("baseline"))
    return {key: float(ablated) - baseline for key, ablated in p.items()}, baseline

def top(change, k=3):                                              # largest |change| first
    return sorted(change, key=lambda key: -abs(change[key]))[:k]

c_proj_in = lambda layer: model.transformer.h[layer].attn.c_proj.input
zero, baseline = head_sweep(c_proj_in)

assert abs(baseline - 0.070) < 1e-3
assert top(zero) == [(0, 10), (0, 9), (0, 7)]
assert zero[0, 0] > 0.04                       # zeroing L0H0 raises P(' Paris')
```

`attn.output[0]` is past the projection, so the same slice of it is 64 mixed
hidden dimensions and not a head. It raises nothing and ranks other "heads":

```python
wrong, _ = head_sweep(lambda layer: model.transformer.h[layer].attn.output[0])   # WRONG site
assert top(wrong) == [(0, 4), (0, 5), (0, 2)]                                     # not L0H10, L0H9, L0H7
```

## Mean and resample

The replacement comes from a run of its own. Average over other prompts of the
same task, not over unrelated text and not over the prompt being ablated, and
over real tokens only when the batch is padded:

```python
corpus = [
    "The Colosseum is in the city of",
    "The Statue of Liberty is in the city of",
    "Big Ben is in the city of",
    "The Brandenburg Gate is in the city of",
]
with torch.no_grad(), model.trace(corpus):
    real = model.inputs[1]["attention_mask"].bool()            # (prompts, tokens), False on padding
    head_means = nnsight.save([block.attn.c_proj.input[real].mean(0) for block in model.transformer.h])

with torch.no_grad(), model.trace(corpus[0]):                  # as many tokens as `prompt`: positions align
    donor = nnsight.save([block.attn.c_proj.input for block in model.transformer.h])

mean, _ = head_sweep(c_proj_in, lambda layer, lo, hi: head_means[layer][lo:hi])
resample, _ = head_sweep(c_proj_in, lambda layer, lo, hi: donor[layer][:, :, lo:hi])

print(f"{'':<10}{'three largest changes':<45}{'L0H10':>8}{'median |change|':>18}")
for name, change in [("zero", zero), ("mean", mean), ("resample", resample)]:
    largest = "   ".join(f"L{layer}H{head} {change[layer, head]:+.3f}" for layer, head in top(change))
    median = statistics.median(abs(c) for c in change.values())
    print(f"{name:<10}{largest:<45}{change[0, 10]:>+8.3f}{median:>18.4f}")

assert top(mean, 1) == top(resample, 1) == [(9, 8)]
assert abs(mean[0, 10]) < 0.25 * abs(zero[0, 10])
```

```
          three largest changes                           L0H10   median |change|
zero      L0H10 -0.064   L0H9 -0.057   L0H7 -0.044       -0.064            0.0038
mean      L9H8 -0.029   L3H3 -0.022   L0H7 -0.017        -0.012            0.0021
resample  L9H8 -0.029   L8H11 -0.021   L6H4 -0.021       -0.003            0.0021
```

Under zero the three largest effects are all in layer 0. Once L0H10 writes a
typical value in place of nothing, four fifths of its effect is gone and a
layer-9 head leads.

## Other sites

```python
LAYER = 8

with torch.no_grad(), model.trace() as tracer:
    p = nnsight.save({})
    with tracer.invoke(prompt):
        p["baseline"] = p_paris()
    with tracer.invoke(prompt):                                    # a whole MLP
        model.transformer.h[LAYER].mlp.output[:] = 0
        p["mlp"] = p_paris()
    with tracer.invoke(prompt):                                    # a whole attention layer (a tuple output)
        model.transformer.h[LAYER].attn.output[0][:] = 0
        p["attn"] = p_paris()
    with tracer.invoke(prompt):                                    # one MLP neuron
        model.transformer.h[LAYER].mlp.act.output[:, :, 100] = 0
        p["neuron 100"] = p_paris()
    with tracer.invoke(prompt):                                    # one token position, at every layer
        for layer in range(n_layers):
            model.transformer.h[layer].output[:, 4, :] = 0
        p["position 4"] = p_paris()
    with tracer.invoke(prompt):                                    # a set of heads, together
        for layer, head in sorted(top(zero)):
            model.transformer.h[layer].attn.c_proj.input[:, :, head * head_dim:(head + 1) * head_dim] = 0
        p["top 3 heads"] = p_paris()
    with tracer.invoke(prompt):                                    # the residual stream, not block 8
        model.transformer.h[LAYER].output[:] = 0
        p["residual"] = p_paris()

change = {name: float(ablated) - float(p["baseline"]) for name, ablated in p.items() if name != "baseline"}
for name, c in change.items():
    print(f"{name:<12}{c:+.4f}")

assert abs(change["residual"] + baseline) < 1e-6                   # nothing of the answer is left
assert abs(change["top 3 heads"]) < 0.5 * abs(sum(zero[key] for key in top(zero)))
```

```
mlp         -0.0003
attn        -0.0264
neuron 100  +0.0001
position 4  -0.0324
top 3 heads -0.0661
residual    -0.0700
```

- **`h[LAYER].output` is the residual stream**, everything every earlier block
  wrote. Replacing it asks whether what has been computed by that layer is needed
  downstream, not what block `LAYER` contributes. A block's own contribution is
  its `attn` and `mlp` outputs.
- To intervene on SAE features in place of neurons:
  [sae-and-dictionary-learning](sae-and-dictionary-learning.md).
- `mlp.skip(torch.zeros_like(mlp.input))` gives the same number as zeroing
  `mlp.output` without running the module, but a skip has to cover every invoke
  of a trace, so it cannot sit beside a baseline invoke:
  [control-flow](control-flow.md).

## Reading the numbers

- **Report the signed change against the baseline invoke of the same trace.** An
  ablation can raise the metric: zeroing L0H0 adds 0.041.
- **Name the ablation with the result.** L0H10 is −0.064 under zero, −0.012 under
  mean and −0.003 under resample.
- **The sweep is its own null.** Half the heads move the metric by more than
  0.002 under mean ablation, so a change of 0.004 is ordinary and L9H8's 0.029
  is not.
- **A set is not the sum of its members.** Zeroing the three top heads together
  costs 0.066; their separate effects sum to 0.165, more than the baseline
  probability. Ablate the set the claim is about.
- **Ablation tests necessity, not sufficiency.** For sufficiency use the
  denoising direction of [activation-patching](activation-patching.md). To rank
  the components of a large model before ablating any, use
  [attribution-patching](attribution-patching.md); to validate a set as a
  circuit, [circuit-discovery](circuit-discovery.md).
