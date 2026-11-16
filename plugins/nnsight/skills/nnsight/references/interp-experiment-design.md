# Designing an interpretability experiment

A checklist for before and after the technique guides: which technique answers
the question, a metric, three wiring checks, a control, aligned prompts, and what
the result lets you claim.

```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True)

clean = "The Eiffel Tower is in the city of"
corrupt = "The Colosseum is in the city of"
paris = model.tokenizer.encode(" Paris", add_special_tokens=False)[0]
rome = model.tokenizer.encode(" Rome", add_special_tokens=False)[0]
```

## 1. The question picks the technique

| Question | Guide |
|---|---|
| At which layer is the answer readable? | [logit-lens](logit-lens.md) |
| Is this information decodable from the activations at all? | [probing](probing.md) |
| Which layer, position or head carries it, given a contrast prompt? | [activation-patching](activation-patching.md) |
| Where is a fact stored, from one prompt? | [causal-tracing](causal-tracing.md) |
| Is this component necessary? | [ablation](ablation.md) |
| Which of very many components matter? | [attribution-patching](attribution-patching.md), then a real patch on the top ones |
| Which set of components does the task? | [circuit-discovery](circuit-discovery.md) |
| What does this head attend to? | [attention-analysis](attention-analysis.md) |
| Which interpretable features are active? | [sae-and-dictionary-learning](sae-and-dictionary-learning.md) |
| Can the behavior be controlled at inference? | [model-steering](model-steering.md) |
| Can what the model knows be changed? | [model-editing-and-lora](model-editing-and-lora.md) |

A lens, a probe and an attention pattern show that information is readable.
Patching and ablation show that the model uses it. Steering and editing show
that it can be moved. None of the three supports the claim of the next.

## 2. A metric with two endpoints

Use a logit difference between the answer and its alternative, not a
probability. `P(answer)` also moves when probability shifts among tokens that
have nothing to do with the contrast, and near 0 or 1 its gradient vanishes,
which zeroes every score of [attribution-patching](attribution-patching.md).

```python
def logit_diff():
    logits = model.output.logits[0, -1]
    return logits[paris] - logits[rome]

with torch.no_grad(), model.trace() as tracer:
    with tracer.invoke(clean):
        donor = model.transformer.h[6].output.save()        # for the no-op check below
        clean_metric = logit_diff().save()
    with tracer.invoke(corrupt):
        corrupt_metric = logit_diff().save()

span = float(clean_metric) - float(corrupt_metric)
print(f"clean {float(clean_metric):+.3f}   corrupt {float(corrupt_metric):+.3f}   span {span:.3f}")
```

```
clean +2.425   corrupt -1.487   span 3.912
```

Report every later number as a fraction of that span, `(value - corrupt) /
span`, which compares across prompts, layers and models.

## 3. Three wiring checks

```python
with torch.no_grad(), model.trace() as tracer:
    check = nnsight.save({})
    with tracer.invoke(clean):                               # 1. the baseline, in this trace
        check["baseline"] = logit_diff()
    with tracer.invoke(clean):                               # 2. a no-op: patch a value with itself
        model.transformer.h[6].output[:] = donor
        check["no-op"] = logit_diff()
    with tracer.invoke(clean):                               # 3. an extreme: zero the layer
        model.transformer.h[6].output[:] = 0
        check["zeroed"] = logit_diff()

check = {name: float(value) for name, value in check.items()}
print("   ".join(f"{name} {value:+.3f}" for name, value in check.items()))
assert abs(check["baseline"] - float(clean_metric)) < 1e-3
assert abs(check["no-op"] - check["baseline"]) < 1e-3
assert abs(check["zeroed"] - check["baseline"]) > 0.5 * span
```

```
baseline +2.425   no-op +2.425   zeroed +0.084
```

1. **The unmodified baseline runs in the same trace as the intervention** and
   reproduces the known behavior. A number carried over from another run is not
   a baseline.
2. **A no-op must change nothing.** When patching a value with itself moves the
   metric, the indexing, the position or the donor is wrong.
3. **An extreme must change something.** When zeroing a whole layer leaves the
   metric where it was, the write is not landing.

Make the extreme zero or a real activation, never a large constant. Every block
reads the residual stream through a LayerNorm, which subtracts the mean over
channels, so a constant added to every channel never arrives:

```python
with torch.no_grad(), model.trace() as tracer:
    shifted = nnsight.save({})
    for constant in (0, 10000):
        with tracer.invoke(clean):
            model.transformer.h[6].output[:] += constant
            shifted[constant] = model.output.logits[0, -1].log_softmax(-1)[paris]

assert abs(float(shifted[10000]) - float(shifted[0])) < 1e-2     # +10000 everywhere: no effect
```

## 4. A control that does worse

| Claim | Control |
|---|---|
| this component matters | the same intervention on a random component |
| this direction encodes X | a random direction of the same norm |
| this set is the circuit | a random set of the same size, and the empty set |
| this probe found a representation | shuffled labels; a probe at layer 0; held-out words, not held-out rows |
| this steering vector works | the negated vector; a random vector |

```python
with torch.no_grad(), model.trace(clean):
    resid_norm = model.transformer.h[6].output[0, -1].norm().save()

generator = torch.Generator().manual_seed(0)
random_dir = torch.randn(model.config.n_embd, generator=generator).to(resid_norm.device)
random_dir = random_dir / random_dir.norm()

with torch.no_grad(), model.trace(clean):
    model.transformer.h[6].output[:, -1, :] += 0.5 * resid_norm * random_dir
    random_metric = logit_diff().save()

print(f"random direction at half the residual norm: {float(random_metric) - float(clean_metric):+.3f}")
```

```
random direction at half the residual norm: -1.061
```

What the random direction moves, here a quarter of the span, is the floor a real
direction has to clear.

**Normalise only to a value the intervention can reach.** `(value - floor) /
(clean - floor)` means something when the intervention can produce `floor`.
Mean-ablating attention heads cannot reach the corrupt prompt's metric, so the
zero of that scale is the metric with every head ablated:
[circuit-discovery](circuit-discovery.md) measures the difference.

## 5. Prompts

- A set of prompts that share the structure under test, and a held-out set the
  analysis never saw. One prompt is an anecdote.
- Minimal pairs: the same token count and structure, differing only in the
  variable. Cross the variable with its nearest confound so neither predicts
  the other.
- Vary the surface form (names, order, paraphrase) so the finding is about the
  task and not the string.

Filter pairs on token count. A pair of different lengths raises nothing, and
position `i` is a different thing in its two prompts:

```python
candidates = [
    ("The Eiffel Tower is in the city of", "The Colosseum is in the city of"),
    ("The Statue of Liberty is in the city of", "Big Ben is in the city of"),
]
n_tokens = lambda text: len(model.tokenizer(text).input_ids)
pairs = [(a, b) for a, b in candidates if n_tokens(a) == n_tokens(b)]

assert pairs == candidates[:1]                 # the second pair is 9 tokens against 7
```

## 6. Readable here is not used here

A probe's accuracy at a position does not say that the model reads the
information from that position. On GPT-2 subject-verb agreement ("The keys near
the cabinet" is continued with ` are`), a difference-of-means probe on the last
token classifies held-out subject number at 0.71 to 0.79 in layers 2 to 5, where
a random direction scores about 0.51. Patching the last token at those layers recovers
at most 0.07 of the logit gap, and patching the subject token recovers 0.47 to
0.94. Recovery at the last token reaches one half only at layer 7.

Pair every claim that a site carries something with a patch, or a swap along the
probe direction, at the same layer and position: [probing](probing.md).

## 7. Before you claim it

- The three wiring checks pass.
- A control did worse, and the effect is a fraction of a span the intervention
  can reach.
- More than one prompt, a held-out set, and pairs of equal length.
- A seed wherever noise is drawn or parameters are initialised (causal tracing,
  probes, DAS, SAEs), and the answer holds across seeds.
- The search space is stated. The top one of 144 scored heads is partly
  selection: confirm it on prompts the ranking did not see.
- An approximation (attribution patching, a probe, an SAE) is checked against a
  real intervention on a subset.
- The ablation type, the patch direction and the corrupt distribution are
  stated.
- The claim matches the evidence: readable is not used, sufficient is not
  necessary, and a bright cell of a heatmap says an intervention there changed
  the output, not what the component computes.
- The result that would have falsified the claim was written down before the
  run.
- Negative results are reported.
