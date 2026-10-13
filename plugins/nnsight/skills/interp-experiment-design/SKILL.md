---
name: interp-experiment-design
description: A planning checklist and router for interpretability experiments, not a technique. Use to plan a study, when asking "is this result real?" or "is this too good to be true?", to decide which method answers a question (logit lens, probing, patching, causal tracing, ablation, attribution, circuits, SAEs, steering, editing), or to choose what controls and which metric an experiment needs. Covers logit-difference metrics, the no-op and extreme-intervention wiring checks, controls and baselines, minimal pairs, the correlational-to-causal ladder, and statistical hygiene.
---

# Designing an Interpretability Experiment

Most wrong interpretability results are not bugs in the intervention code. They
come from a metric that measures the wrong thing, a missing control, or a claim
one step stronger than the evidence. This skill is the checklist; the technique
skills it routes to hold the code.

<!-- test: setup -->
```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

clean = "The Eiffel Tower is in the city of"
corrupt = "The Colosseum is in the city of"
paris = model.tokenizer.encode(" Paris")[0]
rome = model.tokenizer.encode(" Rome")[0]
```

## 1. Pick the question, then the technique

| Question | Technique |
|---|---|
| Where does the answer become readable? | `logit-lens` |
| Is this information present at all? | `probing` |
| Which component carries it? | `activation-patching` |
| Where is a fact stored (restore one state into a noised run)? | `causal-tracing` |
| Is this component necessary? | `ablation` |
| Which components, at scale? | `attribution-patching` → verify |
| Which subgraph implements the task? | `circuit-discovery` |
| What does this head attend to? | `attention-analysis` |
| Which interpretable features are active? | `sae-and-dictionary-learning` |
| Can I control the behavior? | `model-steering` |
| Can I change what it knows? | `model-editing-and-lora` |

The ladder: lens and probes are **correlational**, patching and ablation are
**causal**, steering and editing are **manipulative**. Each rung supports a
stronger claim than the one below, and none supports the rung above.

## 2. Build a metric that isolates the effect

**Prefer a logit difference over a probability.** `P(answer)` mixes "the right
answer got likelier" with "everything got likelier".

```python
with model.trace() as tracer:
    with tracer.invoke(clean):
        logits = model.output.logits[0, -1]
        clean_metric = (logits[paris] - logits[rome]).detach().save()
    with tracer.invoke(corrupt):
        logits = model.output.logits[0, -1]
        corrupt_metric = (logits[paris] - logits[rome]).detach().save()

span = float(clean_metric) - float(corrupt_metric)
print(f"clean {float(clean_metric):+.3f}  corrupt {float(corrupt_metric):+.3f}  span {span:.3f}")
```

Those endpoints turn every later number into a **fraction of the gap recovered**,
comparable across prompts, layers, and models. Avoid saturating metrics:
probabilities near 0 or 1 have vanishing gradients, which silently breaks
`attribution-patching`.

## 3. Sanity-check the wiring before trusting any result

**A no-op intervention must do nothing.** Patch a value with itself:

```python
with model.trace(clean):
    donor = model.transformer.h[6].output.detach().save()

with model.trace(clean):
    model.transformer.h[6].output[:] = donor          # identical values
    logits = model.output.logits[0, -1]
    noop = (logits[paris] - logits[rome]).detach().save()

print(f"no-op patch: {float(noop):+.3f}  (should equal clean {float(clean_metric):+.3f})")
```

If that moves, your indexing, positions, or donor source is wrong.

**An extreme intervention must do something.** If zeroing a whole layer leaves the
metric unchanged, the intervention is not landing:

```python
with model.trace(clean):
    model.transformer.h[6].output[:] = 0
    logits = model.output.logits[0, -1]
    destroyed = (logits[paris] - logits[rome]).detach().save()

print(f"layer zeroed: {float(destroyed):+.3f}  (should be far from {float(clean_metric):+.3f})")
```

Make the extreme value zero or a real tensor, never a large constant: every block
reads the residual through a LayerNorm, which subtracts the mean, so a constant
added to every channel never arrives — and `mlp.output[:] = 1000` looks like no
intervention at all:

```python
with model.trace() as tracer:
    probes = nnsight.save([])
    for value in (0, 10000):
        with tracer.invoke(clean):
            model.transformer.h[6].output[:] += value
            probes.append(model.output.logits[0, -1].log_softmax(-1)[paris].detach())

assert abs(float(probes[1]) - float(probes[0])) < 1e-2    # +10000 everywhere: no effect
```

**The unmodified baseline must reproduce the known behavior** — in the same trace
as the intervention, not from memory of a previous session.

## 4. Controls

An effect is only evidence if the null does worse:

| Claim | Control |
|---|---|
| this component matters | the same intervention on a random component |
| this direction encodes X | a random direction of the same norm |
| this circuit is the mechanism | a random subgraph of the same size, **and the empty circuit** |
| this probe found a representation | shuffled labels; a probe at layer 0 |
| this steering vector works | the negated vector; a random vector |

```python
generator = torch.Generator(device=model.device).manual_seed(0)

with model.trace(clean):
    resid_norm = model.transformer.h[6].output[0, -1].norm().detach().save()

random_dir = torch.randn(768, generator=generator, device=model.device)
random_dir = random_dir / random_dir.norm()

with model.trace(clean):
    model.transformer.h[6].output[:, -1, :] += 0.5 * float(resid_norm) * random_dir
    logits = model.output.logits[0, -1]
    random_effect = (logits[paris] - logits[rome]).detach().save()

print(f"random direction moved the metric by {float(random_effect) - float(clean_metric):+.3f}")
```

Whatever a random direction achieves is the floor your real direction must clear.

**Find the floor before you normalise to it.** `(value − baseline) / (clean −
baseline)` means something only if the intervention can reach `baseline`.
Ablating attention heads cannot reach a corrupt prompt's metric — the MLPs and
residual stream are untouched — so the empty circuit is the real zero. Without it
a circuit of ten irrelevant heads can score 85% (the `circuit-discovery` skill
measures exactly that).

## 5. Prompts: one is an anecdote

- Use a **set** of prompts sharing the structure under test, and a held-out set
  the analysis never saw.
- Use **minimal pairs**: identical token count and structure, differing only in
  the variable. Assert the lengths match.
- Vary surface form (paraphrase, reorder, change names) so the finding is about
  the task, not the string.

The alignment assert catches a pair that looks fine and is not:

<!-- test: expect-error AssertionError -->
```python
pairs = [
    ("The Eiffel Tower is in the city of", "The Colosseum is in the city of"),
    ("The Statue of Liberty is in the city of", "Big Ben is in the city of"),
]
for a, b in pairs:
    assert len(model.tokenizer(a).input_ids) == len(model.tokenizer(b).input_ids), (a, b)
```

Those two tokenize to different lengths, so position `i` is a different thing in
each, and nothing else would have raised. Filter candidate pairs on the same test
rather than trusting the set.

## 6. Statistical hygiene

- **Multiple comparisons.** Scoring 144 heads and reporting the top one means the
  top one is partly noise. Report the distribution, hold out, or correct.
- **Seeds** wherever noise is injected or parameters initialized (causal tracing,
  probes, DAS, SAEs). If the answer changes with the seed, it is not an answer.
- **Effect sizes over binaries.** "Recovers 60% of the gap", not "patching worked".
- **State the search space** and any truncation behind a top-K list.

## 7. Compute budget

Sweep in invokes: N variants of one intervention are N `tracer.invoke`s in one
forward pass, so a 144-condition head scan is 12 batched traces, not 144. The
`nnsight` skill has the rest of the primitives (caching, `tracer.stop()`), and the
`remote` skill covers models too large to host.

## 8. Before you claim it

- No-op and extreme checks pass; the baseline ran in the same trace.
- A control did worse; the effect is a fraction of a known span.
- More than one prompt, plus a held-out set; search space stated.
- Ablation type, patch direction, and corrupt distribution stated.
- The claim matches the rung: readable ≠ used, sufficient ≠ necessary,
  correlated ≠ causal.
- Negative results reported too.

## Common ways experiments go wrong

- **Measuring the intervention instead of the model.** Large interventions push
  the network off-distribution; prefer mean/resample over zero where you can.
- **Confirming the hypothesis you started with.** Decide in advance what result
  would falsify the claim, and check for it.
- **Reading a heatmap as a mechanism.** A bright cell says an intervention there
  changed the output, not what the component computes.
- **Trusting an unvalidated approximation.** Attribution patching, probes, and SAEs
  are models of the model; validate against ground truth on a subset.
- **Generalizing across scale.** Where something lives rarely transfers across
  model families or sizes without re-running.

## Related skills

- `nnsight` — the primitives every experiment is built from
- `activation-patching`, `causal-tracing`, `ablation`, `probing`,
  `circuit-discovery`, `sae-and-dictionary-learning` — techniques, each with its
  own controls
- `debugging` — when the code, rather than the design, is the problem
