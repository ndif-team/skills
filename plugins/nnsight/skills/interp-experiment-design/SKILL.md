---
name: interp-experiment-design
description: Design an interpretability experiment that supports the claim it is meant to support — choosing a metric, building prompt sets and minimal pairs, the controls and baselines that separate a finding from an artifact, sanity checks that catch wiring bugs before they become results, the correlational-to-causal ladder, statistical hygiene, and compute budgeting. Use before writing intervention code, when interpreting a result that looks too good, or when deciding which technique answers a given question.
---

# Designing an Interpretability Experiment

Most wrong interpretability results are not bugs in the intervention code. They
come from a metric that measures the wrong thing, a missing control, or a claim
one step stronger than the evidence. This skill is the checklist.

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
| Is this component necessary? | `ablation` |
| Which components, at scale? | `attribution-patching` → verify |
| Which subgraph implements the task? | `circuit-discovery` |
| What does this head attend to? | `attention-analysis` |
| Can I control the behavior? | `model-steering` |
| Can I change what it knows? | `model-editing-and-lora` |

Notice the ladder: lens and probes are **correlational**, patching and ablation
are **causal**, steering and editing are **manipulative**. Each rung supports a
stronger claim than the one below, and none of them supports the rung above.

## 2. Build a metric that isolates the effect

A metric must move when the thing you care about moves and stay put otherwise.

**Prefer a logit difference over a probability.** `P(answer)` mixes "the right
answer got likelier" with "everything got likelier" — an intervention that
flattens the distribution changes it for uninteresting reasons.

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
which is comparable across prompts, layers, and models. Raw logits are not.

Avoid metrics that saturate: probabilities near 0 or 1 have vanishing gradients,
which silently breaks any gradient-based method (see `attribution-patching`).

## 3. Sanity-check the wiring before trusting any result

Three checks, each one line of extra code, each catching a class of bug that
otherwise produces a publishable-looking figure.

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

If that moves, your indexing, positions, or donor source is wrong — stop and fix
it before running anything else.

**An extreme intervention must do something.** If zeroing a whole layer leaves the
metric unchanged, the intervention is not landing:

```python
with model.trace(clean):
    model.transformer.h[6].output[:] = 0
    logits = model.output.logits[0, -1]
    destroyed = (logits[paris] - logits[rome]).detach().save()

print(f"layer zeroed: {float(destroyed):+.3f}  (should be far from {float(clean_metric):+.3f})")
```

**The unmodified baseline must reproduce the known behavior.** Run it in the same
trace as the intervention, not from memory of a previous session.

## 4. Controls

An effect is only evidence if the null does worse. Pick the control that matches
the claim:

| Claim | Control |
|---|---|
| this component matters | the same intervention on a random component |
| this direction encodes X | a random direction of the same norm |
| this circuit is the mechanism | a random subgraph of the same size |
| this probe found a representation | shuffled labels; a probe at layer 0 |
| this steering vector works | the negated vector; a random vector |

```python
generator = torch.Generator(device=model.device).manual_seed(0)
resid_norm = None

with model.trace(clean):
    resid = model.transformer.h[6].output
    resid_norm = resid[0, -1].norm().detach().save()

real = torch.randn(768, generator=generator, device=model.device)
real = real / real.norm()

with model.trace() as tracer:
    with tracer.invoke(clean):
        model.transformer.h[6].output[:, -1, :] += 0.5 * float(resid_norm) * real
        logits = model.output.logits[0, -1]
        random_effect = (logits[paris] - logits[rome]).detach().save()

print(f"random direction moved the metric by "
      f"{float(random_effect) - float(clean_metric):+.3f}")
```

Whatever a random direction achieves is the floor your real direction must clear.

## 5. Prompts: one is an anecdote

A single prompt gives one sample of a noisy process. Layer indices, head rankings,
and steering coefficients all shift across paraphrases.

- Use a **set** of prompts sharing the structure under test (8–100 for a
  qualitative result; more for a quantitative one).
- Use **minimal pairs**: identical token count, identical structure, differing only
  in the variable. Assert the lengths match — mismatched pairs silently make every
  position index meaningless.
- Hold out prompts the analysis never saw and report the effect there.
- Vary surface form deliberately (paraphrase, reorder, change names) to check the
  finding is about the task rather than the string.

The alignment assert is not decoration. Here it is catching a pair that looks fine
and is not:

<!-- test: expect-error AssertionError -->
```python
pairs = [
    ("The Eiffel Tower is in the city of", "The Colosseum is in the city of"),
    ("The Statue of Liberty is in the city of", "Big Ben is in the city of"),
]
for a, b in pairs:
    assert len(model.tokenizer(a).input_ids) == len(model.tokenizer(b).input_ids), (a, b)
```

"The Statue of Liberty…" and "Big Ben…" tokenize to different lengths, so position
`i` is a different thing in each. Without the assert, every position-indexed result
from that pair would be meaningless — and nothing would have raised.

Filter the set instead of trusting it:

```python
candidates = [
    ("The Eiffel Tower is in the city of", "The Colosseum is in the city of"),
    ("The Statue of Liberty is in the city of", "Big Ben is in the city of"),
    ("The Brandenburg Gate is in the city of", "The Sydney Opera is in the city of"),
]
pairs = [(a, b) for a, b in candidates
         if len(model.tokenizer(a).input_ids) == len(model.tokenizer(b).input_ids)]

print(f"{len(pairs)} of {len(candidates)} candidate pairs are aligned")
```

## 6. Statistical hygiene

**Multiple comparisons are everywhere.** Scoring 144 heads and reporting the top
one means the top one is partly noise. Report the whole distribution, keep a
held-out set, or correct.

**Seeds matter** wherever noise is injected (causal tracing) or parameters are
initialized (probes, DAS, SAEs). Average over several; if the answer changes with
the seed, it is not an answer.

**Effect sizes over binaries.** "Recovers 60% of the gap" is information; "patching
worked" is not.

**Report what you searched.** A top-5 list from a search over 12 layers means
something different from the same list out of 144 heads × 10 positions. State the
search space and any truncation.

## 7. Compute budget

Design the experiment to fit, using the `nnsight` skill's primitives:

| Situation | Move |
|---|---|
| N variants of one intervention | one `tracer.invoke` each → one forward pass |
| activations from many modules | one trace, loop inside, or `tracer.cache()` |
| only layers 0..L matter | `tracer.stop()` after L |
| the donor run is reused | cache it once, outside the sweep |
| every component, large model | attribution first, verify the top-K |
| model too large to host | NDIF — see `nnsight-remote` |

The pattern that changes an experiment's feasibility most is the sweep-in-invokes
one: a 144-condition head scan is 12 batched traces, not 144.

## 8. Before you claim it

- [ ] The metric moves for the reason you think (no-op and extreme checks pass)
- [ ] A baseline ran in the same trace as the intervention
- [ ] A control (random component / direction / label) did worse
- [ ] More than one prompt, and a held-out set
- [ ] Effect reported as a fraction of a known span, not a raw number
- [ ] The search space and any pruning are stated
- [ ] Ablation type / patch direction / corrupt distribution are stated — results
      are not comparable across those choices
- [ ] The claim matches the rung: readable ≠ used; sufficient ≠ necessary;
      correlated ≠ causal
- [ ] The negative results are reported too — a component that did *not* matter is
      as informative as one that did

## Common ways experiments go wrong

**Measuring the intervention instead of the model.** Large interventions push the
network off-distribution; you learn about the perturbation, not the mechanism.
Keep interventions in-distribution (mean/resample rather than zero) where you can.

**Confirming the hypothesis you started with.** Decide in advance what result
would falsify the claim, and check for it.

**Reading a heatmap as a mechanism.** A bright cell says an intervention there
changed the output. It does not say what the component computes.

**Trusting an approximation you never validated.** Attribution patching, probes,
and SAEs are all models of the model. Validate against ground truth on a subset.

**Generalizing across scale.** GPT-2 small is not a small Llama. Findings about
where something lives rarely transfer across model families or sizes without
re-running.

## Related skills

- `nnsight` — the primitives every experiment is built from
- `activation-patching`, `ablation`, `probing`, `circuit-discovery` — techniques,
  each with its own controls section
- `nnsight-debugging` — when the code, rather than the design, is the problem
