---
name: model-steering
description: Change a model's behavior at inference time by adding a direction to its residual stream — contrastive activation addition, steering vectors, persistent edits, and function vectors extracted from in-context-learning prompts. Use to induce or suppress a behavior (sentiment, refusal, style, a task), to test whether a direction found by a probe is causal, or to build always-on model modifications. Covers the two things that decide whether steering works: scaling the vector relative to the activation norm, and sweeping the coefficient to find the band where behavior changes before fluency collapses.
---

# Model Steering

Steering adds a direction to the residual stream during the forward pass:

```
h_L  ←  h_L + α · v
```

Get `v` from the difference between activations on contrasting prompts, and `α`
from a sweep. Everything else is detail.

<!-- test: setup -->
```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)
LAYER = 6
prompt = "The movie was"
```

## Deriving a steering vector

Contrastive activation addition: run positive and negative examples, take the
difference of their mean activations. Both sets go in one forward pass.

```python
positive = ["I love this", "This is wonderful", "What a great day", "I am so happy"]
negative = ["I hate this", "This is terrible", "What an awful day", "I am so sad"]

with model.trace() as tracer:
    with tracer.invoke(positive):
        pos_mean = model.transformer.h[LAYER].output[:, -1, :].mean(0).detach().save()
    with tracer.invoke(negative):
        neg_mean = model.transformer.h[LAYER].output[:, -1, :].mean(0).detach().save()

raw = pos_mean - neg_mean
direction = raw / raw.norm()
print(f"raw norm {raw.norm().item():.1f}, unit direction {tuple(direction.shape)}")
```

More pairs is better — a handful of examples encodes prompt idiosyncrasies as much
as the concept. Use contrast pairs that differ *only* in the attribute you want.

## Scale relative to the activation, then sweep

**This is where most steering attempts fail.** The raw difference vector has no
meaningful magnitude: here its norm is 27 while the residual stream it is added to
has norm ~93. Adding it with a coefficient of 5 or 10 — numbers that appear in
many tutorials — swamps the representation and produces word salad.

Scale the unit direction by the *measured* activation norm, then sweep:

```python
with model.trace(prompt):
    scale = model.transformer.h[LAYER].output[0, -1].norm().detach().save()

print(f"residual norm at layer {LAYER}: {float(scale):.1f}")

for alpha in [0.0, 0.25, 0.5, 1.0, 2.0]:
    with model.generate(prompt, max_new_tokens=12) as tracer:
        for step in tracer.iter[:12]:
            model.transformer.h[LAYER].output[:, -1, :] += alpha * float(scale) * direction
        ids = tracer.result.save()
    print(f"alpha {alpha:>4}: {model.tokenizer.decode(ids[0])!r}")
```

```
alpha  0.0: 'The movie was released in Japan on May 7, 2016.'
alpha 0.25: 'The movie was released in Japan on May 7, 2016.'
alpha  0.5: 'The movie was a great way to get your hands on the first time you'
alpha  1.0: 'The movie was a "tremie and the first step to my new'
alpha  2.0: 'The movie was the first and the first and the first and the I and'
```

The working band is narrow: nothing happens at 0.25, the intended effect appears
at 0.5, and by 1.0 fluency is already degrading into repetition. **Always report
the sweep, not a single coefficient** — "steering worked" at one α with no
neighbours shown is not evidence.

(The `tracer.iter[:12]` loop may warn that the last iteration was never reached
when the model emits fewer forward passes than requested; values from the steps
that ran are kept. See the `nnsight` skill → generation.)

## Where to inject

| Choice | Effect |
|---|---|
| **Layer** | Middle layers (⅓–⅔ depth) usually work best. Early layers get overwritten; late layers only nudge the surface form. Sweep it like α. |
| **Positions** | `[:, -1, :]` steers only the token being predicted; `[:, :, :]` steers every position and is stronger but blunter. |
| **Every step vs first step** | Inside a `tracer.iter[...]` loop the vector is re-added on every forward **pass**. Pass 0 is the *prefill*, so the vector lands on every prompt position at once; passes 1+ are one generated token each. Use `tracer.iter[1:]` to steer only the generated tokens. Applied once, the effect decays. |

## Persistent steering

`model.edit(inplace=True)` bakes the intervention into every subsequent run — no
`with` block needed at the call site:

```python
with model.edit(inplace=True) as tracer:
    for _ in tracer.iter[:]:
        model.transformer.h[LAYER].output[:, -1, :] += 0.5 * float(scale) * direction

with model.generate(prompt, max_new_tokens=10) as tracer:
    ids = tracer.result.save()
print("steered:", model.tokenizer.decode(ids[0]))

model.clear_edits()

with model.generate(prompt, max_new_tokens=10) as tracer:
    ids = tracer.result.save()
print("cleared:", model.tokenizer.decode(ids[0]))
```

Use `model.edit()` without `inplace=True` to get a steered *copy* and keep the
original clean for comparison. An edit you forget to clear silently contaminates
every later experiment — clear it in a `finally`, or use the copy form.

## Function vectors

A function vector (Todd et al.) encodes a *task* rather than an attribute. Extract
the activation that in-context examples create, then inject it into a zero-shot
prompt so the model performs the task without demonstrations.

```python
icl = [
    "France -> Paris, Japan -> Tokyo, Italy -> Rome, Spain ->",
    "Germany -> Berlin, Egypt -> Cairo, Peru -> Lima, Chile ->",
    "Canada -> Ottawa, Kenya -> Nairobi, Cuba -> Havana, Norway ->",
]
neutral = [
    "Spain is a country and Spain ->",
    "Chile is a country and Chile ->",
    "Norway is a country and Norway ->",
]

with model.trace() as tracer:
    with tracer.invoke(icl):
        icl_state = model.transformer.h[LAYER].output[:, -1, :].mean(0).detach().save()
    with tracer.invoke(neutral):
        neutral_state = model.transformer.h[LAYER].output[:, -1, :].mean(0).detach().save()

function_vector = icl_state - neutral_state

zero_shot = "Portugal ->"
with model.trace() as tracer:
    with tracer.invoke(zero_shot):
        plain = model.output.logits[0, -1].argmax().save()
    with tracer.invoke(zero_shot):
        model.transformer.h[LAYER].output[:, -1, :] += function_vector
        steered = model.output.logits[0, -1].argmax().save()

print(f"plain   {model.tokenizer.decode(plain)!r}")
print(f"steered {model.tokenizer.decode(steered)!r}")
```

```
plain   '\n'
steered ' Lisbon'
```

The zero-shot prompt `"Portugal ->"` gives a newline on its own; with the function
vector added at layer 6 the model answers the country→capital task it was never
shown. That is the whole claim of the method, reproduced on GPT-2 small.

The published version is more selective: it averages the outputs of a small set of
**attention heads** identified by causal attribution as task-carrying, rather than
taking the whole residual difference. Extract per-head activations with
`model.transformer.h[L].attn.c_proj.input[:, :, head*d:(head+1)*d]` (see the
`activation-patching` skill for the head-slicing convention) and select heads by
attribution score. The selective version transfers across prompt formats far
better; the crude difference above tends to carry format as well as task.

A function vector cannot install a capability the model lacks — if the model fails
the task *with* demonstrations, no vector extracted from them will fix that.

## Evaluating a steering vector

A vector that changes the output is not automatically a vector that encodes the
concept. Check:

**Dose-response.** Behavior should increase smoothly with α before collapsing. A
step change from nothing to gibberish means you are disrupting, not steering.

**Fluency.** Track a proxy — next-token entropy, or the probability of the
unsteered top token — alongside the effect, so you can see collapse coming:

```python
with model.trace() as tracer:
    entropies = nnsight.save([])
    for alpha in [0.0, 0.5, 1.0, 2.0]:
        with tracer.invoke(prompt):
            model.transformer.h[LAYER].output[:, -1, :] += alpha * float(scale) * direction
            probs = model.output.logits[0, -1].softmax(-1)
            entropies.append(-(probs * probs.log()).sum().detach())

for alpha, h in zip([0.0, 0.5, 1.0, 2.0], entropies):
    print(f"alpha {alpha:>4}  next-token entropy {float(h):.3f}")
```

**Held-out prompts.** The vector was derived from specific contrast pairs. Test it
on prompts with different topics and structure.

**A control direction.** Steer with a random vector of the same norm. If it
produces a comparable behavior change, your vector is not carrying the concept.

**The opposite direction.** Negating `v` should suppress the behavior. A vector
that only works in one direction is suspicious.

## Common failures

| Symptom | Cause |
|---|---|
| Output is word salad at every α > 0 | vector not scaled to the activation norm |
| Nothing changes at any α | wrong layer, or steering only the last position when the effect needs earlier tokens |
| Effect vanishes after the first generated token | intervention not inside a `tracer.iter[...]` loop |
| Works on the derivation prompts only | too few contrast pairs; the vector encodes the prompts |
| Later experiments behave strangely | an `edit(inplace=True)` was never cleared |

## Related skills

- `nnsight` — invokes, generation, edits
- `probing` — finding directions by training a classifier rather than differencing
- `sae-and-dictionary-learning` — steering with interpretable features instead of raw directions
- `activation-patching` — verifying that a direction is causal, not correlational
