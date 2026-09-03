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
meaningful magnitude: here its norm is 27 while the residual it is added to has
norm 93. Multiplying the raw vector by 5 or 10, the coefficients that appear in
many tutorials, is α = 1.5 and α = 2.9 on the scale below: both past the point
where this model starts repeating itself.

Scale the unit direction by the *measured* activation norm, then sweep. Track a
cheap collapse signal alongside the text. The fraction of generated bigrams that
repeat is zero for fluent output and jumps when the model starts looping:

```python
with model.trace(prompt):
    scale = model.transformer.h[LAYER].output[0, -1].norm().detach().save()

print(f"residual norm at layer {LAYER}: {float(scale):.1f}")

alphas = [0.0, 0.25, 0.5, 1.0, 2.0]
generations = {}
for alpha in alphas:
    with model.generate(prompt, max_new_tokens=12, min_new_tokens=12) as tracer:
        for step in tracer.iter[:12]:
            model.transformer.h[LAYER].output[:, -1, :] += alpha * float(scale) * direction
        ids = tracer.result.save()
    sequence = ids[0].tolist()
    bigrams = list(zip(sequence, sequence[1:]))
    repeat = 1 - len(set(bigrams)) / len(bigrams)
    generations[alpha] = model.tokenizer.decode(sequence)
    print(f"alpha {alpha:>4}  repeat {repeat:.2f}  {generations[alpha]!r}")

# Two rows that come out identical almost always mean one of them never ran.
assert len(set(generations.values())) == len(alphas)
```

```
residual norm at layer 6: 93.2
alpha  0.0  repeat 0.00  'The movie was released in Japan on May 7, 2016.\n\nThe'
alpha 0.25  repeat 0.00  'The movie was released in the UK on May 7, 2016.\n\n'
alpha  0.5  repeat 0.00  'The movie was a great way to get your hands on the first time you'
alpha  1.0  repeat 0.00  'The movie was a "tremie and the first step to my new'
alpha  2.0  repeat 0.43  'The movie was the first and the first and the first and the I and'
```

Read the sweep as three regimes. At 0.25 the output changes without carrying the
concept: the sentence keeps its shape and Japan becomes the UK. The sentiment the
vector encodes arrives at 0.5. By 2.0 the model is looping.

That sub-threshold row is why you sweep instead of testing one coefficient.
**An output that moved is not an output that moved for your reason.** Report the
whole sweep.

**`min_new_tokens=12` is load-bearing here.** Steering changes when the model
emits EOS: at α = 0.25 this one stops after 11 forward passes. A bounded
`tracer.iter[:12]` that the run does not reach the end of cuts the loop short and
drops every statement after it, `ids = tracer.result.save()` included, so nnsight
raises `OutOfOrderError` naming the iteration asked for and the count reached.
Pinning the step count also makes the rows the same length and therefore
comparable. See the `nnsight` skill → generation.

## Where to inject

| Choice | Effect |
|---|---|
| **Layer** | Middle layers (⅓–⅔ depth) usually work best. Early layers get overwritten; late layers only nudge the surface form. Sweep it like α. |
| **Positions** | `[:, -1, :]` steers only the token being predicted; `[:, :, :]` steers every position and is stronger but blunter. |
| **Every step vs first step** | Inside a `tracer.iter[...]` loop the vector is re-added on every forward **pass**. Pass 0 is the *prefill*, so the vector lands on every prompt position at once; passes 1+ are one generated token each. `tracer.iter[1:12]` steers only the generated tokens. Applied once, the effect decays. |

Give that loop both ends. An open `tracer.iter[1:]` runs to the end of the
generation and then unwinds everything after it, so a `tracer.result.save()`
below the loop never executes and the name it assigns stays unbound.

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
every later experiment; clear it in a `finally`, or use the copy form.

The open `tracer.iter[:]` is right inside an edit, where nothing follows the loop,
and every `generate` on the edited model warns that the step after the last one
was never reached. The edit still applied on every step that ran.

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

# A matched-norm random direction: the control that says whether the vector's
# content matters, or only its magnitude.
generator = torch.Generator().manual_seed(0)
control = torch.randn(function_vector.shape, generator=generator).to(function_vector)
control = control / control.norm() * function_vector.norm()

for country in ["Portugal", "Brazil", "Poland", "Greece", "Sweden", "Austria", "Thailand"]:
    zero_shot = f"{country} ->"
    with model.trace() as tracer:
        with tracer.invoke(zero_shot):
            plain = model.output.logits[0, -1].argmax().save()
        with tracer.invoke(zero_shot):
            model.transformer.h[LAYER].output[:, -1, :] += function_vector
            steered = model.output.logits[0, -1].argmax().save()
        with tracer.invoke(zero_shot):
            model.transformer.h[LAYER].output[:, -1, :] += control
            random = model.output.logits[0, -1].argmax().save()
    print(f"{zero_shot:<14} plain {model.tokenizer.decode(plain)!r:>12}"
          f"   +fv {model.tokenizer.decode(steered)!r:>12}"
          f"   +random {model.tokenizer.decode(random)!r:>12}")
```

```
Portugal ->    plain         '\n'   +fv    ' Lisbon'   +random  ' Portugal'
Brazil ->      plain         '\n'   +fv       ' New'   +random        ' US'
Poland ->      plain         '\n'   +fv    ' Poland'   +random         '\n'
Greece ->      plain         '\n'   +fv    ' Greece'   +random         '\n'
Sweden ->      plain    ' Sweden'   +fv    ' Sweden'   +random    ' Sweden'
Austria ->     plain   ' Austria'   +fv    ' Vienna'   +random   ' Germany'
Thailand ->    plain  ' Thailand'   +fv   ' Bangkok'   +random  ' Thailand'
```

The vector produces the capital on three of the seven countries. Three of the
remaining four come back as the country's own name and Brazil gets a wrong
continuation. The random control is not inert either: it changes three of the
seven outputs, so on any single prompt "the output moved" is worth nothing
without the control column beside it. The `Portugal` row on its own would have
shown a method that works.

Three of seven on GPT-2 small is what a whole-residual difference buys. The
published version is more selective: it averages the outputs of a small set of
**attention heads** identified by causal attribution as task-carrying, rather than
taking the whole residual difference. Extract per-head activations with
`model.transformer.h[L].attn.c_proj.input[:, :, head*d:(head+1)*d]` (see the
`activation-patching` skill for the head-slicing convention) and select heads by
attribution score. The selective version transfers across prompt formats far
better; the crude difference above carries format as well as task.

Run the few-shot control before you believe any of it. Put the demonstrations in
the prompt and see whether the model does the task at all:

```python
for country in ["Portugal", "Brazil", "Poland", "Greece", "Sweden", "Austria", "Thailand"]:
    with model.trace(f"France -> Paris, Japan -> Tokyo, Italy -> Rome, {country} ->"):
        answer = model.output.logits[0, -1].argmax().save()
    print(f"  {country:<10} -> {model.tokenizer.decode(answer)!r}")
```

```
  Portugal   -> ' Rome'
  Brazil     -> ' Buenos'
  Poland     -> ' Warsaw'
  Greece     -> ' Athens'
  Sweden     -> ' Stockholm'
  Austria    -> ' Vienna'
  Thailand   -> ' Bangkok'
```

Five of seven with demonstrations. `Brazil` answers with Argentina's capital,
and `Portugal` copies the last demonstration instead of answering. `Portugal` is
also the one prompt in the table above where the vector produces the right
capital. A function vector is not supposed to install a capability the model
lacks, so a hit on a prompt the model fails few-shot is a result to explain
before publishing it.

## Evaluating a steering vector

A vector that changes the output is not automatically a vector that encodes the
concept. Check:

**Dose-response.** Behavior should increase smoothly with α before collapsing. A
step change from nothing to gibberish means you are disrupting, not steering. The
probability the steered model still assigns to the *unsteered* top token is a
cheap continuous version of that curve, and it should fall monotonically:

```python
with model.trace(prompt):
    unsteered = model.output.logits[0, -1].argmax().save()
unsteered_top = int(unsteered)

with model.trace() as tracer:
    kept = nnsight.save([])
    for alpha in alphas:
        with tracer.invoke(prompt):
            model.transformer.h[LAYER].output[:, -1, :] += alpha * float(scale) * direction
            kept.append(model.output.logits[0, -1].softmax(-1)[unsteered_top].detach())

probabilities = [float(p) for p in kept]
for alpha, p in zip(alphas, probabilities):
    print(f"alpha {alpha:>5}  p(unsteered top token) {p:.4f}")

assert probabilities == sorted(probabilities, reverse=True)
```

```
alpha   0.0  p(unsteered top token) 0.0880
alpha  0.25  p(unsteered top token) 0.0703
alpha   0.5  p(unsteered top token) 0.0281
alpha   1.0  p(unsteered top token) 0.0045
alpha   2.0  p(unsteered top token) 0.0004
```

**Fluency.** Use the repeated-bigram fraction from the sweep, not next-token
entropy. Entropy on this vector reads 5.43 at α = 0, 6.02 at 0.5 and 5.95 at 2.0:
non-monotonic, and barely distinguishable at the coefficient where the output is
`'the first and the first and the first'`. A single next-token distribution cannot
see a loop that only exists across steps.

**Held-out prompts.** The vector was derived from specific contrast pairs. Test it
on prompts with different topics and structure.

**A control direction.** Steer with a random vector of the same norm. If it
produces a comparable behavior change, your vector is not carrying the concept.

**The opposite direction.** Negating `v` should suppress the behavior. The
sentiment vector above does not pass this one: α = −0.5 and −1.0 give neutral
continuations rather than negative ones, and −2.0 breaks the text down. A
difference-of-means direction is often asymmetric like that, so check before
describing one as bidirectional.

## Common failures

| Symptom | Cause |
|---|---|
| Output is word salad at every α > 0 | vector not scaled to the activation norm |
| Nothing changes at any α | wrong layer, or steering only the last position when the effect needs earlier tokens |
| Effect vanishes after the first generated token | intervention not inside a `tracer.iter[...]` loop |
| Works on the derivation prompts only | too few contrast pairs; the vector encodes the prompts |
| Later experiments behave strangely | an `edit(inplace=True)` was never cleared |
| `OutOfOrderError` naming an iteration of a bounded `tracer.iter[:N]` | steering changed when the model emits EOS, so the run made fewer passes than the loop asked for; add `min_new_tokens=N` |
| Two rows of a sweep are identical | check they are not the same generation printed twice |

## Related skills

- `nnsight` — invokes, generation, edits
- `probing` — finding directions by training a classifier rather than differencing
- `sae-and-dictionary-learning` — steering with interpretable features instead of raw directions
- `activation-patching` — verifying that a direction is causal, not correlational
