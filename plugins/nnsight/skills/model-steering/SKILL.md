---
name: model-steering
description: "Change a model's behavior at inference time by adding a direction to its residual stream — contrastive activation addition, steering vectors, always-on steering via model.edit(), and function vectors extracted from in-context-learning prompts. Use to induce or suppress a behavior (sentiment, refusal, style, a task), to test whether a direction found by a probe is causal, or to keep a steering vector applied on every run (weight-level changes belong to model-editing-and-lora). Covers the two things that decide whether steering works: scaling the vector relative to the activation norm, and sweeping the coefficient to find the band where behavior changes before fluency collapses."
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
        # detach: a saved activation's live grad_fn keeps the whole forward graph alive
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

Three regimes: at 0.25 the output changes without carrying the concept (Japan
becomes the UK), the sentiment arrives at 0.5, and by 2.0 the model loops. That
sub-threshold row is why you sweep: **an output that moved is not an output that
moved for your reason.** Report the whole sweep.

**`min_new_tokens=12` is load-bearing.** Steering changes when the model emits
EOS: at α = 0.25 this one stops after 11 passes. A `tracer.iter[...]` loop the run
does not reach the end of is cut short with only a `UserWarning: … was never
reached`, and every statement after it is dropped — `ids = tracer.result.save()`
included, so that row comes back unbound. The same happens to an open
`tracer.iter[1:]`, which always runs to the end of the generation. Pin the step
count with `min_new_tokens=N`; it also makes the rows the same length and
comparable.

## Where to inject

| Choice | Effect |
|---|---|
| **Layer** | Middle layers (⅓–⅔ depth) usually work best. Early layers get overwritten; late layers only nudge the surface form. Sweep it like α. |
| **Positions** | `[:, -1, :]` steers only the token being predicted; `[:, :, :]` steers every position and is stronger but blunter. |
| **Every step vs first step** | Inside a `tracer.iter[...]` loop the write repeats on every forward **pass**. Pass 0 is the *prefill*, whose hidden state is `[B, prompt_len, H]`: a `[:, -1, :]` write there touches only the last prompt position, and `[:, :, :]` touches all of them — the slice decides what gets steered, not the step. Passes 1+ carry one generated token each, so `tracer.iter[1:12]` steers only generated tokens. Applied once, the effect decays. |
| **Timing** | A layer-L write on step k reaches the layers *above* L on that same pass, and the KV cache carries it to layers above L on every later step. Layers ≤ L stay bit-identical to the unedited run until a generated token actually changes: a 0.3× edit that does not flip the argmax leaves layers 3 and 6 (pre-edit) identical on later steps while layer 9 differs. So an edit on `tracer.iter[a:b]` can first change generated token a (absolute position prompt_len + a — the slice counts forward passes, step 0 is the prefill), and a later one only if it was too weak to flip that argmax. |

## Persistent steering

`model.edit(inplace=True)` replays the intervention on every subsequent run:

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

The open `tracer.iter[:]` is right inside an edit, where nothing follows the loop;
its "never reached" warning on each `generate` is expected.

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

capitals = {"Portugal": " Lisbon", "Brazil": " Brasília", "Poland": " Warsaw",
            "Greece": " Athens", "Sweden": " Stockholm", "Austria": " Vienna",
            "Thailand": " Bangkok"}

plain_out, fv_out, random_out = {}, {}, {}
for country in capitals:
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
    plain_out[country] = model.tokenizer.decode(plain)
    fv_out[country] = model.tokenizer.decode(steered)
    random_out[country] = model.tokenizer.decode(random)
    print(f"{zero_shot:<14} plain {plain_out[country]!r:>12}"
          f"   +fv {fv_out[country]!r:>12}"
          f"   +random {random_out[country]!r:>12}")

hits = sum(fv_out[c] == capitals[c] for c in capitals)
assert hits > sum(random_out[c] == capitals[c] for c in capitals)   # beats the control
assert any(random_out[c] != plain_out[c] for c in capitals)         # control is not inert
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

The vector produces the capital on three of seven; the rest echo the country or
continue wrongly. The random control is not inert — it changes three outputs — so
on any single prompt "the output moved" is worth nothing without the control
column beside it.

Three of seven is what a whole-residual difference buys on GPT-2 small; it carries
format as well as task. The published version averages the outputs of a few
**attention heads** selected by causal attribution (slice per-head activations
from `attn.c_proj.input` — the `activation-patching` skill has the head-slicing
convention), and transfers across prompt formats far better.

Run the few-shot control before believing any hit: put demonstrations in the
prompt and check the model can do the task for that input at all. A function
vector should not install a capability the model lacks, so a hit on a prompt the
model fails few-shot is a result to explain before publishing it.

## Evaluating a steering vector

**Dose-response.** Behavior should increase smoothly with α before collapsing; a
step from nothing to gibberish is disruption, not steering. The probability still
assigned to the *unsteered* top token is a cheap continuous version of that curve
and should fall monotonically:

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
entropy: a single next-token distribution cannot see a loop that only exists
across steps (here it is non-monotonic in α and flat where the output loops).

**Held-out prompts.** Test on topics and structures unlike the contrast pairs.

**A matched-norm random control**, as in the function-vector table: a comparable
change from it means your vector is not carrying the concept.

**The opposite direction.** Negating `v` should suppress the behavior. The
sentiment vector above fails: α = −0.5 and −1.0 give neutral continuations, and
−2.0 breaks the text. Difference-of-means directions are often asymmetric; check
before calling one bidirectional.

## Common failures

| Symptom | Cause |
|---|---|
| Output is word salad at every α > 0 | vector not scaled to the activation norm |
| Nothing changes at any α | wrong layer, or steering only the last position when the effect needs earlier tokens |
| Effect vanishes after the first generated token | intervention not inside a `tracer.iter[...]` loop |
| Works on the derivation prompts only | too few contrast pairs; the vector encodes the prompts |
| Two rows of a sweep are identical | check they are not the same generation printed twice |

## Related skills

- `nnsight` — invokes, generation, edits
- `probing` — finding directions by training a classifier rather than differencing
- `sae-and-dictionary-learning` — steering with interpretable features instead of raw directions
- `activation-patching` — verifying that a direction is causal, not correlational
