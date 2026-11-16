# Model steering

Steering adds a direction to the residual stream at one layer on every forward
pass: `h ← h + α · s · v`. The unit direction `v` comes from contrasting
activations, `s` is the measured norm of the residual at that layer, the
coefficient `α` comes from a sweep, and a random direction of the same norm is
the control that says a change is due to `v`.

```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True)
LAYER = 6
prompt = "The movie was"
```

## A contrastive vector in one trace

The direction is the difference between the mean last-token activations of two
prompt sets. Both sets, and the norm of the residual the vector is added to, come
from one forward pass:

```python
positive = ["I love this", "This is wonderful", "What a great day", "I am so happy"]
negative = ["I hate this", "This is terrible", "What an awful day", "I am so sad"]

with model.trace() as tracer:
    with tracer.invoke(positive):
        pos_mean = model.transformer.h[LAYER].output[:, -1, :].mean(0).detach().save()
    with tracer.invoke(negative):
        neg_mean = model.transformer.h[LAYER].output[:, -1, :].mean(0).detach().save()
    with tracer.invoke(prompt):
        scale = model.transformer.h[LAYER].output[0, -1].norm().detach().save()

raw = pos_mean - neg_mean
direction = raw / raw.norm()
scale = float(scale)
print(f"raw norm {raw.norm().item():.1f}, residual norm {scale:.1f}")
```

```
raw norm 27.3, residual norm 93.2
```

Two disjoint sets of four pairs give directions with cosine 0.61 here, so four
pairs carry their own wording as well as the concept. Use more, make each pair
differ only in the attribute, and evaluate on prompts unlike them. A direction
from a [probe](probing.md) or an
[SAE feature](sae-and-dictionary-learning.md) is scaled, swept and evaluated the
same way.

## Scale by the residual norm, then sweep

The raw difference has no meaningful magnitude: its norm is 27 where the residual
has norm 93. Multiply the unit direction by the measured residual norm, so that α
is a fraction of the activation, and sweep α. Print a collapse signal beside the
text: the fraction of generated bigrams that repeat is 0 for fluent output and
jumps when the model loops.

```python
alphas = [0.0, 0.25, 0.5, 1.0, 2.0]
generations = {}
for alpha in alphas:
    with model.generate(prompt, max_new_tokens=12, min_new_tokens=12, do_sample=False) as tracer:
        for step in tracer.iter[:12]:
            model.transformer.h[LAYER].output[:, -1, :] += alpha * scale * direction
        ids = tracer.result.save()
    sequence = ids[0].tolist()
    bigrams = list(zip(sequence, sequence[1:]))
    repeat = 1 - len(set(bigrams)) / len(bigrams)
    generations[alpha] = model.tokenizer.decode(sequence)
    print(f"alpha {alpha:>4}  repeat {repeat:.2f}  {generations[alpha]!r}")

assert len(set(generations.values())) == len(alphas)   # two equal rows: one of them did not run
```

```
alpha  0.0  repeat 0.00  'The movie was released in Japan on May 7, 2016.\n\nThe'
alpha 0.25  repeat 0.00  'The movie was released in the UK on May 7, 2016.\n\n'
alpha  0.5  repeat 0.00  'The movie was a great way to get your hands on the first time you'
alpha  1.0  repeat 0.00  'The movie was a "tremie and the first step to my new'
alpha  2.0  repeat 0.43  'The movie was the first and the first and the first and the I and'
```

At 0.25 the output changes without carrying the concept, the sentiment arrives at
0.5, and at 2.0 the model loops. An output that moved has not moved for your
reason: report the whole sweep, with the controls under
[Evaluating a steering vector](#evaluating-a-steering-vector).

Steering changes when the model emits EOS. Without `min_new_tokens=12` the
α = 0.25 run ends after 11 steps, the loop is cut short and that row's `ids` is
never bound.

## Where to inject

- **Layer.** Sweep it like α, deriving `direction` and `scale` at each layer: the
  residual norm on this prompt runs from 57 at layer 0 to 487 at layer 11. At
  α = 0.5 layer 6 gives the fluent row above, layers 1 and 3 already repeat
  (layer 3: `the story of the story of the story`), and layer 11 loops on `with
  a Canon EF` with no sentiment in the text.
- **Positions.** `[:, -1, :]` writes the position being predicted. `[:, :, :]`
  writes every position the step holds and is stronger: here the sentiment
  arrives at α = 0.25 (`a success, and the cast and crew are all very excited`)
  where the last-position write needs 0.5.
- **Steps.** The two slices differ only on step 0, the prefill, where
  `[:, :, :]` covers the whole prompt. Every later step holds one token.
  `tracer.iter[1:12]` steers generated positions only, and the first generated
  token is then the unsteered model's.

A write at layer L changes the layers at or below L on later steps only through
the token it changes ([generation](generation.md)).

## Persistent steering

`model.edit(inplace=True)` stores the write and replays it on every later run,
until `model.clear_edits()`:

```python
with model.edit(inplace=True) as tracer:
    for _ in tracer.iter[:]:
        model.transformer.h[LAYER].output[:, -1, :] += 0.5 * scale * direction

steered = model.generate(prompt, max_new_tokens=10, min_new_tokens=10, do_sample=False)
model.clear_edits()
cleared = model.generate(prompt, max_new_tokens=10, min_new_tokens=10, do_sample=False)

print("steered:", repr(model.tokenizer.decode(steered[0])))
print("cleared:", repr(model.tokenizer.decode(cleared[0])))
assert not torch.equal(steered, cleared)
```

```
steered: 'The movie was a great way to get your hands on the first'
cleared: 'The movie was released in Japan on May 7, 2016.\n'
```

The loop over `tracer.iter[:]` applies the edit on every generation step; without
it the edit applies on the first pass only. Each run then ends with a
`UserWarning: ... was never reached` from the open loop, which is expected here.
An edit left in place steers every later experiment. Clearing it, steering a copy
with `model.edit()` and the other rules of edits are in
[control-flow](control-flow.md).

## Function vectors

A function vector carries a task. Take the last-token state that in-context
demonstrations produce, subtract the state of a prompt with no demonstrations,
and add the difference to a zero-shot prompt. Two controls go in the same trace:
a random vector of the same norm, and the few-shot prompt itself, which shows
whether the model can do the task for that input at all.

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
few_shot = "France -> Paris, Japan -> Tokyo, Italy -> Rome, "
capitals = {"Portugal": " Lisbon", "Russia": " Moscow", "Poland": " Warsaw",
            "Greece": " Athens", "Sweden": " Stockholm", "Austria": " Vienna",
            "Thailand": " Bangkok"}
answer = {country: model.tokenizer.encode(capital, add_special_tokens=False)[0]
          for country, capital in capitals.items()}

with model.trace() as tracer:
    with tracer.invoke(icl):
        icl_state = model.transformer.h[LAYER].output[:, -1, :].mean(0).detach().save()
    with tracer.invoke(neutral):
        neutral_state = model.transformer.h[LAYER].output[:, -1, :].mean(0).detach().save()

function_vector = icl_state - neutral_state
generator = torch.Generator().manual_seed(0)
matched = torch.randn(function_vector.shape, generator=generator).to(function_vector)
matched = matched / matched.norm() * function_vector.norm()

columns = {"plain": None, "+fv": function_vector, "+random": matched, "few-shot": None}
with model.trace() as tracer:
    top = nnsight.save({})
    for country in capitals:
        for column, vector in columns.items():
            demos = few_shot if column == "few-shot" else ""
            with tracer.invoke(f"{demos}{country} ->"):
                if vector is not None:
                    model.transformer.h[LAYER].output[:, -1, :] += vector
                top[country, column] = model.output.logits[0, -1].argmax()

hits = {column: sum(int(top[c, column]) == answer[c] for c in capitals) for column in columns}
for country in capitals:
    print(f"{country:<9}" + "".join(
        f"{column} {model.tokenizer.decode(top[country, column])!r:<12}" for column in columns))
print(hits)

assert hits["+fv"] > hits["+random"]                                   # beats the control
assert any(top[c, "+random"] != top[c, "plain"] for c in capitals)     # the control is not inert
```

```
Portugal plain '\n'        +fv ' Lisbon'   +random ' Portugal' few-shot ' Rome'
Russia   plain '\n'        +fv ' Moscow'   +random ' ('        few-shot ' Moscow'
Poland   plain '\n'        +fv ' Poland'   +random '\n'        few-shot ' Warsaw'
Greece   plain '\n'        +fv ' Greece'   +random '\n'        few-shot ' Athens'
Sweden   plain ' Sweden'   +fv ' Sweden'   +random ' Sweden'   few-shot ' Stockholm'
Austria  plain ' Austria'  +fv ' Vienna'   +random ' Germany'  few-shot ' Vienna'
Thailand plain ' Thailand' +fv ' Bangkok'  +random ' Thailand' few-shot ' Bangkok'
{'plain': 0, '+fv': 4, '+random': 0, 'few-shot': 6}
```

The random vector changes three outputs, so on one prompt "the output moved" says
nothing without the control column. A hit where the few-shot column misses
(Portugal) is a result to explain before reporting it: the vector should not
install a capability the model lacks.

This is the simplest form. The published method (Todd et al.) builds the vector
from the mean outputs of a few attention heads chosen by their causal effect:
per-head slicing is in [access-and-modify](access-and-modify.md), ranking heads
by effect in [activation-patching](activation-patching.md).

## Evaluating a steering vector

A concept metric, read for the vector, for its negation and for a matched-norm
random direction, is a dose-response, a negation test and a control in one
forward pass:

```python
def first_ids(words):
    return [model.tokenizer.encode(word, add_special_tokens=False)[0] for word in words]

good = first_ids([" great", " good", " amazing", " wonderful", " fantastic"])
bad = first_ids([" terrible", " bad", " awful", " horrible", " boring"])

generator = torch.Generator().manual_seed(0)
random_direction = torch.randn(direction.shape, generator=generator).to(direction)
random_direction = random_direction / random_direction.norm()

signed = [-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0]
with torch.no_grad(), model.trace() as tracer:
    curve = nnsight.save({})
    for name, vector in (("direction", direction), ("random", random_direction)):
        for alpha in signed:
            with tracer.invoke(prompt):
                model.transformer.h[LAYER].output[:, -1, :] += alpha * scale * vector
                logits = model.output.logits[0, -1]
                curve[name, alpha] = (logits[good].mean() - logits[bad].mean()).item()

steered = [curve["direction", alpha] for alpha in signed]
control = [curve["random", alpha] for alpha in signed]
print("alpha    ", "  ".join(f"{alpha:+5.2f}" for alpha in signed))
print("direction", "  ".join(f"{value:+5.2f}" for value in steered))
print("random   ", "  ".join(f"{value:+5.2f}" for value in control))

assert steered == sorted(steered)                                         # monotonic, both signs
assert max(control) - min(control) < 0.2 * (steered[-1] - steered[0])     # the control stays flat
```

```
alpha     -2.00  -1.00  -0.50  -0.25  +0.00  +0.25  +0.50  +1.00  +2.00
direction -2.41  -1.94  -0.79  +0.12  +1.39  +2.56  +3.99  +5.86  +8.32
random    +1.55  +0.89  +1.15  +1.26  +1.39  +1.60  +1.79  +1.89  +2.02
```

Read this curve beside the generations, because a next-token metric and the text
disagree in two places:

- **It cannot choose α.** The metric is still rising at α = 2.0, where the sweep's
  text has collapsed into a loop.
- **It overstates negation.** The metric falls below zero for negative α, while
  the greedy text at α = -0.5 stays neutral (`The movie was released on May 4,
  2014.`) and at -2.0 breaks. Call a direction bidirectional only when the text
  follows.
