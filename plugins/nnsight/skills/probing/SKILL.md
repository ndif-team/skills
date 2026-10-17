---
name: probing
description: Load when training a classifier on activations — asking whether a property (truth, sentiment, an entity) is linearly decodable, at which layer, and whether the model actually uses it. Covers one-pass collection, logistic-regression and difference-in-means probes, layer sweeps, controls (lexical splits, random directions), causal tests along the probe direction (steering, interchange, ablation), and concept erasure (LEACE).
---

# Probing

A probe asks: **is this property linearly decodable from the model's internal
state?** Train a classifier on activations, report accuracy per layer.

The technique is easy. Interpreting it is not — a probe measures what *you* can
decode, not what the model computes or uses. Most of this skill is the controls
that tell those apart.

<!-- test: setup -->
```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)
n_layers = len(model.transformer.h)
hidden = model.config.n_embd
```

## 1. Build a dataset

Vary the property; hold everything else fixed. Templates keep surface form
constant across labels.

```python
positive_words = ["wonderful", "fantastic", "delightful", "excellent", "brilliant",
                  "joyful", "superb", "enjoyable", "lovely", "amazing",
                  "great", "charming", "pleasant", "terrific", "marvelous", "splendid"]
negative_words = ["terrible", "awful", "dreadful", "disgusting", "horrible",
                  "miserable", "dismal", "boring", "dull", "abysmal",
                  "bad", "unpleasant", "annoying", "atrocious", "lousy", "painful"]
templates = ["The movie was {}.", "I found the book {}.",
             "That meal was {}.", "Their performance was {}."]

texts = ([t.format(w) for w in positive_words for t in templates]
         + [t.format(w) for w in negative_words for t in templates])
labels = torch.tensor([1.0] * (len(positive_words) * len(templates))
                      + [0.0] * (len(negative_words) * len(templates)))
print(f"{len(texts)} examples")
```

## 2. Collect activations — one forward pass

The whole dataset and every layer come from a single batched trace:

```python
with torch.no_grad():                              # see the note below — not optional
    with model.trace(texts):
        activations = nnsight.save([
            block.output[:, -1, :].detach().cpu().float() for block in model.transformer.h
        ])

print(len(activations), activations[0].shape)      # 12 layers, [128, 768]
```

**Wrap collection in `torch.no_grad()`.** A trace runs with autograd on, so a saved
activation comes back with a live `grad_fn` that pins the whole forward graph —
several times the activation memory, for bit-identical values, growing with
sequence length and batch size. `.detach()` on the saved tensor does not help: the
graph is already built by then.

For datasets too large for one batch, chunk the texts and concatenate — still one
pass per chunk, never one pass per layer. `tracer.cache()` is the alternative when
you want inputs and outputs of many modules at once.

## 3. Train a probe per layer

Logistic regression with weight decay, no external dependencies. **Split by
lexical item, not by row.** A random row split puts "The movie was wonderful."
in test and "I found the book wonderful." in train, so for any concept the words
carry, test accuracy measures recall of the word list. Hold out whole adjectives:

```python
generator = torch.Generator().manual_seed(0)
word_of_row = torch.arange(len(texts)) // len(templates)       # 0..31, one per adjective
n_pos = len(positive_words)
held_words = set(torch.randperm(n_pos, generator=generator)[:5].tolist())
held_words |= {n_pos + i for i in torch.randperm(n_pos, generator=generator)[:5].tolist()}
is_test = torch.tensor([int(w) in held_words for w in word_of_row])
train_idx, test_idx = torch.where(~is_test)[0], torch.where(is_test)[0]

def train_probe(features, y, y_train_idx, y_test_idx, steps=300, l2=0.02):
    mean, std = features[y_train_idx].mean(0), features[y_train_idx].std(0) + 1e-6
    x_train = (features[y_train_idx] - mean) / std
    x_test = (features[y_test_idx] - mean) / std

    weight = torch.zeros(features.shape[1], requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=0.02)
    for _ in range(steps):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            x_train @ weight + bias, y[y_train_idx]
        ) + l2 * weight.pow(2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    accuracy = (((x_test @ weight + bias) > 0).float() == y[y_test_idx]).float().mean()
    return float(accuracy), weight.detach()     # weight lives in z-scored space

for layer in range(0, n_layers, 2):
    accuracy, _ = train_probe(activations[layer], labels, train_idx, test_idx)
    print(f"layer {layer:2d}  test accuracy {accuracy:.2f}")
```

## 4. Read that result correctly

Layer 0 scores 0.95 on adjectives the probe never saw, and every later layer 1.00.
**High accuracy at layer 0 is a red flag, not a finding.** Layer 0 has barely
processed anything — a probe that succeeds there is reading the *words*: sentiment
adjectives' embeddings already share a direction, and one attention layer copies it
to the final position. With a row split it would also score 1.00 by memorizing
"wonderful" and "terrible" as tokens; the lexical split removes that, not the
lexical semantics.

The shape of a curve tells you what you found:

| Curve | Interpretation |
|---|---|
| High from layer 0 | surface feature — token identity, position, formatting |
| Rises in the middle and plateaus | a genuinely computed representation |
| Only high in late layers | close to the output; may just be the prediction itself |
| Flat at chance | not linearly decodable *here* — try other positions or sublayers |

## 5. Controls

Run all three. They are cheap and each kills a different false positive.

**Shuffled labels** — destroys the real relationship; anything above chance is
memorization capacity, and tells you your probe is too expressive for the dataset:

```python
shuffled = labels[torch.randperm(len(labels), generator=generator)]
control_accuracy, _ = train_probe(activations[6], shuffled, train_idx, test_idx)
print(f"shuffled-label control at layer 6: {control_accuracy:.2f}   (chance = 0.50)")
```

**Held-out template** — train on three templates, test on the fourth. If accuracy
collapses, the probe learned the template, not the property:

```python
per_template = len(templates)
held_out = torch.tensor([i for i in range(len(texts)) if i % per_template == 3])
kept = torch.tensor([i for i in range(len(texts)) if i % per_template != 3])
accuracy, _ = train_probe(activations[6], labels, kept, held_out)
print(f"held-out template at layer 6: {accuracy:.2f}")
```

```
shuffled-label control at layer 6: 0.50   (chance = 0.50)
held-out template at layer 6: 1.00
```

The shuffled control lands at chance — the probe is not simply memorizing, so the
regularization is doing its job. The held-out template still scores 1.00, which
means the probe is not keyed to the template either. Combined with the layer-0
result, the honest summary is: *the probe reads the adjectives' lexical
sentiment*, which generalizes across words and templates and is available from the
embeddings up. That
is a real property of the input, not a discovered internal representation.

To probe something the model must *compute*, the property has to not be readable
off the tokens: entailment between two sentences, whether a stated fact is true,
the eventual answer to a question. Design the dataset so that no surface cue
predicts the label.

**A random direction** of the same norm, scored the same way, is the floor for any
claim that "this direction encodes X". Score each as a 1-D classifier — project,
threshold at the midpoint of the class means — next to the probe's own direction.
`train_probe` returns its weight in z-scored space; **as a direction in activation
space it is `weight / std`** — for steering, a 1-D classifier or an interchange
alike. (The mass-mean directions below are built in activation space already.)

```python
features = activations[6]
std = features[train_idx].std(0) + 1e-6
_, weight = train_probe(features, labels, train_idx, test_idx)
probe_dir = weight / std                                    # z-scored -> activation space

def one_d_accuracy(direction):
    train_proj, test_proj = features[train_idx] @ direction, features[test_idx] @ direction
    pos = train_proj[labels[train_idx] == 1].mean()
    neg = train_proj[labels[train_idx] == 0].mean()
    pred = (torch.sign(pos - neg) * (test_proj - (pos + neg) / 2)) > 0
    return float((pred.float() == labels[test_idx]).float().mean())

random_accs = torch.tensor([one_d_accuracy(r / r.norm() * probe_dir.norm())
                            for r in torch.randn(50, hidden, generator=generator)])
print(f"probe direction {one_d_accuracy(probe_dir):.3f}   "
      f"50 random directions: mean {random_accs.mean():.3f}  max {random_accs.max():.3f}")
```

```
probe direction 1.000   50 random directions: mean 0.659  max 0.950
```

Here a random direction averages 0.66 and reaches 0.95: sentiment adjectives are
so separable at layer 6 that the probe's 1.00 is barely above the floor. On a
concept the tokens do not give away (subject number read at the attractor, the
dataset of the interchange below) random directions average 0.54 at layer 8
against the mass-mean direction's 0.98.
The floor depends on the dataset; measure it, do not assume 0.5. It does the most
work in the causal step below, where it is run against the probe direction on the
same prompt.

## 6. Difference-in-means probes

Instead of fitting, take the mean activation difference between classes. This
"mass-mean" direction is the one used in geometry-of-truth work — it is more robust
on small datasets and, importantly, tends to be **more causally effective** than a
trained probe, because a trained probe is free to use directions that discriminate
without being the ones the model acts on.

```python
def mass_mean_direction(features, index):
    positive = features[index][labels[index] == 1].mean(0)
    negative = features[index][labels[index] == 0].mean(0)
    direction = positive - negative
    return direction / direction.norm()

LAYER = 6
direction = mass_mean_direction(activations[LAYER], train_idx)
projections = activations[LAYER][test_idx] @ direction
threshold = (activations[LAYER][train_idx] @ direction).mean()
accuracy = ((projections > threshold).float() == labels[test_idx]).float().mean()
print(f"difference-in-means accuracy at layer {LAYER}: {float(accuracy):.2f}")
```

## 7. Causal validation — the step that matters

A probe is correlational. To claim the model *uses* the direction, intervene along
it and check the behavior moves:

```python
scale = 0.5
probe_direction = direction.to(model.device)

test_prompt = "The movie was"
good = model.tokenizer.encode(" great")[0]
bad = model.tokenizer.encode(" bad")[0]

def logit_gap(vector):
    """logit(' great') - logit(' bad'), after adding `vector` at LAYER or nothing."""
    with model.trace(test_prompt):
        if vector is not None:
            norm = model.transformer.h[LAYER].output[0, -1].norm()
            model.transformer.h[LAYER].output[:, -1, :] += scale * norm * vector
        logits = model.output.logits[0, -1]
        gap = (logits[good] - logits[bad]).detach().save()
    return float(gap)

baseline = logit_gap(None)
steered = logit_gap(probe_direction)
negated = logit_gap(-probe_direction)

random_generator = torch.Generator().manual_seed(0)
random_gaps = []
for _ in range(8):
    noise = torch.randn(probe_direction.shape, generator=random_generator)
    random_gaps.append(logit_gap((noise / noise.norm()).to(model.device)))

print(f"baseline {baseline:+.3f}   probe {steered:+.3f}   negated {negated:+.3f}")
print(f"random directions: mean {sum(random_gaps) / len(random_gaps):+.3f}"
      f"   max {max(random_gaps):+.3f}")

assert steered > baseline > negated
assert steered > max(random_gaps)
```

```
baseline +1.130   probe +5.203   negated -3.024
random directions: mean +1.020   max +1.846
```

The direction moves the gap by `+4.07` and its negation by `-4.15`; eight random
directions of the same norm land between `-0.02` and `+1.85`, straddling the
baseline. **The last two lines are the result.** Without them the first two
numbers are equally consistent with "any perturbation of this size moves the
logits", and a steering demonstration with no random control is the failure this
skill is about.

If steering along the probe direction moves the behavior and a random direction of
the same norm does not, the direction is causally relevant. If it does not move
anything, you have a decodable feature the model does not read — a real and common
result worth reporting as such. See the `model-steering` skill for scaling and
sweeping.

### Calibrated: interchange along the direction

Steering at `0.5 × norm` is a quick check, but its size is arbitrary and an
unnatural push moves almost anything. The calibrated test works at natural scale:
take a minimal pair, and **swap only the pair's projection along the direction**.
If that 1-D swap recovers as much as patching the whole residual, the direction
carries what that site contributes. It needs a concept with a pair metric, so this
uses subject number ("The keys near the cabinet" → ` are`), read at the last
position, where the token is the attractor and does not give the subject away:

```python
nouns = [(w, w + "s") for w in ("key dog car book girl boy bird house table player doctor "
                                 "road window letter song tree farmer cat student teacher "
                                 "lake king ship wall").split()]
preps = [" near", " behind", " beside", " with", " for"]
ARE, IS = model.tokenizer.encode(" are")[0], model.tokenizer.encode(" is")[0]
gen = torch.Generator().manual_seed(0)

def pairs_for(lemmas, per_lemma=8):
    """(singular-subject, plural-subject) prompts; prep and attractor shared."""
    out = []
    for s in lemmas:
        for _ in range(per_lemma):
            a, p, an = (int(torch.randint(n, (1,), generator=gen)) for n in (len(nouns), len(preps), 2))
            p = preps[p]
            out.append(tuple(f"The {nouns[s][n]}{p} the {nouns[a][an]}" for n in (0, 1)))
    return out

order = torch.randperm(len(nouns), generator=gen).tolist()     # split by lemma
train_pairs, test_pairs = pairs_for(order[:16]), pairs_for(order[16:])
assert all(len(model.tokenizer(t).input_ids) == 5 for pair in train_pairs + test_pairs for t in pair)

def last_resid(prompts):
    with torch.no_grad(), model.trace(prompts):
        acts = nnsight.save([b.output[:, -1, :].detach() for b in model.transformer.h])
        gap = (model.output.logits[:, -1, ARE] - model.output.logits[:, -1, IS]).save()
    return acts, gap

sing_train, _ = last_resid([s for s, _ in train_pairs])
plur_train, _ = last_resid([p for _, p in train_pairs])
corrupt = [s for s, _ in test_pairs]                    # singular: the donor fixes ` are`
clean_resid, clean_gap = last_resid([p for _, p in test_pairs])
_, corrupt_gap = last_resid(corrupt)
hi, lo = float(clean_gap.mean()), float(corrupt_gap.mean())
unembed = model.lm_head.weight[ARE] - model.lm_head.weight[IS]
```

Per layer, one trace: the full-residual patch, the 1-D interchange, a random shift
of the *same per-row magnitude* (the null that matches scale), then mean-ablation of
the direction (projection set to its training mean) on both halves of the test set
for necessity, and the direction's cosine with the ` are` − ` is` unembedding:

```python
def gap_now():
    return model.output.logits[:, -1, ARE] - model.output.logits[:, -1, IS]

def recovery(gap):
    return (float(gap.mean()) - lo) / (hi - lo)

both = corrupt + [p for _, p in test_pairs]
truth = torch.tensor([0] * len(corrupt) + [1] * len(corrupt))
accuracy = lambda gap: float(((gap > 0).long().cpu() == truth).float().mean())

_, base_gap = last_resid(both)
print(f"unablated agreement accuracy {accuracy(base_gap):.3f}")
print("layer  full   1-D   rand-shift   ablated acc (random dir)   cos(d, unembed)")
rows = {}
with torch.no_grad():
    for layer in (3, 5, 7, 9, 10, 11):
        diff = plur_train[layer].mean(0) - sing_train[layer].mean(0)
        d, r = diff / diff.norm(), torch.randn(hidden, generator=gen).to(diff.device)
        r = r / r.norm()
        train_mean = torch.cat([sing_train[layer], plur_train[layer]]).mean(0)
        with model.trace() as tracer:
            out = nnsight.save({})
            with tracer.invoke(corrupt):                               # full residual
                model.transformer.h[layer].output[:, -1, :] = clean_resid[layer]
                out["full"] = gap_now()
            for name, u in (("1d", d), ("rand", r)):                   # 1-D swap / matched null
                with tracer.invoke(corrupt):
                    cur = model.transformer.h[layer].output[:, -1, :]
                    shift = (clean_resid[layer] - cur) @ d
                    if name == "rand":
                        shift = shift.abs()                            # same size, wrong direction
                    model.transformer.h[layer].output[:, -1, :] = cur + shift[:, None] * u
                    out[name] = gap_now()
            for name, u in (("abl", d), ("abl_rand", r)):              # mean-ablation: necessity
                with tracer.invoke(both):
                    cur = model.transformer.h[layer].output[:, -1, :]
                    model.transformer.h[layer].output[:, -1, :] = (
                        cur + (train_mean @ u - cur @ u)[:, None] * u)
                    out[name] = gap_now()
        cos = float(torch.nn.functional.cosine_similarity(d, unembed, dim=0))
        rows[layer] = full, one_d, rand, abl, abl_rand = (
            recovery(out["full"]), recovery(out["1d"]), recovery(out["rand"]),
            accuracy(out["abl"]), accuracy(out["abl_rand"]))
        print(f"L{layer:<5}{full:+.2f}  {one_d:+.2f}   {rand:+.2f}        "
              f"{abl:.3f} ({abl_rand:.3f})             {cos:+.2f}")

assert all(abs(full - one_d) < 0.1 and abs(rand) < 0.1 for full, one_d, rand, _, _ in rows.values())
assert rows[11][3] < 0.6 and min(row[4] for row in rows.values()) > 0.95
```

```
unablated agreement accuracy 1.000
layer  full   1-D   rand-shift   ablated acc (random dir)   cos(d, unembed)
L3    +0.00  +0.00   -0.00        1.000 (1.000)             +0.13
L5    +0.07  +0.06   -0.00        1.000 (1.000)             +0.10
L7    +0.51  +0.46   -0.02        0.922 (1.000)             +0.45
L9    +0.74  +0.69   -0.03        0.766 (1.000)             +0.61
L10   +0.96  +0.95   +0.05        0.641 (1.000)             +0.78
L11   +1.00  +0.97   -0.08        0.523 (1.000)             +0.70
```

- **1-D vs full** — the swap along one direction recovers within 0.05 of the whole
  residual at every layer: at this position, subject number *is* that direction.
- **rand-shift** — the same per-row shift along a random direction stays within
  ±0.08. That is the null that rules out "any push of this size".
- **ablated** — mean-ablating the direction takes agreement from 1.000 to chance by
  layer 11; the same ablation along a random direction changes nothing. Swap shows
  sufficiency, ablation necessity.
- **cos(d, unembed)** — rises from 0.13 to 0.78 by layer 10. Late, the direction
  is largely the output itself; the claim "an abstract number feature" is clean
  only mid-stack, where the effect exists and the cosine is still low.

Read the first column too: recovery is ~0 before layer 7, yet at this position the
mass-mean direction already classifies held-out subject number at 0.71–0.79 over
layers 2–5 (random directions: 0.51). The `interp-experiment-design`
skill covers that trap (position migration).

## 8. Concept erasure — the strong version of the causal test

Erasure removes the concept's linear subspace (LEACE, INLP) from the residual
stream **inside the forward pass**, so everything downstream runs on the erased
state — projecting cached activations afterwards proves nothing about the model.
Judge it with a probe *retrained* on activations collected with the eraser
running, against a rank-matched random eraser, and refit each later site on the
already-erased stream: a per-site eraser fitted on baseline activations lets the
concept back in one block later. A retrained probe that scores *below* chance on
held-out items after erasure (0.40 on one split in the reference) means the signal is gone and
the probe fit noise that anti-generalizes, not that something is inverted. Code,
behavioral cost and measured tables:
[references/concept-erasure.md](references/concept-erasure.md).

## Practical notes

**Which activation.** The residual stream at the last position is the default.
Also worth probing: specific token positions (where the property is mentioned),
attention outputs (what was moved), MLP outputs (what was computed).

**Class balance and dataset size.** Small probing sets overfit trivially. Balance
classes, and treat any result from fewer than a few hundred examples as
provisional.

**Regularization.** With `hidden` ≫ examples, an unregularized probe fits anything.
Keep weight decay on and report it.

## Related skills

- `model-steering` — turning a probe direction into an intervention
- `sae-and-dictionary-learning` — unsupervised features instead of supervised probes
- `activation-patching` — establishing causality component-wise
- `nnsight` — batched activation collection, caching
- `interp-experiment-design` — controls and metric design in general
