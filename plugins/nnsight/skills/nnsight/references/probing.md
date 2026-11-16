# Probing

A probe is a classifier trained on activations. Its accuracy says a property can
be decoded at a site, not that the model computes it there or uses it. The
controls and the causal tests below are what tell those apart.

```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True)
n_layers, hidden = len(model.transformer.h), model.config.n_embd

positive_words = ["wonderful", "fantastic", "delightful", "excellent", "brilliant",
                  "joyful", "superb", "enjoyable", "lovely", "amazing",
                  "great", "charming", "pleasant", "terrific", "marvelous", "splendid"]
negative_words = ["terrible", "awful", "dreadful", "disgusting", "horrible",
                  "miserable", "dismal", "boring", "dull", "abysmal",
                  "bad", "unpleasant", "annoying", "atrocious", "lousy", "painful"]
words = positive_words + negative_words
templates = ["The movie was {}.", "I found the book {}.",
             "That meal was {}.", "Their performance was {}."]
texts = [t.format(w) for w in words for t in templates]      # 4 rows per adjective
labels = torch.tensor([1.0] * (len(positive_words) * len(templates))
                      + [0.0] * (len(negative_words) * len(templates)))
```

Templates vary the property and hold the surface form fixed.

## Collect every layer in one pass

```python
with torch.no_grad(), model.trace(texts):
    activations = nnsight.save([block.output[:, -1, :].float().cpu()
                                for block in model.transformer.h])

assert len(activations) == n_layers and tuple(activations[0].shape) == (len(texts), hidden)
```

A dataset too large for one batch is one pass per chunk of texts, never one pass
per layer. The residual stream at the last token is the default site. Any other
position needs a per-row index in a padded batch ([batching](batching.md)).
`block.attn.output[0]` (what attention moved in) and `block.mlp.output` are
collected the same way, each read before its block's `.output`.

## A probe per layer, split by lexical item

Logistic regression on z-scored activations, in plain torch. **Hold out whole
lexical items, not rows.** A row split puts "The movie was wonderful." in test
and "I found the book wonderful." in train, and test accuracy then measures
recall of the word list.

```python
generator = torch.Generator().manual_seed(0)
word_of_row = torch.arange(len(texts)) // len(templates)
n_pos = len(positive_words)
held_words = set(torch.randperm(n_pos, generator=generator)[:5].tolist())
held_words |= {n_pos + i for i in torch.randperm(n_pos, generator=generator)[:5].tolist()}
is_test = torch.tensor([int(w) in held_words for w in word_of_row])
train_idx, test_idx = torch.where(~is_test)[0], torch.where(is_test)[0]

def train_probe(features, y, train, test, steps=300, l2=0.02):
    mean, std = features[train].mean(0), features[train].std(0) + 1e-6
    x_train, x_test = (features[train] - mean) / std, (features[test] - mean) / std
    weight = torch.zeros(features.shape[1], requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([weight, bias], lr=0.02)
    for _ in range(steps):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            x_train @ weight + bias, y[train]) + l2 * weight.pow(2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    accuracy = (((x_test @ weight + bias) > 0).float() == y[test]).float().mean()
    return float(accuracy), weight.detach()          # the weight is in z-scored space

per_layer = [train_probe(activations[layer], labels, train_idx, test_idx)[0]
             for layer in range(n_layers)]
print("held-out accuracy per layer:", [round(a, 2) for a in per_layer])
```

```
held-out accuracy per layer: [0.95, 0.95, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

**High accuracy at the first layer is a red flag.** The last token of every row
is the same ".", so whatever block 0 holds there was copied from the adjective by
one attention layer: the probe reads the words' lexical sentiment, a property of
the input. To probe something the model computes, build a dataset whose label no
token gives away (whether a stated fact is true, the number of a subject read at
a later noun as in the [interchange](#calibrated-interchange-along-the-direction)
below). Accuracy that appears only in the last layers may be the prediction
itself: check the direction's cosine with the answer's unembedding, as that
example does. Accuracy at chance means not linearly decodable at this site.

## Controls

**Shuffled labels**: give every adjective a random label, the same on all its
rows, and train again. **Held-out template**: train on three templates and test
on the fourth. A collapse there means the probe learned the template.

```python
def random_word_labels():
    per_word = labels[::len(templates)][torch.randperm(len(words), generator=generator)]
    return per_word[word_of_row]

rows = torch.randperm(len(texts), generator=generator)
row_train, row_test = rows[40:], rows[:40]
by_row, by_word = [], []
for _ in range(20):
    y = random_word_labels()
    by_row.append(train_probe(activations[6], y, row_train, row_test)[0])
    by_word.append(train_probe(activations[6], y, train_idx, test_idx)[0])
by_row, by_word = torch.tensor(by_row), torch.tensor(by_word)
print(f"random word labels, row split:     mean {by_row.mean():.2f}")
print(f"random word labels, lexical split: mean {by_word.mean():.2f}  max {by_word.max():.2f}")

fourth = torch.arange(len(texts)) % len(templates) == 3
accuracy, _ = train_probe(activations[6], labels, torch.where(~fourth)[0], torch.where(fourth)[0])
print(f"held-out template: {accuracy:.2f}")

assert by_row.mean() > 0.8 > by_word.mean()
```

```
random word labels, row split:     mean 0.90
random word labels, lexical split: mean 0.52  max 0.73
held-out template: 1.00
```

With a row split the probe scores 0.90 on labels that mean nothing: it recalls
words. With the lexical split the same labels average chance, and their spread
(up to 0.73 here) is the band a real accuracy has to clear. The band is set by
the ten held-out adjectives, not by the 40 rows.

## Directions: the probe's, mass-mean, and the random floor

A direction from the probe is `weight / std`: `train_probe` fits in z-scored
space, and steering, projection and interchange all happen in activation space.
The difference of class means (the mass-mean direction) needs no fit and is in
activation space already. Score any direction as a 1-D classifier (project,
threshold at the midpoint of the class means) beside random directions of the
same norm. That floor is not 0.5:

```python
LAYER = 6
features = activations[LAYER]
std = features[train_idx].std(0) + 1e-6
probe_dir = train_probe(features, labels, train_idx, test_idx)[1] / std
probe_dir = probe_dir / probe_dir.norm()

positive = features[train_idx][labels[train_idx] == 1].mean(0)
negative = features[train_idx][labels[train_idx] == 0].mean(0)
mass_mean = (positive - negative) / (positive - negative).norm()

def one_d_accuracy(direction):
    train_proj, test_proj = features[train_idx] @ direction, features[test_idx] @ direction
    pos = train_proj[labels[train_idx] == 1].mean()
    neg = train_proj[labels[train_idx] == 0].mean()
    pred = (torch.sign(pos - neg) * (test_proj - (pos + neg) / 2)) > 0
    return float((pred.float() == labels[test_idx]).float().mean())

random_dirs = torch.randn(50, hidden, generator=generator)
random_accs = torch.tensor([one_d_accuracy(r / r.norm()) for r in random_dirs])
print(f"probe {one_d_accuracy(probe_dir):.2f}   mass-mean {one_d_accuracy(mass_mean):.2f}   "
      f"50 random: mean {random_accs.mean():.2f}  max {random_accs.max():.2f}")
```

```
probe 1.00   mass-mean 1.00   50 random: mean 0.68  max 1.00
```

Sentiment adjectives are so separable at this layer that random directions
average 0.68 and one of the fifty matches the probe. Measure the floor on your
dataset before claiming a direction encodes anything.

## Causal test: push along the direction

Decoding is correlational. To claim the model uses a direction, add it and watch
the behavior, next to random directions of the same norm:

```python
good, bad = (model.tokenizer.encode(w, add_special_tokens=False)[0] for w in (" great", " bad"))

def logit_gap(vector, scale=0.5):
    """logit(' great') - logit(' bad'), with `vector` added at LAYER at half the residual's norm."""
    with torch.no_grad(), model.trace("The movie was"):
        if vector is not None:
            norm = model.transformer.h[LAYER].output[0, -1].norm()
            model.transformer.h[LAYER].output[:, -1, :] += scale * norm * vector.to(norm.device)
        logits = model.output.logits[0, -1]
        gap = (logits[good] - logits[bad]).save()
    return float(gap)

baseline = logit_gap(None)
random_gaps = [logit_gap(r / r.norm()) for r in random_dirs[:8]]
print(f"baseline {baseline:+.2f}")
print(f"mass-mean {logit_gap(mass_mean):+.2f} / negated {logit_gap(-mass_mean):+.2f}")
print(f"probe     {logit_gap(probe_dir):+.2f} / negated {logit_gap(-probe_dir):+.2f}")
print(f"8 random  {min(random_gaps):+.2f} to {max(random_gaps):+.2f}")

assert logit_gap(mass_mean) > max(random_gaps) and logit_gap(-mass_mean) < min(random_gaps)
```

```
baseline +1.13
mass-mean +5.20 / negated -3.02
probe     +4.18 / negated -2.32
8 random  +0.68 to +1.82
```

The random row is the result. Without it the rows above are equally consistent
with "any push of this size moves the logits". At the same norm the mass-mean
direction moves the gap further than the fitted probe's (+4.07 against +3.05):
the best classifier is not the best handle, so test both. A direction that
decodes and does not move the behavior is a decodable feature the model does not
read at that site, and is reported as that. Scaling and sweeping the coefficient:
[model-steering](model-steering.md).

## Calibrated: interchange along the direction

A push of half the residual norm is an arbitrary size. The calibrated test works
at natural scale: take a minimal pair and swap only the pair's difference along
the direction. If that 1-D swap recovers what patching the whole residual
recovers, the direction carries what the site contributes. It needs a pair
metric, so this uses subject number ("The keys near the cabinet" continues with
` are`), read at the last token, which is the other noun and does not give the
subject away:

```python
nouns = ("key dog car book girl boy bird house table player doctor road window letter "
         "song tree farmer cat student teacher lake king ship wall").split()
preps = ["near", "behind", "beside", "with", "for"]
ARE, IS = (model.tokenizer.encode(w, add_special_tokens=False)[0] for w in (" are", " is"))
gen = torch.Generator().manual_seed(0)

def pairs_for(subjects, per_subject=8):
    """(singular-subject, plural-subject) prompts that share preposition and second noun."""
    pairs = []
    for subject in subjects:
        for _ in range(per_subject):
            other, prep, plural = (int(torch.randint(n, (1,), generator=gen))
                                   for n in (len(nouns), len(preps), 2))
            tail = f" {preps[prep]} the {nouns[other]}{'s' * plural}"
            pairs.append((f"The {nouns[subject]}{tail}", f"The {nouns[subject]}s{tail}"))
    return pairs

order = torch.randperm(len(nouns), generator=gen).tolist()        # split by subject noun
train_pairs, test_pairs = pairs_for(order[:16]), pairs_for(order[16:])
assert all(len(model.tokenizer(t).input_ids) == 5 for pair in train_pairs + test_pairs for t in pair)

def last_resid(prompts):
    with torch.no_grad(), model.trace(prompts):
        acts = nnsight.save([block.output[:, -1, :] for block in model.transformer.h])
        gap = (model.output.logits[:, -1, ARE] - model.output.logits[:, -1, IS]).save()
    return acts, gap

sing_train, _ = last_resid([s for s, _ in train_pairs])
plur_train, _ = last_resid([p for _, p in train_pairs])
singular, plural = [s for s, _ in test_pairs], [p for _, p in test_pairs]
sing_resid, sing_gap = last_resid(singular)                       # the runs to repair
plur_resid, plur_gap = last_resid(plural)                         # the donors
lo, hi = float(sing_gap.mean()), float(plur_gap.mean())
unembed = model.lm_head.weight[ARE] - model.lm_head.weight[IS]
```

Per layer, one trace holds the full-residual patch, the 1-D swap, the same shift
along a random direction (the null of matched size), and a mean-ablation of the
direction (its projection set to the training mean) on all test prompts, also
along the random direction. `decodes` is the direction's 1-D accuracy on the
held-out prompts at the same site:

```python
def gap_now():
    return model.output.logits[:, -1, ARE] - model.output.logits[:, -1, IS]

def recovery(gap):                                    # 0 = singular run, 1 = plural run
    return (float(gap.mean()) - lo) / (hi - lo)

both = singular + plural
truth = torch.tensor([0] * len(singular) + [1] * len(plural))

def agreement(gap):
    return float(((gap > 0).long().cpu() == truth).float().mean())

print("layer  decodes   full    1-D  random   ablated (random)   cos(d, unembed)")
table = {}
with torch.no_grad():
    for layer in (3, 5, 7, 9, 10, 11):
        diff = plur_train[layer].mean(0) - sing_train[layer].mean(0)
        d = diff / diff.norm()                                    # mass-mean direction
        r = torch.randn(hidden, generator=gen).to(d.device)
        r = r / r.norm()
        train_mean = torch.cat([sing_train[layer], plur_train[layer]]).mean(0)
        with model.trace() as tracer:
            out = nnsight.save({})
            with tracer.invoke(singular):                         # the whole residual
                model.transformer.h[layer].output[:, -1, :] = plur_resid[layer]
                out["full"] = gap_now()
            for name, u in (("1d", d), ("random", r)):            # the pair's gap along d, moved along u
                with tracer.invoke(singular):
                    cur = model.transformer.h[layer].output[:, -1, :]
                    shift = (plur_resid[layer] - cur) @ d
                    model.transformer.h[layer].output[:, -1, :] = cur + shift[:, None] * u
                    out[name] = gap_now()
            for name, u in (("ablated", d), ("ablated_random", r)):
                with tracer.invoke(both):
                    cur = model.transformer.h[layer].output[:, -1, :]
                    model.transformer.h[layer].output[:, -1, :] = (
                        cur + ((train_mean - cur) @ u)[:, None] * u)
                    out[name] = gap_now()
        middle = (plur_train[layer].mean(0) + sing_train[layer].mean(0)) @ d / 2
        decodes = float(torch.cat([plur_resid[layer] @ d > middle,
                                   sing_resid[layer] @ d < middle]).float().mean())
        cos = float(torch.nn.functional.cosine_similarity(d, unembed, dim=0))
        row = table[layer] = dict(
            decodes=decodes, full=recovery(out["full"]), one_d=recovery(out["1d"]),
            random=recovery(out["random"]), ablated=agreement(out["ablated"]),
            ablated_random=agreement(out["ablated_random"]))
        print(f"L{layer:<5} {row['decodes']:.2f}    {row['full']:+.2f}  {row['one_d']:+.2f}   "
              f"{row['random']:+.2f}     {row['ablated']:.2f} ({row['ablated_random']:.2f})"
              f"          {cos:+.2f}")

assert all(abs(row["full"] - row["one_d"]) < 0.1 and abs(row["random"]) < 0.1 for row in table.values())
assert table[11]["ablated"] < 0.6 and min(row["ablated_random"] for row in table.values()) > 0.95
assert table[5]["decodes"] > 0.7 and table[5]["full"] < 0.1       # decodable, and not used here
```

```
layer  decodes   full    1-D  random   ablated (random)   cos(d, unembed)
L3     0.75    +0.00  +0.00   -0.00     1.00 (1.00)          +0.13
L5     0.79    +0.07  +0.06   -0.00     1.00 (1.00)          +0.10
L7     0.98    +0.51  +0.46   -0.02     0.92 (1.00)          +0.45
L9     0.99    +0.74  +0.69   -0.03     0.77 (1.00)          +0.61
L10    1.00    +0.96  +0.95   +0.05     0.64 (1.00)          +0.78
L11    0.87    +1.00  +0.97   -0.08     0.52 (1.00)          +0.70
```

- **1-D against full.** The swap along one direction recovers what the whole
  residual does at every layer: at this position, subject number is that
  direction.
- **random.** The same shift along a random direction recovers nothing, which
  rules out "any change of this size".
- **ablated.** Mean-ablating the direction takes agreement to chance by the last
  layer and the random direction changes nothing. The swap shows sufficiency,
  the ablation necessity.
- **cos(d, unembed)** rises with depth. By layer 10 the direction is largely
  the output itself (0.78), so "a number feature" is a cleaner claim at layer 7,
  where the effect has appeared and the cosine is 0.45.
- **decodes against full.** At layers 3 and 5 the direction classifies held-out
  subject number at 0.75 and 0.79, and patching the whole residual there
  recovers nothing. Decodable at a site is not used at that site: pair every
  probe-site claim with this test
  ([interp-experiment-design](interp-experiment-design.md)).

## Erasing the concept

The complement of pushing along a direction is removing the concept's subspace
inside the forward pass (LEACE) and measuring what breaks, judged by a retrained
probe against a random eraser of the same rank:
[probing-concept-erasure](probing-concept-erasure.md).
