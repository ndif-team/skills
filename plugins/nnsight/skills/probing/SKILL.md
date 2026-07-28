---
name: probing
description: Train classifiers on internal activations to test what information is linearly decodable from a model, where in depth it appears, and whether the model actually uses it. Covers activation collection in one forward pass, logistic-regression and difference-in-means (mass-mean) probes, layer sweeps, the controls that distinguish a real finding from a dataset artifact, and causal validation by steering along the probe direction. Use for truth/sentiment/entity probing, geometry-of-truth style analysis, and turning a probe direction into an intervention.
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
with model.trace(texts):
    activations = nnsight.save([
        block.output[:, -1, :].detach().cpu().float() for block in model.transformer.h
    ])

print(len(activations), activations[0].shape)      # 12 layers, [128, 768]
```

For datasets too large for one batch, chunk the texts and concatenate — still one
pass per chunk, never one pass per layer. `tracer.cache()` is the alternative when
you want inputs and outputs of many modules at once.

## 3. Train a probe per layer

Logistic regression with weight decay, no external dependencies:

```python
generator = torch.Generator().manual_seed(0)
order = torch.randperm(len(texts), generator=generator)
split = int(0.7 * len(texts))
train_idx, test_idx = order[:split], order[split:]

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
    return float(accuracy), weight.detach()

for layer in range(0, n_layers, 2):
    accuracy, _ = train_probe(activations[layer], labels, train_idx, test_idx)
    print(f"layer {layer:2d}  test accuracy {accuracy:.2f}")
```

```
layer  0  test accuracy 1.00
layer  2  test accuracy 1.00
layer  4  test accuracy 1.00
...
```

## 4. Read that result correctly

**100% at layer 0 is a red flag, not a finding.** Layer 0 has barely processed
anything — a probe that succeeds there is reading the *token identity*, because
"wonderful" and "terrible" are different tokens with different embeddings. It has
learned the word list, not a sentiment representation.

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
shuffled-label control at layer 6: 0.46   (chance = 0.50)
held-out template at layer 6: 1.00
```

The shuffled control lands at chance — the probe is not simply memorizing, so the
regularization is doing its job. The held-out template still scores 1.00, which
means the probe is not keyed to the template either. Combined with the layer-0
result, the honest summary is: *the probe has learned the adjective vocabulary*,
which generalizes across templates and is available from the embeddings up. That
is a real property of the input, not a discovered internal representation.

To probe something the model must *compute*, the property has to not be readable
off the tokens: entailment between two sentences, whether a stated fact is true,
the eventual answer to a question. Design the dataset so that no surface cue
predicts the label.

**A random direction** of the same norm, scored the same way, is the floor for any
claim that "this direction encodes X".

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

with model.trace() as tracer:
    with tracer.invoke(test_prompt):
        base = model.output.logits[0, -1]
        baseline_gap = (base[good] - base[bad]).detach().save()

    with tracer.invoke(test_prompt):
        norm = model.transformer.h[LAYER].output[0, -1].norm()
        model.transformer.h[LAYER].output[:, -1, :] += scale * norm * probe_direction
        pushed = model.output.logits[0, -1]
        steered_gap = (pushed[good] - pushed[bad]).detach().save()

print(f"logit(' great') − logit(' bad'):  baseline {float(baseline_gap):+.3f}"
      f"   steered {float(steered_gap):+.3f}")
```

If steering along the probe direction moves the behavior and a random direction of
the same norm does not, the direction is causally relevant. If it does not move
anything, you have a decodable feature the model does not read — a real and common
result worth reporting as such. See the `model-steering` skill for scaling and
sweeping.

## Practical notes

**Which activation.** The residual stream at the last position is the default.
Also worth probing: specific token positions (where the property is mentioned),
attention outputs (what was moved), MLP outputs (what was computed).

**Class balance and dataset size.** Small probing sets overfit trivially. Balance
classes, and treat any result from fewer than a few hundred examples as
provisional.

**Regularization.** With `hidden` ≫ examples, an unregularized probe fits anything.
Keep weight decay on and report it.

**Compare architectures carefully.** Probe accuracy is not comparable across models
with different hidden sizes without holding the probe's capacity fixed.

## Related skills

- `model-steering` — turning a probe direction into an intervention
- `sae-and-dictionary-learning` — unsupervised features instead of supervised probes
- `activation-patching` — establishing causality component-wise
- `nnsight` — batched activation collection, caching
- `interp-experiment-design` — controls and metric design in general
