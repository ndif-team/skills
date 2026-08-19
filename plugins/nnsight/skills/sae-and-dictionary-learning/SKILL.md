---
name: sae-and-dictionary-learning
description: Work with sparse autoencoders and dictionary learning in nnsight — attaching an SAE into a model's forward path, reading feature activations, finding max-activating examples, steering with features, measuring reconstruction error, and training a dictionary. Use when analyzing models at the level of interpretable features rather than neurons or directions, when loading a pretrained SAE (sae_lens, dictionary_learning) into a traced model, or when evaluating whether an SAE is any good — including the metrics that look excellent on a broken SAE.
---

# SAEs and Dictionary Learning

Neurons are polysemantic; SAEs try to fix that by re-expressing activations in an
overcomplete, sparse basis where each dimension is (hopefully) one interpretable
feature.

```
features = ReLU(W_enc (x − b_dec))          # sparse, wide
x̂        = W_dec features + b_dec           # reconstruction
```

Two things this skill covers: how to wire an SAE into a traced model (the
nnsight-specific part) and how to tell whether an SAE is telling you anything.

<!-- test: setup -->
```python
import torch
import nnsight
from nnsight import TransformersModel

torch.manual_seed(0)
model = TransformersModel("openai-community/gpt2", dispatch=True)
LAYER = 6
d_model = model.config.n_embd
d_features = 2048
```

## Attaching an SAE to the model

Define it as a normal `torch.nn.Module`, assign it into the envoy tree, and route
activations through it:

```python
class SparseAutoencoder(torch.nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        self.encoder = torch.nn.Linear(d_in, d_hidden)
        self.decoder = torch.nn.Linear(d_hidden, d_in)

    def encode(self, x):
        return torch.relu(self.encoder(x - self.decoder.bias))

    def forward(self, x):
        return self.decoder(self.encode(x))

sae = SparseAutoencoder(d_model, d_features).to(model.device)
model.transformer.h[LAYER].sae = sae            # now part of the tree
```

**Reading features** — apply it to the activation inside the trace:

```python
prompt = "The Eiffel Tower is in the city of"

with model.trace(prompt):
    resid = model.transformer.h[LAYER].output
    features = model.transformer.h[LAYER].sae.encode(resid).detach().save()

print(features.shape, f"active per token: {(features > 0).float().sum(-1).mean():.1f}")
```

**Splicing it into the forward pass** — replace the activation with its
reconstruction, so everything downstream runs on what the SAE can represent:

```python
with model.trace() as tracer:
    with tracer.invoke(prompt):
        clean = model.output.logits[0, -1].argmax().save()
    with tracer.invoke(prompt):
        resid = model.transformer.h[LAYER].output
        model.transformer.h[LAYER].output[:] = model.transformer.h[LAYER].sae(resid)
        through_sae = model.output.logits[0, -1].argmax().save()

print(f"clean {model.tokenizer.decode(clean)!r}  through SAE {model.tokenizer.decode(through_sae)!r}")
```

That comparison is the **cheapest quality check there is**: if the model's output
changes when you route through the SAE, the reconstruction is losing something the
model uses.

To make the SAE permanent — and its internals observable from other traces — put
the routing in an `edit` and pass `hook=True`:

```python
with model.edit(inplace=True):
    resid = model.transformer.h[LAYER].output
    model.transformer.h[LAYER].output[:] = model.transformer.h[LAYER].sae(resid, hook=True)

with model.trace(prompt):
    encoder_out = model.transformer.h[LAYER].sae.encoder.output.save()   # observable

print(encoder_out.shape)
model.clear_edits()
```

The routing call and the read must be in **different** workers (edit vs trace) —
doing both in one trace body raises `OutOfOrderError`. See the `nnsight` skill →
modules and architectures.

## Training a dictionary

Collect activations, then fit. The nnsight part is one trace:

```python
corpus = [
    "The Eiffel Tower is in Paris and it is very tall.",
    "Dogs and cats are common pets in many homes.",
    "The stock market fell sharply after the announcement.",
    "She wrote a beautiful poem about the ocean.",
    "Python is a programming language used for data science.",
    "The recipe calls for two cups of flour and sugar.",
    "Scientists discovered a new species in the rainforest.",
    "He played guitar in a band during college.",
]

with torch.no_grad():                     # collection only — see note
    with model.trace(corpus):
        collected = model.transformer.h[LAYER].output.detach().save()

data = collected.reshape(-1, d_model).float()
data = data / data.norm(dim=-1, keepdim=True).mean()
print("activation vectors:", data.shape)
```

**Collect under `torch.no_grad()`.** A trace runs with autograd on, so saved
activations arrive with a live `grad_fn` pinning the forward graph — measured at
**3.6x peak memory** for a pure capture. `.detach()` on the saved tensor is too
late; the graph was already built. Dictionary *training* below still needs
gradients, but only through the SAE, not through the frozen model.

```python

optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
for step in range(800):
    reconstruction = sae(data)
    features = sae.encode(data)
    loss = ((reconstruction - data).pow(2).sum(-1).mean()
            + 0.01 * features.abs().sum(-1).mean())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Evaluating it — and why this one is worthless

```python
with torch.no_grad():
    reconstruction = sae(data)
    features = sae.encode(data)

    residual_variance = (reconstruction - data).pow(2).sum()
    total_variance = (data - data.mean(0)).pow(2).sum()
    explained = 1 - residual_variance / total_variance
    l0 = (features > 0).float().sum(-1).mean()
    dead = int((features.max(0).values == 0).sum())

print(f"explained variance {float(explained):.3f}")
print(f"L0 (active features per token) {float(l0):.1f}")
print(f"dead features {dead}/{d_features}")
```

```
explained variance 0.999
L0 (active features per token) 2.6
dead features 2041/2048
```

By the two headline metrics this SAE is superb: it reconstructs 99.9% of the
variance using 2.6 features per token. It is also **completely meaningless** —
2041 of its 2048 features never fire. Seven directions memorized 104 activation
vectors. Nothing here generalizes, and every "feature" you interpret would be an
artifact.

That is the point of the third metric. Real SAE training needs millions of tokens,
resampling or auxiliary losses to revive dead features, and evaluation on held-out
activations. Always report all three numbers together, plus the downstream check
above (does the model's output survive the reconstruction?).

## Using a pretrained SAE

In practice you load someone else's. The wiring is identical — build the module,
load the weights, assign it into the tree:

<!-- test: skip -->
```python
from sae_lens import SAE                     # or: from dictionary_learning import AutoEncoder

sae, config, _ = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre")
sae = sae.to(model.device)
model.transformer.h[6].sae = sae

with model.trace(prompt):
    resid = model.transformer.h[6].output
    features = model.transformer.h[6].sae.encode(resid).save()
```

Check what the SAE was trained on: the **hook point** (`resid_pre` is the block's
input, `resid_post` its output), the layer index, and the model revision. An SAE
applied at the wrong point produces plausible-looking garbage.

## Feature analysis

**Max-activating examples** — what makes a feature fire:

```python
examples = [
    "The Eiffel Tower is in Paris.",
    "Dogs are loyal pets.",
    "The stock market fell sharply.",
    "She wrote a poem about the sea.",
]

with model.trace(examples):
    resid = model.transformer.h[LAYER].output
    all_features = model.transformer.h[LAYER].sae.encode(resid).detach().save()

live = (all_features > 0).any(dim=(0, 1)).nonzero().flatten()
feature = int(live[0])
scores = all_features[..., feature]
best_example, best_position = divmod(int(scores.argmax()), scores.shape[1])
ids = model.tokenizer(examples[best_example])["input_ids"]
tokens = [model.tokenizer.decode([i]) for i in ids]
position = best_position - (scores.shape[1] - len(tokens))      # undo left padding
print(f"feature {feature} fires hardest on {examples[best_example]!r} "
      f"at token {tokens[max(0, position)]!r}")
```

Interpret features from a *large* corpus of max-activating examples, not four
sentences — and check the bottom of the distribution too, since a feature that
fires on everything is not a feature.

**Feature steering** — add a decoder column to the residual stream. This is the
causal test of a feature's meaning, and it is more targeted than steering with a
raw difference vector:

```python
feature_direction = sae.decoder.weight[:, feature].detach()

with model.trace() as tracer:
    with tracer.invoke("The movie was"):
        base = model.output.logits[0, -1].argmax().save()
    with tracer.invoke("The movie was"):
        norm = model.transformer.h[LAYER].output[0, -1].norm()
        model.transformer.h[LAYER].output[:, -1, :] += 0.5 * norm * (
            feature_direction / feature_direction.norm()
        )
        steered = model.output.logits[0, -1].argmax().save()

print(f"base {model.tokenizer.decode(base)!r}  steered {model.tokenizer.decode(steered)!r}")
```

See the `model-steering` skill for coefficient sweeps and controls.

## What to be careful about

**Reconstruction error is not free.** Everything downstream of a spliced-in SAE
runs on an approximation. Report the model's behavior with and without it.

**Dead features and feature splitting.** Dead features waste capacity; feature
splitting means one concept fragments across many features as the dictionary
grows, so "how many features encode X" is not stable across dictionary sizes.

**Interpretability illusions.** A feature that looks like "the Golden Gate Bridge"
on its top-20 examples may be firing on something more general. Validate with
steering, with the full activation distribution, and with examples the feature
does *not* fire on.

**An SAE is a hypothesis about the model, fit to activations.** It can be a good
sparse code and still not carve the model's computation at its joints.

## Related skills

- `nnsight` — attaching modules, `edit`, `hook=True`
- `probing` — supervised directions instead of unsupervised features
- `model-steering` — validating a feature causally
- `ablation` — removing a feature's contribution
