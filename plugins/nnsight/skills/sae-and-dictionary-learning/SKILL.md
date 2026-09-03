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

To *measure* an SAE rather than run the model through it, call the attachment and
throw the result away. The call still fires the submodules, so `sae.encoder.output`
is readable from any later trace and the model's own activations are untouched:

```python
with model.edit(inplace=True):
    model.transformer.h[LAYER].sae(model.transformer.h[LAYER].output, hook=True)

with model.trace(prompt):
    observed = model.transformer.h[LAYER].sae.encoder.output.save()
    answer = model.output.logits[0, -1].argmax().save()

print(observed.shape, repr(model.tokenizer.decode(answer)))
model.clear_edits()

assert model.tokenizer.decode(answer) == model.tokenizer.decode(clean)
```

```
torch.Size([1, 10, 2048]) ' Paris'
```

Splicing and observing are separate decisions. Reconstruction error only enters
the model when you assign back to `.output`.

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
held_out = [
    "The Louvre is a museum on the right bank of the Seine.",
    "Rabbits are quiet companions in a small apartment.",
    "Bond yields rose after the central bank statement.",
    "He painted a watercolour of the harbour at dawn.",
]

def collect(texts):
    """Activations at LAYER, pad positions dropped."""
    mask = model.tokenizer(texts, return_tensors="pt", padding=True)["attention_mask"].bool()
    with torch.no_grad():                 # collection only — see note
        with model.trace(texts):
            acts = model.transformer.h[LAYER].output.detach().save()
    return acts[mask.to(acts.device)].float()

data = collect(corpus)
held = collect(held_out)
unit = data.norm(dim=-1).mean()
data, held = data / unit, held / unit
print("fit vectors:", tuple(data.shape), " held-out vectors:", tuple(held.shape))
```

```
fit vectors: (83, 768)  held-out vectors: (47, 768)
```

**Mask the padding.** Everything inside one `trace(...)` is left-padded to the
longest input in the batch, so a plain `.reshape(-1, d_model)` over this corpus
would hand the optimizer 112 vectors of which 29 are pad. They are not small
vectors: at GPT-2's layer 6 the mean residual norm is 3118 on a pad position
against 377 on a real one, 8.3x, so an unmasked dictionary spends real capacity
modelling the padding and every downstream feature ranking is contaminated by it.
The ratio is architecture-specific — pythia-70m's pad residuals are *smaller*
than its real ones (0.4x) — so measure it rather than assuming either way. The
`nnsight` skill → batching has the padding rule itself.

**Collect a held-out set.** Metrics computed on the tensor the dictionary was fit
to answer a different question from the one you want. See the evaluation section.

**Collect under `torch.no_grad()`.** A trace runs with autograd on, so saved
activations arrive with a live `grad_fn` pinning the forward graph. On a
64-sequence GPT-2 capture that is 1133 MiB against 242 MiB, and the multiple
grows with sequence length and model depth, so measure it on your own workload
rather than budgeting from a single number. `.detach()` on the saved tensor is too
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

Three numbers, on activations the dictionary was **not** fit to:

```python
@torch.no_grad()
def report(module, activations):
    reconstruction = module(activations)
    features = module.encode(activations)
    explained = 1 - ((reconstruction - activations).pow(2).sum()
                     / (activations - activations.mean(0)).pow(2).sum())
    l0 = (features > 0).float().sum(-1).mean()
    dead = int((features.max(0).values == 0).sum())
    return float(explained), float(l0), dead

for label, activations in (("fit", data), ("held-out", held)):
    explained, l0, dead = report(sae, activations)
    print(f"{label:9s} explained variance {explained:.3f}"
          f"   L0 {l0:5.1f} of {d_model}   dead {dead}/{d_features}")

assert dead > 0.9 * d_features          # the number this SAE is caught by
```

```
fit       explained variance 0.997   L0   5.7 of 768   dead 2032/2048
held-out  explained variance 0.996   L0   9.3 of 768   dead 1950/2048
```

By the first two metrics this SAE is superb: it reconstructs 99.7% of the variance
using six features per token, and it holds up on text it never saw. It is also
**completely meaningless** — 2032 of its 2048 features never fire. Sixteen
directions memorized 83 activation vectors. Nothing here generalizes, and every
"feature" you interpret would be an artifact.

### What each number has to clear

**L0 is a ratio to `d_model`, not a distance from zero.** A dictionary that fires
768 features per token on a 768-dimensional activation has re-expressed the
activation, not decomposed it. For scale, Gemma Scope's residual SAEs publish an
average L0 of 82 on a `d_model` of 2304, a few percent. Anything approaching
`d_model` is the tell. Print the two side by side, as above, so the comparison is
unavoidable.

**Reconstruction says nothing on its own.** The null hypothesis for any
dictionary is an identity map in a rotated basis: it reconstructs perfectly, and
every "feature" it reports is a rotated neuron. Keep one on hand and run it beside
the SAE you are evaluating.

```python
class RotatedIdentity(torch.nn.Module):
    """The null hypothesis: exact reconstruction, zero decomposition."""
    def __init__(self, d):
        super().__init__()
        rotation, _ = torch.linalg.qr(torch.randn(d, d))
        self.register_buffer("R", rotation)
        self.n_features = 2 * d              # one half per sign

    def encode(self, x):
        z = x @ self.R
        return torch.cat([torch.relu(z), torch.relu(-z)], -1)

    def forward(self, x):
        z = self.encode(x)
        return (z[..., :self.R.shape[0]] - z[..., self.R.shape[0]:]) @ self.R.T

null = RotatedIdentity(d_model).to(model.device)
explained, l0, dead = report(null, held)
print(f"rotated identity: explained variance {explained:.3f}"
      f"   L0 {l0:5.1f} of {d_model}   dead {dead}/{2 * d_model}")

assert explained > 0.999 and round(l0) == d_model     # perfect, and disqualified
```

```
rotated identity: explained variance 1.000   L0 768.0 of 768   dead 39/1536
```

Run the same null over 8,192 held-out GPT-2 layer-6 activations and the rest of
the battery falls too: dead features 0/1536, cosine similarity 1.000, ΔCE −0.000,
and **loss recovered 1.0000**. Every metric people report comes back perfect on
data the null never saw, except L0, which comes back at exactly `d_model`. Add
this row to any evaluation you write. It costs a few lines and it is the only
control that separates "reconstructs the activation" from "decomposes the
activation".

**Deadness is measured on data you did not fit to.** Take a dictionary whose
4,096 decoder rows *are* 4,096 sampled training activations, a lookup table with
no structure at all. It reports 334 dead features (8%) on its fitting set and
2,834 (69%) held out. Explained variance barely moves across that split (0.331 to
0.317) and loss recovered falls 0.82 to 0.63, so deadness is where the split
shows up first. A memorizing dictionary looks alive on exactly the vectors it
memorized.

Real SAE training needs millions of tokens, and resampling or auxiliary losses to
revive dead features. Report all three numbers on held-out activations, plus the
downstream check above: does the model's output survive the reconstruction?

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

**Max-activating examples** — what makes a feature fire. Rank over real tokens
only: a batch of examples is left-padded to its longest member, and on GPT-2 an
SAE fires *hardest* on the pad positions.

```python
examples = [
    "The Eiffel Tower is in Paris.",
    "Dogs are loyal pets.",
    "The stock market fell sharply.",
    "She wrote a poem about the sea.",
]
batch = model.tokenizer(examples, return_tensors="pt", padding=True)
real = batch["attention_mask"].bool().to(model.device)

with model.trace(examples):
    resid = model.transformer.h[LAYER].output
    all_features = model.transformer.h[LAYER].sae.encode(resid).detach().save()

live = (all_features > 0).any(dim=0).any(dim=0).nonzero().flatten()
fabricated = sum(
    not real.reshape(-1)[int(all_features[..., int(f)].reshape(-1).argmax())]
    for f in live
)
print(f"live features: {len(live)}")
print(f"…whose unmasked argmax lands on a pad token: {fabricated}")

feature = int(live[all_features[..., live].amax(dim=(0, 1)).argmax()])
scores = all_features[..., feature].masked_fill(~real, float("-inf"))   # pads out
best_example, best_position = divmod(int(scores.argmax()), scores.shape[1])
ids = model.tokenizer(examples[best_example])["input_ids"]
tokens = [model.tokenizer.decode([i]) for i in ids]
position = best_position - (scores.shape[1] - len(tokens))      # undo left padding
print(f"feature {feature} fires hardest on {examples[best_example]!r} "
      f"at token {tokens[position]!r}")

assert 0 <= position < len(tokens)
```

```
live features: 1678
…whose unmasked argmax lands on a pad token: 327
feature 376 fires hardest on 'Dogs are loyal pets.' at token 'D'
```

Without `masked_fill`, 327 of those 1,678 features would be reported against a pad
position, and the left-padding correction turns negative there — so a
`tokens[max(0, position)]` guard prints token 0 of some example and reads like an
answer. Masking is what makes the printed token real. Bound `position` and let it
raise if it is not.

Interpret features from a *large* corpus of max-activating examples, not four
sentences, and check the bottom of the distribution too, since a feature that
fires on everything is not a feature.

**Feature steering** — add a decoder column to the residual stream. This is the
causal test of a feature's meaning, and it is more targeted than steering with a
raw difference vector. Sweep it, like any other steering coefficient, and scale
against the measured residual norm:

```python
feature_direction = sae.decoder.weight[:, feature].detach()
feature_direction = feature_direction / feature_direction.norm()

with model.trace("The movie was"):
    residual_norm = model.transformer.h[LAYER].output[0, -1].norm().detach().save()
    base = model.output.logits[0, -1].argmax().save()
print(f"base {model.tokenizer.decode(base)!r}   residual norm {float(residual_norm):.1f}")

for alpha in [0.5, 1.0, 2.0, 4.0]:
    with model.trace("The movie was"):
        model.transformer.h[LAYER].output[:, -1, :] += (
            alpha * float(residual_norm) * feature_direction
        )
        steered = model.output.logits[0, -1].argmax().save()
    print(f"  alpha {alpha}: {model.tokenizer.decode(steered)!r}")
```

```
base ' released'   residual norm 93.2
  alpha 0.5: ' a'
  alpha 1.0: ' also'
  alpha 2.0: ' also'
  alpha 4.0: ' "'
```

Read the residual norm before the logits. A trace body runs in model order, so
asking for layer 6 after the run has reached `lm_head` raises `OutOfOrderError`.

Do not read anything into *these* tokens. The feature comes from the dictionary
above, which memorized 83 vectors, so this is a dose-response curve for an
arbitrary direction. Steering only tests a feature's meaning once the dictionary
survives the evaluation section. Use a pretrained SAE for that, and see the
`model-steering` skill for the matched-norm control that says whether the
direction's content mattered or only its magnitude.

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
