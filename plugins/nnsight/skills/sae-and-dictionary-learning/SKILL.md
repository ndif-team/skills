---
name: sae-and-dictionary-learning
description: Work with sparse autoencoders and dictionary learning in nnsight — attaching an SAE into a model's forward path, reading feature activations, finding max-activating examples, steering with features, measuring reconstruction error, and training a dictionary. Use when analyzing models at the level of interpretable features rather than neurons or directions, when loading a pretrained SAE (sae_lens, dictionary_learning) into a traced model, or when evaluating whether an SAE is any good — including the metrics that look excellent on a broken SAE.
---

# SAEs and Dictionary Learning

An SAE re-expresses activations in an overcomplete, sparse basis where each
dimension is (hopefully) one interpretable feature:

```
features = ReLU(W_enc (x − b_dec))          # sparse, wide
x̂        = W_dec features + b_dec           # reconstruction
```

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

Define it as a normal `torch.nn.Module` and assign it into the envoy tree:

```python
class SparseAutoencoder(torch.nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        # the ReLU lives inside `encoder`, so `encoder.output` is the feature activations
        self.encoder = torch.nn.Sequential(torch.nn.Linear(d_in, d_hidden), torch.nn.ReLU())
        self.decoder = torch.nn.Linear(d_hidden, d_in)

    def encode(self, x):
        return self.encoder(x - self.decoder.bias)

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
reconstruction:

```python
with model.trace() as tracer:
    with tracer.invoke(prompt):
        clean = model.output.logits[0, -1].argmax().save()
    with tracer.invoke(prompt):
        resid = model.transformer.h[LAYER].output
        model.transformer.h[LAYER].output = model.transformer.h[LAYER].sae(resid)
        through_sae = model.output.logits[0, -1].argmax().save()

print(f"clean {model.tokenizer.decode(clean)!r}  through SAE {model.tokenizer.decode(through_sae)!r}")
```

That comparison is the **cheapest quality check there is**: if the model's output
changes when you route through the SAE, the reconstruction is losing something the
model uses. Put a number on it — KL(original ‖ spliced) — between two controls: an
unchanged copy (≈ 0, or the harness itself moves the output) and a zeroed layer:

```python
def splice_kl(replace):
    """KL(original ‖ spliced) at the last position, with h[LAYER].output := replace(it)."""
    with torch.no_grad(), model.trace() as tracer:
        with tracer.invoke(prompt):
            original = model.output.logits[0, -1].log_softmax(-1).save()
        with tracer.invoke(prompt):
            model.transformer.h[LAYER].output = replace(model.transformer.h[LAYER].output)
            spliced = model.output.logits[0, -1].log_softmax(-1).save()
    return float((original.exp() * (original - spliced)).sum())

copy, through, zeroed = (splice_kl(f) for f in (lambda r: r.clone(), model.transformer.h[LAYER].sae,
                                                 torch.zeros_like))
print(f"KL  copy {copy:.1e}   SAE {through:.2f}   zeroed layer {zeroed:.2f}")
assert copy < 1e-4 and through < zeroed
```

```
KL  copy 0.0e+00   SAE 14.03   zeroed layer 16.56
```

This untrained SAE costs 85% of what deleting the layer does.

**Replace the output, do not write into it.** `output[:] = sae(resid)` overwrites
the SAE's own input, and any gradient through the layer then raises
`RuntimeError: one of the variables needed for gradient computation has been
modified by an inplace operation`. `output = sae(resid)` keeps the graph.

To make the SAE's internals observable from later traces, call it inside an
`edit` with `hook=True`. Calling it and discarding the result *observes*; assigning
the result to `.output` inside the edit would *splice* permanently. Reconstruction
error enters the model only when you assign back to `.output`:

```python
with model.edit(inplace=True):
    model.transformer.h[LAYER].sae(model.transformer.h[LAYER].output, hook=True)

with model.trace(prompt):
    observed = model.transformer.h[LAYER].sae.encoder.output.save()   # post-ReLU features
    answer = model.output.logits[0, -1].argmax().save()

print(observed.shape, repr(model.tokenizer.decode(answer)))
model.clear_edits()

assert model.tokenizer.decode(answer) == model.tokenizer.decode(clean)
assert (observed >= 0).all()          # the encoder module contains the ReLU
```

```
torch.Size([1, 10, 2048]) ' Paris'
```

The routing call and the read must be in **different** workers (edit vs trace);
both in one trace body raises `OutOfOrderError`. So does forgetting `hook=True` —
`... already ran past it`, though the submodule never ran: on an attached module,
read that message as "is it hooked?".

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
    with torch.no_grad():                 # collection only — see note
        with model.trace(texts):
            args, kwargs = model.inputs   # the mask the forward actually ran with
            mask = kwargs["attention_mask"].bool().save()
            acts = model.transformer.h[LAYER].output.detach().save()
    return acts[mask].float()

data = collect(corpus)
held = collect(held_out)
unit = data.norm(dim=-1).mean()
data, held = data / unit, held / unit
print("fit vectors:", tuple(data.shape), " held-out vectors:", tuple(held.shape))
```

```
fit vectors: (83, 768)  held-out vectors: (47, 768)
```

**Mask the padding — everywhere you read activations from a batch.** A batched
`trace(...)` left-pads every input to the longest, so a plain
`.reshape(-1, d_model)` over this corpus would hand the optimizer 112 vectors of
which 29 are pad. They are not small: at GPT-2's layer 6 the mean residual norm
is 3118 on a pad position against 92 on a real token past the first, so an
unmasked dictionary spends capacity modelling the padding and every feature
ranking downstream inherits it. Each sentence's *first* real token is the same
kind of outlier (3044) — see the max-activation section. Take the mask from `model.inputs` inside the trace rather than
re-tokenizing: it is the one the forward ran with. The `nnsight` skill →
batching has the padding rule itself.

**Collect under `torch.no_grad()`, and `.detach()` what you save.** A trace runs
with autograd on, so a saved activation arrives with a live `grad_fn` that keeps
the forward graph alive for as long as you hold the tensor. `.detach()` on the
saved tensor is what drops that reference; `torch.no_grad()` stops the graph
being built at all, which lowers peak memory during the forward. `collect()` does
both. Dictionary *training* below still needs gradients, but only through the
SAE, not through the frozen model.

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

Three numbers, on activations the dictionary was **not** fit to — metrics on the
fitting set answer a different question:

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

By the first two metrics this SAE is superb — 99.7% of variance from six
features per token, held up on unseen text — and it is **meaningless**: 2032 of
its 2048 features never fire. Sixteen directions memorized 83 vectors, and every
"feature" you interpreted would be an artifact.

### What each number has to clear

**L0 is a ratio to `d_model`, not a distance from zero.** A dictionary firing 768
features per token on a 768-dimensional activation has re-expressed it, not
decomposed it. For scale, Gemma Scope's residual SAEs average L0 82 on a
`d_model` of 2304. Print the two side by side, as above.

**Reconstruction says nothing on its own.** The null hypothesis is an identity map
in a rotated basis: perfect reconstruction, every "feature" a rotated neuron. Run
it beside the SAE you are evaluating.

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

On a larger held-out set the null also scores loss recovered 1.0000 and no dead
features: every metric comes back perfect except L0, at exactly `d_model`. Add
this row to any evaluation; it is the only control that separates "reconstructs
the activation" from "decomposes the activation".

**Deadness is measured on data you did not fit to.** A memorizing dictionary
looks alive on exactly the vectors it memorized, and deadness is where the
fit/held-out split shows up first, before explained variance moves.

Real SAE training needs millions of tokens, and resampling or auxiliary losses to
revive dead features.

## Using a pretrained SAE

In practice you load someone else's. The wiring is identical — build the module,
load the weights, assign it into the tree, and read it at the point it was
trained on:

<!-- test: skip -->
```python
from sae_lens import SAE                     # or: from dictionary_learning import AutoEncoder

# sae_lens >= 6 returns the SAE alone; 3.x-5.x returned (sae, cfg_dict, sparsity).
pretrained = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre")
if isinstance(pretrained, tuple):
    pretrained = pretrained[0]
model.transformer.h[6].pretrained_sae = pretrained.to(model.device)

with model.trace(prompt):
    resid_pre = model.transformer.h[6].input           # hook_resid_pre = the block's INPUT
    features = model.transformer.h[6].pretrained_sae.encode(resid_pre).save()
```

Check what the SAE was trained on: the **hook point** (`blocks.L.hook_resid_pre`
is `h[L].input`, equal to `h[L-1].output`; `hook_resid_post` is `h[L].output`),
the layer index, and the model revision. An SAE applied at the wrong point
produces plausible-looking garbage.

## Feature analysis

**Max-activating examples** — what makes a feature fire. The trace's own
`input_ids` and mask give the token at the winning position without any
left-padding arithmetic. Exclude the **first real token** as well as the pads:
GPT-2's first-position residual has norm ~3000 against ~92 elsewhere, so an
unexcluded max returns the same features for every prompt:

```python
examples = [
    "The Eiffel Tower is in Paris.",
    "Dogs are loyal pets.",
    "The stock market fell sharply.",
    "She wrote a poem about the sea.",
]

with torch.no_grad():
    with model.trace(examples):
        args, kwargs = model.inputs
        input_ids = kwargs["input_ids"].save()
        real = kwargs["attention_mask"].bool().save()
        resid = model.transformer.h[LAYER].output
        all_features = model.transformer.h[LAYER].sae.encode(resid).detach().save()

first = real & (real.cumsum(dim=1) == 1)                  # each prompt's first real token
keep = real & ~first                                      # pads and the first-token sink out
acts = all_features.masked_fill(~keep[..., None], 0)
feature = int(acts.amax(dim=(0, 1)).argmax())
best_example, best_position = divmod(int(acts[..., feature].argmax()), acts.shape[1])
token = model.tokenizer.decode(input_ids[best_example, best_position])
print(f"feature {feature} fires hardest on {examples[best_example]!r} at token {token!r}")

assert keep[best_example, best_position]
unmasked_top = [all_features[i].amax(0).topk(5).indices.tolist() for i in range(len(examples))]
assert all(top == unmasked_top[0] for top in unmasked_top)   # unmasked: one top-5 for every prompt
```

```
feature 1888 fires hardest on 'The Eiffel Tower is in Paris.' at token 'iff'
```

Unmasked, 47% of the live features here put their top activation on a pad or a
first token, and all four prompts share one top-5.

Interpret features from a *large* corpus, not four sentences, and check the
bottom of the distribution too: a feature that fires on everything is not a
feature.

**Feature steering** — add a decoder column to the residual stream, the causal
test of a feature's meaning. Sweep the coefficient, scaled to the measured
residual norm:

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
  alpha 1.0: ','
  alpha 2.0: '-'
  alpha 4.0: '-'
```

Read the residual norm before the logits: a trace body runs in model order, so
layer 6 after `lm_head` raises `OutOfOrderError`.

Read nothing into *these* tokens: the feature comes from the memorizing
dictionary above, so this is a dose-response curve for an arbitrary direction.
Steering tests a feature's meaning only once the dictionary survives evaluation;
the `model-steering` skill has the matched-norm control that says whether the
direction's content mattered or only its magnitude.

## What to be careful about

- **Feature splitting.** One concept fragments across more features as the
  dictionary grows, so "how many features encode X" is not stable across sizes.
- **Interpretability illusions.** A feature that looks like "the Golden Gate
  Bridge" on its top-20 examples may fire on something more general. Validate
  with steering, the full activation distribution, and examples it does *not*
  fire on.
- **An SAE is a hypothesis fit to activations.** It can be a good sparse code and
  still not carve the model's computation at its joints.

## Related skills

- `nnsight` — attaching modules, `edit`, `hook=True`
- `probing` — supervised directions instead of unsupervised features
- `model-steering` — validating a feature causally
- `ablation` — removing a feature's contribution
