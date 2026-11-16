# SAEs and dictionary learning

A sparse autoencoder re-expresses an activation in a wide basis in which few
dimensions are active at once:

```
features = ReLU(W_enc (x - b_dec))      # wide, mostly zero
x_hat    = W_dec features + b_dec       # the reconstruction
```

```python
import torch
import nnsight
from nnsight import TransformersModel

torch.manual_seed(0)
model = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True)
LAYER, d_model, d_features = 6, model.config.n_embd, 2048
prompt = "The Eiffel Tower is in the city of"
```

## Attach an SAE and read its features

```python
class SparseAutoencoder(torch.nn.Module):
    def __init__(self, d_in, d_hidden):
        super().__init__()
        # the ReLU is inside `encoder`, so `encoder.output` is the feature activations
        self.encoder = torch.nn.Sequential(torch.nn.Linear(d_in, d_hidden), torch.nn.ReLU())
        self.decoder = torch.nn.Linear(d_hidden, d_in)

    def encode(self, x):
        return self.encoder(x - self.decoder.bias)

    def forward(self, x):
        return self.decoder(self.encode(x))

sae = SparseAutoencoder(d_model, d_features).to(model.device)
model.transformer.h[LAYER].sae = sae                  # a module of the model now

with torch.no_grad(), model.trace(prompt):
    resid = model.transformer.h[LAYER].output
    features = model.transformer.h[LAYER].sae.encode(resid).save()

assert tuple(features.shape) == (1, 10, d_features)
```

Assigning a module into the tree attaches it. An activation inside a trace is a
real tensor, so calling the unattached `sae.encode(resid)` there returns the same
values. Attaching is what lets an edit route through the SAE and a trace read its
internals, below.

## Splice it in and measure the cost

Assigning the reconstruction to the site's output makes the rest of the forward
run on it. The KL between the original and the spliced next-token distribution
is what the reconstruction costs the model. Read it between two controls: an
unchanged copy, which must give zero or the harness itself moves the output, and
a zeroed layer, the cost of deleting the site.

```python
def splice_kl(replace):
    """KL(original || spliced) at the last token, with h[LAYER].output = replace(it)."""
    with torch.no_grad(), model.trace() as tracer:
        with tracer.invoke(prompt):
            original = model.output.logits[0, -1].log_softmax(-1).save()
        with tracer.invoke(prompt):
            model.transformer.h[LAYER].output = replace(model.transformer.h[LAYER].output)
            spliced = model.output.logits[0, -1].log_softmax(-1).save()
    return float((original.exp() * (original - spliced)).sum())

copy = splice_kl(lambda r: r.clone())
untrained = splice_kl(model.transformer.h[LAYER].sae)
zeroed = splice_kl(torch.zeros_like)
print(f"KL  copy {copy:.1e}   untrained SAE {untrained:.2f}   zeroed layer {zeroed:.2f}")
assert copy < 1e-4 and untrained < zeroed
```

```
KL  copy 0.0e+00   untrained SAE 14.03   zeroed layer 16.56
```

This untrained SAE costs most of what deleting the layer does. Assign the
reconstruction, as above: the in-place form `output[:] = sae(resid)` overwrites
the SAE's own input, and a backward pass then raises for an SAE whose encoder
reads that input directly ([gradients](gradients.md)).

## Read the SAE's internals in later traces

An edit that calls the attached SAE with `hook=True` makes its submodules
readable in every later trace. Discarding the result observes. Assigning it to
`.output` inside the edit splices on every run.

```python
with model.edit(inplace=True):
    model.transformer.h[LAYER].sae(model.transformer.h[LAYER].output, hook=True)

with torch.no_grad(), model.trace(prompt):
    observed = model.transformer.h[LAYER].sae.encoder.output.save()    # post-ReLU features
    answer = model.output.logits[0, -1].argmax().save()
model.clear_edits()

assert torch.allclose(observed, features, atol=1e-5)
assert model.tokenizer.decode(answer) == " Paris"      # observing leaves the output alone
```

Without `hook=True`, or with the call and the read in one trace body, the read
raises `OutOfOrderError: '...sae.encoder.output.i0' was requested but the model
already ran past it`. The pattern is in
[modules-and-architectures](modules-and-architectures.md).

## Collect activations: drop pads and first tokens

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
    """Residuals at LAYER, token ids, the real-token mask, and the positions to keep."""
    with torch.no_grad(), model.trace(texts):
        args, kwargs = model.inputs                    # what the forward ran with
        ids = kwargs["input_ids"].save()
        real = kwargs["attention_mask"].bool().save()
        acts = model.transformer.h[LAYER].output.float().save()
    keep = real & (real.cumsum(dim=1) > 1)             # not a pad, not a text's first token
    return acts, ids, real, keep

acts, ids, real, keep = collect(corpus)
first, norms = real & ~keep, acts.norm(dim=-1)
print(f"{int((~real).sum())} pads (mean norm {norms[~real].mean():.0f}), {int(first.sum())} first "
      f"tokens ({norms[first].mean():.0f}), {int(keep.sum())} kept ({norms[keep].mean():.0f})")

everything = acts[real]                                # pads dropped, first tokens left in
two_vectors = torch.where(first[..., None], acts[first].mean(0), acts[keep].mean(0))[real]
explained = 1 - ((two_vectors - everything).pow(2).sum()
                 / (everything - everything.mean(0)).pow(2).sum())
print(f"first tokens left in: two constant vectors explain {explained:.3f} of the variance")
assert explained > 0.99
```

```
29 pads (mean norm 3118), 8 first tokens (3044), 75 kept (92)
first tokens left in: two constant vectors explain 0.996 of the variance
```

GPT-2's residual at a pad position and at each text's first token is some thirty
times larger than anywhere else. Left in, those positions are what a dictionary
fits, what explained variance measures, and what every feature ranking returns.
The same mask drops the BOS token on tokenizers that add one. Padding and masks
in general: [batching](batching.md).

## Train a dictionary

```python
held_acts, held_ids, _, held_keep = collect(held_out)
unit = acts[keep].norm(dim=-1).mean()                  # fit at unit scale
data, held = acts[keep] / unit, held_acts[held_keep] / unit

optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
for step in range(800):
    codes = sae.encode(data)
    loss = (sae.decoder(codes) - data).pow(2).sum(-1).mean() + 0.01 * codes.abs().sum(-1).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Training differentiates through the SAE only: the activations were collected
under `no_grad`. The dictionary is fit on activations divided by `unit`, so every
later use divides by it too.

## Evaluate: three numbers, on held-out data, beside a null

```python
@torch.no_grad()
def report(module, activations):
    reconstruction, codes = module(activations), module.encode(activations)
    explained = 1 - ((reconstruction - activations).pow(2).sum()
                     / (activations - activations.mean(0)).pow(2).sum())
    l0 = (codes > 0).float().sum(-1).mean()            # features active per token
    dead = int((codes.max(0).values == 0).sum())       # features that never fire
    return float(explained), float(l0), dead

class RotatedIdentity(torch.nn.Module):
    """The null: exact reconstruction, no decomposition. Every feature is a rotated axis."""
    def __init__(self, d):
        super().__init__()
        self.register_buffer("R", torch.linalg.qr(torch.randn(d, d))[0])

    def encode(self, x):
        z = x @ self.R
        return torch.cat([torch.relu(z), torch.relu(-z)], -1)      # one feature per sign

    def forward(self, x):
        z = self.encode(x)
        return (z[..., :self.R.shape[0]] - z[..., self.R.shape[0]:]) @ self.R.T

null = RotatedIdentity(d_model).to(model.device)
for label, module, activations, width in (("SAE, fit set", sae, data, d_features),
                                          ("SAE, held-out", sae, held, d_features),
                                          ("null, held-out", null, held, 2 * d_model)):
    explained, l0, dead = report(module, activations)
    print(f"{label:15s} explained variance {explained:.3f}   L0 {l0:5.1f} of {d_model}"
          f"   dead {dead}/{width}")

trained = splice_kl(lambda r: sae(r / unit) * unit)
print(f"spliced KL {trained:.2f}   (zeroed layer {zeroed:.2f})")

assert report(sae, data)[0] > 0.99 > 0.5 > report(sae, held)[0]
assert report(sae, held)[2] > 0.9 * d_features and round(report(null, held)[1]) == d_model
```

```
SAE, fit set    explained variance 1.000   L0   3.9 of 768   dead 1917/2048
SAE, held-out   explained variance 0.163   L0   6.4 of 768   dead 1917/2048
null, held-out  explained variance 1.000   L0 768.0 of 768   dead 41/1536
spliced KL 5.04   (zeroed layer 16.56)
```

- **Evaluate on activations the dictionary was not fit to.** On its fitting set
  this SAE reconstructs everything from four features per token. On held-out
  text explained variance is 0.16, and splicing it in costs a third of deleting
  the layer: it memorized 75 vectors.
- **Count dead features.** 1,917 of the 2,048 never fire on either set, so a
  feature read off this dictionary is an artifact.
- **Reconstruction alone says nothing.** The null reconstructs exactly and
  decomposes nothing. Only L0 catches it, at exactly `d_model`: read L0 against
  `d_model`, not against zero.

## A pretrained SAE goes where it was trained

Build or load the module, attach it, and apply it at the site it was trained
on. Where a release names the site as a TransformerLens hook,
`blocks.L.hook_resid_pre` is `h[L].input`, the same tensor as `h[L-1].output`,
and `blocks.L.hook_resid_post` is `h[L].output`:

```python
with torch.no_grad(), model.trace(prompt):
    previous = model.transformer.h[LAYER - 1].output.save()
    resid_pre = model.transformer.h[LAYER].input.save()

assert torch.equal(resid_pre, previous)
```

One site off, an SAE still runs and returns features. A published residual
dictionary for Pythia-70m (512 dimensions, 32,768 features), trained on layer
3's output, explains 0.99 of the variance there with an L0 of 76. On the same
block's input it gives explained variance -19.6 and an L0 of 6,244, with no
error. Measure explained variance and L0 at your site and compare them with the
release's numbers before reading any feature.

The next block is illustrative: it needs the `sae_lens` package and downloads
the SAE.

```python
from sae_lens import SAE

pretrained = SAE.from_pretrained(release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre")
model.transformer.h[6].pretrained_sae = pretrained.to(model.device)

with torch.no_grad(), model.trace(prompt):
    resid_pre = model.transformer.h[6].input           # hook_resid_pre: the block's input
    features = model.transformer.h[6].pretrained_sae.encode(resid_pre).save()
```

## Max-activating examples

Mask before taking a maximum over tokens, with the mask `collect` returns. Here
on the held-out texts:

```python
with torch.no_grad():
    all_features = sae.encode(held_acts / unit)        # (texts, tokens, features), pads included

masked = all_features.masked_fill(~held_keep[..., None], 0)
feature = int(masked.amax(dim=(0, 1)).argmax())
best_text, best_position = divmod(int(masked[..., feature].argmax()), masked.shape[1])
token = model.tokenizer.decode(held_ids[best_text, best_position])
print(f"feature {feature} fires hardest on {held_out[best_text]!r} at token {token!r}")

flat = all_features.flatten(0, 1)
live = flat.amax(0) > 0
on_outlier = (~held_keep.flatten())[flat.argmax(0)][live].float().mean()
print(f"unmasked, {on_outlier:.0%} of the live features peak on a pad or a first token")
assert held_keep[best_text, best_position] and on_outlier > 0.5
```

```
feature 1865 fires hardest on 'He painted a watercolour of the harbour at dawn.' at token ' painted'
unmasked, 90% of the live features peak on a pad or a first token
```

Read a feature from a large corpus and from its whole activation distribution,
including the inputs it stays silent on. A top example from four sentences shows
the masking, not a meaning.

## Steer with a feature, or ablate it

Adding a feature's decoder column to the residual stream is the causal test of
what the feature means. Sweep the coefficient in units of the residual norm at
the site:

```python
column = sae.decoder.weight[:, feature].detach()
direction = column / column.norm()

with torch.no_grad(), model.trace() as tracer:
    picks = nnsight.save({})
    for alpha in (0.0, 0.5, 1.0, 2.0, 4.0):
        with tracer.invoke("The movie was"):
            norm = model.transformer.h[LAYER].output[0, -1].norm()
            model.transformer.h[LAYER].output[:, -1, :] += alpha * norm * direction
            picks[alpha] = model.output.logits[0, -1].argmax()

print({alpha: model.tokenizer.decode(token) for alpha, token in picks.items()})
assert picks[0.0] != picks[4.0]
```

```
{0.0: ' released', 0.5: ' released', 1.0: ' a', 2.0: ',', 4.0: ','}
```

To ablate a feature, subtract its own contribution, its activation times its
decoder column. That equals decoding with the feature zeroed and adding back
what the SAE failed to reconstruct, so nothing but the feature changes. To clamp
it at a value `v`, add `(v - activation) * unit` times the column instead.

```python
text = held_out[best_text]

with torch.no_grad(), model.trace() as tracer:
    with tracer.invoke(text):
        before = model.output.logits[0, -1].log_softmax(-1).save()
    with tracer.invoke(text):
        resid = model.transformer.h[LAYER].output
        activation = sae.encode(resid / unit)[..., feature]        # (1, tokens)
        activation[:, 0] = 0                                       # not fit on first tokens
        model.transformer.h[LAYER].output = resid - (activation * unit)[..., None] * column
        after = model.output.logits[0, -1].log_softmax(-1).save()

kl = float((before.exp() * (before - after)).sum())
print(f"KL after ablating feature {feature}: {kl:.2f}")
assert kl > 0.01
```

```
KL after ablating feature 1865: 0.46
```

This feature comes from the memorizing dictionary above, so both results
describe an arbitrary direction. Steering and ablation test a feature's meaning
only once the dictionary has passed evaluation, and then beside a random
direction of the same norm: [model-steering](model-steering.md).
