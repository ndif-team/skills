# Concept erasure inside the forward pass

Steering asks whether pushing along a probe direction changes the behavior.
Erasure asks the complement: **remove the concept's linear subspace from the
residual stream, let the rest of the network run on the erased state, and see what
breaks.**

Both standard methods are affine maps of the same shape, `x -> mu + (x - mu) @ M`,
which hold the mean fixed and remove variance along the erased directions.
**LEACE** ([Belrose et al., 2023](https://arxiv.org/abs/2306.03819)) is the
smallest such map under which no linear probe beats chance; for a binary concept
it is rank 1 and has no hyperparameters. **INLP**
([Ravfogel et al., 2020](https://arxiv.org/abs/2004.07667)) iterates
probe-then-project, which gives a whole rank family from one fit.

## Setup: a dataset, activations and a probe

The same sentiment dataset and logistic-regression probe as the probing skill:

<!-- test: setup -->
```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)
n_layers = len(model.transformer.h)
hidden = model.config.n_embd
LAYER = 6

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
```

## Fitting LEACE

```python
def fit_leace(features, y):
    """LEACE: returns (M, mu) for x -> mu + (x - mu) @ M."""
    X, z = features.double(), y.double()
    n, d = X.shape
    mu = X.mean(0)
    centered, z_centered = X - mu, (z - z.mean()).unsqueeze(1)

    sigma_xx = (centered.T @ centered) / (n - 1)
    sigma_xz = (centered.T @ z_centered) / (n - 1)

    values, vectors = torch.linalg.eigh(sigma_xx)
    values = values.clamp(min=0)
    keep = values > 1e-8 * values.max()
    whiten = (vectors * torch.where(keep, values.clamp(min=1e-30).rsqrt(),
                                    torch.zeros_like(values))) @ vectors.T
    unwhiten = (vectors * torch.where(keep, values.sqrt(),
                                      torch.zeros_like(values))) @ vectors.T

    basis = whiten @ sigma_xz
    basis = basis / basis.norm()
    return torch.eye(d, dtype=torch.float64) - (unwhiten @ (basis @ basis.T) @ whiten).T, mu
```

## Erase inside the forward pass

**The erasure has to happen inside the forward pass.** Multiplying cached
activations by `M` afterwards and retraining a probe shows only that a projection
defeats a probe on a matrix of numbers — the model computed its logits from the
unerased state. An assignment to `.output` inside the trace is the causal version,
and everything downstream consumes the erased tensor:

```python
def collect(erasers=None):
    """Residual stream at the last position, per layer, with the erasers running."""
    erasers = erasers or {}
    with torch.no_grad():
        with model.trace(texts):
            out = nnsight.save([])
            for site, block in enumerate(model.transformer.h):     # forward order
                if site in erasers:
                    M, mu = erasers[site]
                    block.output = mu + (block.output - mu) @ M
                out.append(block.output[:, -1, :].detach().cpu().float())
    return out

activations = collect()
SITES = list(range(LAYER, n_layers))
device = model.device

def to_device(spec):
    return {s: (M.float().to(device), mu.float().to(device)) for s, (M, mu) in spec.items()}

leace = to_device({s: fit_leace(activations[s][train_idx], labels[train_idx])
                   for s in SITES})

# rank-matched control: remove one arbitrary direction at the same sites
control_generator = torch.Generator().manual_seed(0)
random_erasers = {}
for site in SITES:
    vector = torch.randn(hidden, generator=control_generator)
    vector = vector / vector.norm()
    random_erasers[site] = (torch.eye(hidden) - torch.outer(vector, vector),
                            activations[site][train_idx].mean(0))
random_erasers = to_device(random_erasers)

def decodability(erasers):
    erased = collect(erasers)
    return [round(train_probe(erased[s], labels, train_idx, test_idx)[0], 3) for s in SITES]

print("no erasure        :", decodability(None))
print("LEACE, all sites  :", decodability(leace))
print("LEACE, site 6 only:", decodability({LAYER: leace[LAYER]}))
print("random rank-1     :", decodability(random_erasers))
```

```
no erasure        : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
LEACE, all sites  : [0.462, 0.974, 1.0, 1.0, 0.974, 1.0]
LEACE, site 6 only: [0.462, 1.0, 1.0, 1.0, 0.974, 1.0]
random rank-1     : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

Three things to read off that table:

- **The retrained probe is the anchor.** After any projection the *original*
  probe fails by arithmetic — `P w` is nearly zero. Only a probe refitted on
  activations collected with the erasure running says whether the concept is gone
  for *any* linear probe.
- **The random control removes exactly as many directions and erases nothing.**
  The result is about *which* directions, not how many.
- **One site is not enough, and neither is naive multi-site.** Decodability is
  back at 0.974 one block later. Each site's eraser was fitted on that site's
  *baseline* activations, but once block 6 is erased, block 7 sees a different
  distribution.

## Refit sequentially

Fit greedily — erase 6, collect 7 with 6 erased, fit 7, and so on — and the
erasure holds at every later site:

```python
sequential = {}
for site in SITES:
    seen = collect(to_device(sequential)) if sequential else activations
    sequential[site] = fit_leace(seen[site][train_idx], labels[train_idx])

result = decodability(to_device(sequential))
print("sequential LEACE  :", result)

assert max(result) < 0.6
```

```
sequential LEACE  : [0.462, 0.487, 0.462, 0.462, 0.41, 0.462]
```

**An eraser only erases on the distribution it was fitted on.** LEACE's guarantee
is exactly that, and it is easy to fit on pooled token positions and then report a
last-token result the eraser never touched. Say which distribution you fitted.

**Erasure needs a behavioral measure.** On this dataset the probe reads the
adjective vocabulary, so erasing it is not erasing something the model computes.
Run the same machinery on a concept whose use is visible at the logits, and report
the behavioral change and the perplexity cost against the rank-matched random
control. To make an erasure persistent, apply it under `model.edit(inplace=True)`
(the `model-editing-and-lora` skill).
