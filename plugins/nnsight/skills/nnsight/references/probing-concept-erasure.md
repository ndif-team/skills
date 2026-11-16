# Concept erasure inside the forward pass

Depth for the [probing guide](probing.md). Steering pushes along a direction.
Erasure removes the concept's linear subspace from the residual stream, lets the
rest of the network run on the erased state, and measures what breaks. LEACE
(Belrose et al., 2023) fits an affine map `x -> mu + (x - mu) @ M` that removes
the directions along which the activations covary with the label and keeps the
mean. For a binary label that is one direction.

## Setup

The concept needs a behavior to break. In "The keys near the cabinet", ` are`
minus ` is` separates plural subjects from singular ones, and the last token (the
second noun) does not give the subject away.

```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", task="text-generation", dispatch=True)
n_layers, hidden = len(model.transformer.h), model.config.n_embd

nouns = ("key dog car book girl boy bird house table player doctor road window letter "
         "song tree farmer cat student teacher lake king ship wall").split()
preps = ["near", "behind", "beside", "with", "for"]
ARE, IS = (model.tokenizer.encode(w, add_special_tokens=False)[0] for w in (" are", " is"))
gen = torch.Generator().manual_seed(0)
draw = lambda n: int(torch.randint(n, (1,), generator=gen))

texts, labels, noun_of_row = [], [], []
for subject in range(len(nouns)):
    for _ in range(8):
        tail = f" {preps[draw(len(preps))]} the {nouns[draw(len(nouns))]}{'s' * draw(2)}"
        for plural in (0, 1):
            texts.append(f"The {nouns[subject]}{'s' * plural}{tail}")
            labels.append(float(plural))
            noun_of_row.append(subject)
labels, noun_of_row = torch.tensor(labels), torch.tensor(noun_of_row)
assert all(len(model.tokenizer(t).input_ids) == 5 for t in texts)      # equal lengths: no padding

held_nouns = torch.randperm(len(nouns), generator=gen)[:8]             # split by subject noun
is_test = torch.isin(noun_of_row, held_nouns)
train_idx, test_idx = torch.where(~is_test)[0], torch.where(is_test)[0]

def probe_accuracy(features, y, train, test, steps=300, l2=0.02):
    """Held-out accuracy of a logistic-regression probe trained from scratch."""
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
    return float((((x_test @ weight + bias) > 0).float() == y[test]).float().mean())
```

## Fit LEACE

```python
def fit_leace(features, y):
    """(M, mu) for x -> mu + (x - mu) @ M."""
    X, z = features.double(), y.double()
    n, d = X.shape
    mu = X.mean(0)
    centered, z_centered = X - mu, (z - z.mean()).unsqueeze(1)
    sigma_xx = (centered.T @ centered) / (n - 1)
    sigma_xz = (centered.T @ z_centered) / (n - 1)

    values, vectors = torch.linalg.eigh(sigma_xx)
    values = values.clamp(min=0)
    keep = values > 1e-8 * values.max()
    zeros = torch.zeros_like(values)
    whiten = (vectors * torch.where(keep, values.clamp(min=1e-30).rsqrt(), zeros)) @ vectors.T
    unwhiten = (vectors * torch.where(keep, values.sqrt(), zeros)) @ vectors.T

    basis = whiten @ sigma_xz
    basis = basis / basis.norm()
    M = torch.eye(d, dtype=torch.float64) - (unwhiten @ (basis @ basis.T) @ whiten).T
    return M.float().to(model.device), mu.float().to(model.device)

def random_eraser(mu):
    """The control: remove one arbitrary direction, same rank as LEACE."""
    v = torch.randn(hidden, generator=gen)
    v = v / v.norm()
    return (torch.eye(hidden) - torch.outer(v, v)).to(model.device), mu.to(model.device)
```

## Erase in the forward pass, judge with a retrained probe

Multiplying saved activations by `M` and probing the product says nothing about
the model, which computed its logits from the unerased state. Assign inside the
trace and everything downstream consumes the erased tensor. The erasers are
fitted on last-token activations, so they are applied at the last token only.

```python
def run(erasers=None):
    """Last-token residual per layer and the ` are` - ` is` gap, with the erasers running."""
    erasers = erasers or {}
    with torch.no_grad(), model.trace(texts):
        acts = nnsight.save([])
        for site, block in enumerate(model.transformer.h):          # forward order
            if site in erasers:
                M, mu = erasers[site]
                block.output[:, -1, :] = mu + (block.output[:, -1, :] - mu) @ M
            acts.append(block.output[:, -1, :].float().cpu())
        gap = (model.output.logits[:, -1, ARE] - model.output.logits[:, -1, IS]).cpu().save()
    return acts, gap

SITES = [8, 9, 10, 11]
base, _ = run()

def fit(acts, site):
    return fit_leace(acts[site][train_idx], labels[train_idx])

independent = {site: fit(base, site) for site in SITES}             # each fitted on the unerased run
assert int(torch.linalg.matrix_rank(torch.eye(hidden) - independent[8][0].cpu())) == 1
sequential = {}
for site in SITES:                                                  # each fitted with the earlier ones running
    sequential[site] = fit(run(sequential)[0], site)
random_rank1 = {site: random_eraser(base[site][train_idx].mean(0)) for site in SITES}

def report(name, erasers):
    acts, gap = run(erasers)
    probe = [round(probe_accuracy(acts[site], labels, train_idx, test_idx), 2) for site in SITES]
    held, y = gap[test_idx], labels[test_idx]
    separation = float(held[y == 1].mean() - held[y == 0].mean())
    agreement = float(((held > 0).float() == y).float().mean())
    print(f"{name:28s} probe {probe}   gap {separation:+.2f}   agreement {agreement:.2f}")
    return probe, separation

print(f"retrained probe at sites {SITES}, plural-minus-singular gap, agreement accuracy")
results = {name: report(name, erasers) for name, erasers in (
    ("no erasure", None),
    ("LEACE at site 8 only", {8: independent[8]}),
    ("LEACE, fitted independently", independent),
    ("LEACE, fitted in sequence", sequential),
    ("random rank-1", random_rank1))}

assert max(results["LEACE, fitted in sequence"][0]) < 0.6 < min(results["random rank-1"][0])
assert max(results["LEACE, fitted independently"][0]) > 0.75
assert abs(results["LEACE, fitted in sequence"][1]) < 0.1 * results["random rank-1"][1]
```

```
retrained probe at sites [8, 9, 10, 11], plural-minus-singular gap, agreement accuracy
no erasure                   probe [1.0, 0.99, 1.0, 1.0]   gap +6.33   agreement 0.99
LEACE at site 8 only         probe [0.55, 0.73, 0.91, 0.92]   gap +2.28   agreement 0.73
LEACE, fitted independently  probe [0.55, 0.77, 0.91, 0.89]   gap -0.01   agreement 0.45
LEACE, fitted in sequence    probe [0.55, 0.5, 0.56, 0.52]   gap -0.45   agreement 0.42
random rank-1                probe [1.0, 0.99, 1.0, 1.0]   gap +6.31   agreement 0.99
```

Test rows are the eight held-out subject nouns; 0.5 is chance in both columns.

- **Judge with a retrained probe.** A probe trained before the erasure tests one
  direction. One refitted on activations collected with the eraser running tests
  whether any linear readout is left.
- **The random eraser removes as many directions and erases nothing**, in the
  probe or in the behavior. The result is about which direction, not how many.
- **One site is not enough.** One block after the erased site the concept is
  decodable again (0.73, then 0.91).
- **Refit in sequence.** An eraser fitted on unerased activations meets a
  different distribution once an earlier site is erased, and leaves the concept
  decodable. Fit site 8, collect site 9 with site 8 erased, fit site 9, and so on.
- **The behavior is not the judge.** Independently fitted erasers already bring
  the logit gap to zero while a probe still reads the concept at 0.77 to 0.91.
- **Report the spread over splits.** What is left after sequential erasure is
  noise: over five draws of the held-out nouns the gap lands between -0.45 and
  +0.53 and the probe between 0.48 and 0.57.

For a dataset larger than one batch, run the erasers inside every chunk's trace
and fit each sequential site on the concatenated chunks.

## What else the erasure breaks

An eraser is fitted on one distribution, here the last token of five-token
prompts, and the test above applies it only there. An erasure left on for
ordinary text runs at every position, off that distribution, so measure its cost
under that policy and beside the random eraser. Skip **position 0** (at layer 8
its residual norm is about 3,100 against 130 for later tokens, and no fitted mean
describes it) and **pad positions** (one trace per text below, so there are none;
the mask is in [batching](batching.md)).

```python
neutral = [
    "The history of the city goes back more than two thousand years, when traders first settled along the river.",
    "Photosynthesis is the process by which green plants use sunlight to make food from carbon dioxide and water.",
    "She opened the window, looked at the rain, and decided to stay inside with a cup of tea and a good novel.",
    "The committee will meet next Tuesday to discuss the budget for the coming fiscal year and other matters.",
    "Running a marathon requires months of training, careful nutrition, and plenty of rest between long runs.",
    "After the storm passed, the villagers gathered in the square to assess the damage to their homes and fields.",
]

def neutral_logprobs(erasers, first=1):
    """Next-token log-probs on unrelated text, erasers applied from position `first` on."""
    out = []
    for text in neutral:
        with torch.no_grad(), model.trace(text):
            for site, block in enumerate(model.transformer.h):
                if site in erasers:
                    M, mu = erasers[site]
                    block.output[:, first:, :] = mu + (block.output[:, first:, :] - mu) @ M
            logprobs = model.output.logits[0].log_softmax(-1).cpu().save()
        out.append(logprobs)
    return out

reference = neutral_logprobs({})

def cost(name, erasers, first=1):
    logprobs = neutral_logprobs(erasers, first)
    kl = torch.cat([(ref.exp() * (ref - lp)).sum(-1) for lp, ref in zip(logprobs, reference)])
    top1 = torch.cat([lp.argmax(-1) == ref.argmax(-1) for lp, ref in zip(logprobs, reference)])
    print(f"{name:32s} KL {kl.mean():.3f}   top-1 agreement {top1.float().mean():.3f}")
    return float(kl.mean())

leace_cost = cost("LEACE, positions 1 on", sequential)
random_cost = cost("random rank-1, positions 1 on", random_rank1)
with_first = cost("LEACE, position 0 too", sequential, first=0)
cost("random rank-1, position 0 too", random_rank1, first=0)

assert leace_cost > 5 * random_cost and with_first > 3 * leace_cost
```

```
LEACE, positions 1 on            KL 0.050   top-1 agreement 0.888
random rank-1, positions 1 on    KL 0.004   top-1 agreement 0.936
LEACE, position 0 too            KL 0.312   top-1 agreement 0.808
random rank-1, position 0 too    KL 0.053   top-1 agreement 0.912
```

Erasing number costs something on text that is not about number, and the
random control is what shows the cost belongs to this direction. Including
position 0 multiplies it by six: most of the damage of a careless erasure is the
off-distribution position, not the concept.

To keep an erasure on for every later run, make the same assignment under
`model.edit(inplace=True)` with the same position policy:
[control-flow](control-flow.md).
