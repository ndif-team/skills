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

generator = torch.Generator().manual_seed(0)               # split by adjective, as there
word_of_row = torch.arange(len(texts)) // len(templates)
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
and everything downstream consumes the erased tensor. The erasers are fitted on
last-position activations, so they are applied at the last position only (the
position policy is spelled out under the behavioral cost below):

```python
def collect(erasers=None, prompts=texts):
    """Residual stream at the last position, per layer, with the erasers running."""
    erasers = erasers or {}
    with torch.no_grad():
        with model.trace(prompts):
            out = nnsight.save([])
            for site, block in enumerate(model.transformer.h):     # forward order
                if site in erasers:
                    M, mu = erasers[site]
                    last = block.output[:, -1, :]                   # the fitted position only
                    block.output[:, -1, :] = mu + (last - mu) @ M
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
naive = decodability(leace)
print("LEACE, all sites  :", naive)
print("LEACE, site 6 only:", decodability({LAYER: leace[LAYER]}))
print("random rank-1     :", decodability(random_erasers))
```

```
no erasure        : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
LEACE, all sites  : [0.7, 1.0, 1.0, 1.0, 0.975, 0.975]
LEACE, site 6 only: [0.7, 1.0, 1.0, 1.0, 1.0, 1.0]
random rank-1     : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
```

Test rows are held-out adjectives (40 rows, so ±0.08 is noise). Three things to
read off that table:

- **The retrained probe is the anchor.** After any projection the *original*
  probe fails by arithmetic — `P w` is nearly zero. Only a probe refitted on
  activations collected with the erasure running says whether the concept is gone
  for *any* linear probe.
- **The random control removes exactly as many directions and erases nothing.**
  The result is about *which* directions, not how many.
- **One site is not enough, and neither is naive multi-site.** Decodability is
  back at 1.0 one block later. Each site's eraser was fitted on that site's
  *baseline* activations, but once block 6 is erased, block 7 sees a different
  distribution.

## Refit sequentially

Fit greedily — erase 6, collect 7 with 6 erased, fit 7, and so on — and the
concept stops coming back:

```python
sequential = {}
for site in SITES:
    seen = collect(to_device(sequential)) if sequential else activations
    sequential[site] = fit_leace(seen[site][train_idx], labels[train_idx])

result = decodability(to_device(sequential))
print("sequential LEACE  :", result)

assert max(result) < 0.75 and min(naive[1:4]) == 1.0
```

```
sequential LEACE  : [0.7, 0.725, 0.525, 0.525, 0.525, 0.6]
```

**An eraser only erases on the distribution it was fitted on.** LEACE's guarantee
covers the rows it was fitted on. With a random row split the same code gives
0.462 at every site; on held-out adjectives, sites 6–7 keep 0.70–0.73 on this
split, and two other held-out-word draws give 0.40–0.58 at the same sites.
Report the spread over splits, not one split. A value under 0.5, like that 0.40, is
a probe fitting noise that anti-generalizes once the signal is gone, not an
inverted concept. It is also easy to fit on pooled token positions and then report
a last-token result the eraser never touched. Say which distribution you fitted,
and apply the eraser where you fitted it.

**Chunked collection.** For a dataset larger than one batch, loop over chunks with
the erasers applied inside *every* chunk's trace and concatenate the results —
`collect` with `prompts=texts[i:i + chunk]` — and fit each sequential site on the
concatenation of erased chunks.

## Behavioral cost: what the erasure breaks, and what else

Erasure needs a behavioral measure, and the sentiment probe above reads the
adjective vocabulary — nothing the model computes. Subject-verb number is visible
at the logits: in "The keys near the cabinet", ` are` − ` is` separates plural
from singular subjects, and the last token (the attractor) does not give the
subject away. Same machinery — sequential LEACE at sites 8–11, on the last
position it is fitted on — with the task metric on held-out subject nouns:

```python
nouns = [(w, w + "s") for w in ("key dog car book girl boy bird house table player doctor "
                                 "road window letter song tree farmer cat student teacher "
                                 "lake king ship wall").split()]
preps = [" near", " behind", " beside", " with", " for"]
ARE, IS = model.tokenizer.encode(" are")[0], model.tokenizer.encode(" is")[0]
gen = torch.Generator().manual_seed(0)
draw = lambda n: torch.randint(n, (1,), generator=gen).item()

agree_texts, agree_labels, lemma = [], [], []
for s in range(len(nouns)):
    for _ in range(8):
        a, p, an = draw(len(nouns)), preps[draw(len(preps))], draw(2)
        for number in (0, 1):
            agree_texts.append(f"The {nouns[s][number]}{p} the {nouns[a][an]}")
            agree_labels.append(float(number))
            lemma.append(s)
agree_labels, lemma = torch.tensor(agree_labels), torch.tensor(lemma)
held = torch.randperm(len(nouns), generator=gen)[:8]
a_test = torch.isin(lemma, held)
a_train_idx, a_test_idx = torch.where(~a_test)[0], torch.where(a_test)[0]

A_SITES = [8, 9, 10, 11]
a_train_texts = [agree_texts[i] for i in a_train_idx.tolist()]
agree_seq = {}
for site in A_SITES:
    seen = collect(to_device(agree_seq), a_train_texts)
    agree_seq[site] = fit_leace(seen[site], agree_labels[a_train_idx])
agree_leace = to_device(agree_seq)
base_train = collect(None, a_train_texts)
agree_random = {}
for site in A_SITES:
    v = torch.randn(hidden, generator=gen)
    v = v / v.norm()
    agree_random[site] = (torch.eye(hidden) - torch.outer(v, v), base_train[site].mean(0))
agree_random = to_device(agree_random)

def task_gap(erasers, prompts):
    """` are` - ` is` at the last position, erasers applied at the last position."""
    with torch.no_grad():
        with model.trace(prompts):
            for site, block in enumerate(model.transformer.h):
                if site in erasers:
                    M, mu = erasers[site]
                    block.output[:, -1, :] = mu + (block.output[:, -1, :] - mu) @ M
            gap = (model.output.logits[:, -1, ARE] - model.output.logits[:, -1, IS]).cpu().save()
    return gap

held_texts, held_y = [agree_texts[i] for i in a_test_idx.tolist()], agree_labels[a_test_idx]
task = {}
for name, erasers in (("no erasure", {}), ("seq. LEACE", agree_leace), ("random rank-1", agree_random)):
    gap = task_gap(erasers, held_texts)
    separation = float(gap[held_y == 1].mean() - gap[held_y == 0].mean())
    accuracy = float(((gap > 0).float() == held_y).float().mean())
    task[name] = separation
    print(f"{name:14s} plural-minus-singular gap {separation:+.2f}   agreement accuracy {accuracy:.3f}")

assert task["seq. LEACE"] < 0.25 * task["no erasure"] < task["random rank-1"]
```

Then the cost on text that has nothing to do with number. **Position policy:**
the eraser is fitted on last-position activations, so the concept test above
applies it there and nowhere else. A deployed edit runs at every position, which is
off the fitted distribution everywhere but the last — so measure its cost under
exactly that policy, but always skip **position 0** (GPT-2's attention-sink state
has a norm near 3,000, against ~100 elsewhere; no fitted mean describes it) and
**pads** (one trace per text below, so there are none — invokes in one trace
are padded to a common length). Perplexity, top-1
agreement and KL against the unerased run, next to the random eraser of equal rank:

```python
neutral = [
    "The history of the city goes back more than two thousand years, when traders first settled along the river.",
    "Photosynthesis is the process by which green plants use sunlight to make food from carbon dioxide and water.",
    "She opened the window, looked at the rain, and decided to stay inside with a cup of tea and a good novel.",
    "The committee will meet next Tuesday to discuss the budget for the coming fiscal year and other matters.",
    "Running a marathon requires months of training, careful nutrition, and plenty of rest between long runs.",
    "After the storm passed, the villagers gathered in the square to assess the damage to their homes and fields.",
]

def neutral_logprobs(erasers):
    out = []
    for text in neutral:                    # one trace per text: invokes would share padding
        with torch.no_grad():
            with model.trace(text):
                for site, block in enumerate(model.transformer.h):
                    if site in erasers:
                        M, mu = erasers[site]                      # every position but 0
                        block.output[:, 1:, :] = mu + (block.output[:, 1:, :] - mu) @ M
                logprobs = model.output.logits[0, :-1].log_softmax(-1).cpu().save()
        out.append(logprobs)
    return out

targets = [model.tokenizer(t, return_tensors="pt").input_ids[0, 1:] for t in neutral]
reference = neutral_logprobs({})
cost = {}
for name, erasers in (("no erasure", {}), ("seq. LEACE", agree_leace), ("random rank-1", agree_random)):
    logprobs = neutral_logprobs(erasers)
    nll = torch.cat([-lp.gather(-1, t[:, None]).squeeze(-1) for lp, t in zip(logprobs, targets)])
    top1 = torch.cat([lp.argmax(-1) == ref.argmax(-1) for lp, ref in zip(logprobs, reference)])
    kl = torch.cat([(ref.exp() * (ref - lp)).sum(-1) for lp, ref in zip(logprobs, reference)])
    cost[name] = float(kl.mean())
    print(f"{name:14s} perplexity {nll.mean().exp():6.2f}   top-1 agreement "
          f"{top1.float().mean():.3f}   KL {kl.mean():.3f}")

assert cost["seq. LEACE"] > cost["random rank-1"]
```

```
no erasure     plural-minus-singular gap +6.57   agreement accuracy 1.000
seq. LEACE     plural-minus-singular gap +0.62   agreement accuracy 0.586
random rank-1  plural-minus-singular gap +6.55   agreement accuracy 1.000
no erasure     perplexity  18.80   top-1 agreement 1.000   KL 0.000
seq. LEACE     perplexity  20.40   top-1 agreement 0.908   KL 0.077
random rank-1  perplexity  19.03   top-1 agreement 0.933   KL 0.005
```

Erasure takes the task's class separation from +6.57 to +0.62 and agreement to
near chance; the random eraser of equal rank leaves both untouched. The cost on
neutral text is real — KL 0.077 against 0.005 for random — because number is
used all through ordinary text: erasing it is not free, and the random control is
what shows that the cost comes from *this* direction. Applying the same erasers
at position 0 as well raises that cost to perplexity 24.07, KL 0.204 (random:
21.49, 0.055) — most of the damage of a careless deployment is the off-distribution
position, not the concept.

To make an erasure persistent, apply it under `model.edit(inplace=True)` with the
same position policy (the `model-editing-and-lora` skill).
