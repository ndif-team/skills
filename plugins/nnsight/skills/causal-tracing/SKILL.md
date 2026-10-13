---
name: causal-tracing
description: Localize where a model stores and retrieves a fact using the ROME protocol — corrupt the subject tokens with noise, restore individual hidden states from the clean run, and measure how much of the correct answer's probability comes back. Use for factual-recall localization, for producing a (layer × position) causal trace with an early subject site and a late last-token site, for severed traces that ask whether MLPs or attention carry the recovery, and when there is no natural corrupt prompt to pair against. Distinct from activation-patching, which transplants between two real prompts.
---

# Causal Tracing

Causal tracing (Meng et al., ROME) localizes a fact without needing a second
prompt. The protocol is three runs:

1. **Clean** — run the prompt, record every hidden state and the answer's
   probability.
2. **Corrupted** — add noise to the *subject token embeddings*. The answer
   collapses.
3. **Restored** — re-run corrupted, but paste one clean hidden state back in.
   Whatever restores the answer was carrying the fact.

The output is a (layer × position) map of recovered probability. Use the
`activation-patching` skill instead when you have a natural minimal pair; causal
tracing works on any single prompt.

**The examples below run Qwen3-8B. They need a GPU and `--run-slow`, and are not
executed in default CI.**

## Pick a model that can show a trace

The protocol needs a model that holds the fact **confidently** and has the
**depth** to place the two sites apart — not a particular parameter count.

GPT-2 small on this skill's prompt produces no subject site: its subject row,
averaged over the same ten seeds, tops out at 3.4% recovery

```
 L0     L1     L2     L3     L4     L5     L6     L7     L8     L9     L10    L11
0.007  0.009  0.017  0.024  0.027  0.028  0.034  0.033  0.030  0.008  0.004  -0.000
```

against 116% at the same site on Qwen3-8B, and its last-token column reaches
1.000 only at layer 11, which is the architectural identity, not a site.
Llama-3.2-1B — fewer parameters than GPT-2 medium — separates cleanly: clean
P(` Paris`) 0.9495, the `' Tower'` subject row peaking at 0.768 at layer 3, and the
final `' of'` position reaching 0.922 at layer 9 of 16, well before the last block.

Two checks before running the grid:

- read the clean P(answer). Below ~0.5 the denominator of
  `(restored − corrupt) / (clean − corrupt)` is small and the whole grid is noise.
  The `logit-lens` skill's wiring check is the cheap way to ask.
- the last-token site has to land **before** the final block. A site only at the
  last layer is the architectural identity.

If either fails, change the prompt or the model. An empty trace is a fact about
the measurement, not evidence that the fact is distributed.

<!-- test: setup gpu slow -->
```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("Qwen/Qwen3-8B", device="cuda",
                          dtype=torch.bfloat16, dispatch=True)

prompt = "The Eiffel Tower is in the city of"
answer = model.tokenizer.encode(" Paris")[0]
token_ids = model.tokenizer(prompt).input_ids
tokens = [model.tokenizer.decode([i]) for i in token_ids]
print(tokens)
# ['The', ' E', 'iff', 'el', ' Tower', ' is', ' in', ' the', ' city', ' of']

SUBJECT = slice(1, 5)          # ' E','iff','el',' Tower' — the whole subject span
n_layers = len(model.model.layers)
n_pos = len(token_ids)
assert tokens[SUBJECT] == [' E', 'iff', 'el', ' Tower']
```

`SUBJECT` must cover every token of the subject, including the pieces a BPE
tokenizer splits it into. Print the tokens and index them; never assume word
boundaries.

## Step 1: the clean run

Save every block's residual output and the answer's probability — the two halves
of the scale everything else is reported against:

<!-- test: gpu slow -->
```python
with model.trace(prompt):
    clean_states = nnsight.save([block.output.detach() for block in model.model.layers])
    clean_prob = model.output.logits[0, -1].softmax(-1)[answer].detach().save()

print(f"clean P(' Paris') = {float(clean_prob):.4f}")
assert float(clean_prob) > 0.5
```

```
clean P(' Paris') = 0.8008
```

## Step 2: calibrate the noise, then draw it

ROME scales the noise to the embedding distribution. The published figure is 3σ,
but σ there is the spread of the embeddings a subject actually uses, not the
standard deviation of the whole embedding matrix — and the two differ enough per
architecture that **3σ of `embed_tokens.weight` does not corrupt Qwen3-8B**.
Sweep the scale before trusting any recovery number:

<!-- test: gpu slow -->
```python
sigma = model.model.embed_tokens.weight.std().item()

def noise(seed, scale):
    generator = torch.Generator().manual_seed(seed)
    return (torch.randn(1, SUBJECT.stop - SUBJECT.start, model.config.hidden_size,
                        generator=generator) * scale * sigma).to(model.device).to(model.dtype)

for scale in (1, 3, 5):
    with model.trace() as tracer:
        probs = nnsight.save([])
        for seed in range(10):
            with tracer.invoke(prompt):
                model.model.embed_tokens.output[:, SUBJECT, :] += noise(seed, scale)
                probs.append(model.output.logits[0, -1].softmax(-1)[answer].detach())
    values = [float(p) for p in probs]
    print(f"{scale}σ  worst seed {max(values):.4f}   mean {sum(values)/len(values):.4f}")
```

```
1σ  worst seed 0.9297   mean 0.8699
3σ  worst seed 0.9258   mean 0.2355
5σ  worst seed 0.0012   mean 0.0005
```

Judge the scale by its **worst** seed, not its mean. At 3σ the mean says the answer
collapsed and one seed in ten left it at `0.93`; a seed that barely corrupts has a
recovery denominator near zero and dominates any average built on top of it. 5σ
collapses every seed.

Generate the noise **once**, outside the traces, so every restoration run is
corrupted identically:

<!-- test: gpu slow -->
```python
SCALE = 5
SEEDS = range(10)
noises = [noise(seed, SCALE) for seed in SEEDS]

with model.trace() as tracer:
    probs = nnsight.save([])
    for seed in SEEDS:
        with tracer.invoke(prompt):
            model.model.embed_tokens.output[:, SUBJECT, :] += noises[seed]
            probs.append(model.output.logits[0, -1].softmax(-1)[answer].detach())
corrupt_probs = [float(p) for p in probs]

assert max(corrupt_probs) < 0.01, "corruption did not collapse the answer on every seed"
print("corrupt P per seed:", " ".join(f"{p:.4f}" for p in corrupt_probs))
```

```
corrupt P per seed: 0.0004 0.0005 0.0011 0.0012 0.0002 0.0001 0.0007 0.0001 0.0008 0.0001
```

## Step 3: single-state restoration

One clean state at a time, every (layer, position) cell, averaged over the ten
noise draws. Every (seed, position) pair is an invoke, so each layer is one
forward pass:

<!-- test: gpu slow -->
```python
def recovery(restored, seed):
    return (restored - corrupt_probs[seed]) / (float(clean_prob) - corrupt_probs[seed])

per_seed = torch.zeros(len(SEEDS), n_layers, n_pos)
for layer in range(n_layers):
    with model.trace() as tracer:
        restored = nnsight.save([])
        for seed in SEEDS:
            for pos in range(n_pos):
                with tracer.invoke(prompt):
                    model.model.embed_tokens.output[:, SUBJECT, :] += noises[seed]
                    model.model.layers[layer].output[:, pos, :] = clean_states[layer][:, pos, :]
                    restored.append(model.output.logits[0, -1].softmax(-1)[answer].detach())
    for i, p in enumerate(restored):
        seed, pos = divmod(i, n_pos)
        per_seed[seed, layer, pos] = recovery(float(p), seed)
grid = per_seed.mean(0)

print(f"{'token':>10} " + "".join(f"L{l:<6}" for l in range(0, n_layers, 3)))
for pos, token in enumerate(tokens):
    print(f"{token!r:>10} " + "".join(f"{grid[l, pos]:.3f}  " for l in range(0, n_layers, 3)))

assert grid[:15, SUBJECT].max() > 0.9 and grid[24:, SUBJECT].max() < 0.1
assert grid[24:, -1].max() > 0.9 and grid[:15, -1].max() < 0.1
```

```
     token L0     L3     L6     L9     L12    L15    L18    L21    L24    L27    L30    L33
     'The' 0.000  -0.000  0.000  0.000  -0.000  0.000  -0.000  -0.000  0.000  0.000  0.000  -0.000
      ' E' 0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  -0.000  -0.000  0.000
     'iff' 0.451  0.383  0.260  0.297  0.083  0.026  0.028  0.011  0.001  0.001  0.000  0.000
      'el' 0.237  0.766  0.813  0.628  0.341  0.182  0.215  0.128  0.027  0.006  0.004  0.000
  ' Tower' 0.498  0.722  1.104  1.093  1.159  0.985  0.769  0.205  0.010  0.002  0.002  0.000
     ' is' 0.000  0.000  0.000  0.000  0.002  0.003  0.002  0.001  0.001  0.000  0.000  0.000
     ' in' 0.000  -0.000  -0.000  0.000  0.001  0.002  0.003  0.001  0.020  0.009  0.005  0.001
    ' the' -0.000  0.000  -0.000  0.000  -0.000  -0.000  0.000  0.000  0.001  0.000  0.000  0.000
   ' city' -0.000  0.000  0.000  -0.000  -0.000  0.000  0.000  0.000  0.001  0.000  0.000  0.000
     ' of' -0.000  0.000  -0.000  -0.000  -0.000  -0.000  0.000  0.002  0.728  0.980  1.016  0.975
```

Two sites, from single states, with no window needed:

- an **early site** at the last subject token `' Tower'`, full recovery from layer 6
  to layer 15, gone by layer 24 — where the fact is looked up
- a **late site** at the final token `' of'`, nothing until layer 21 and then full
  recovery from layer 24 on — where the retrieved fact determines the output

The tokens between them stay at zero throughout. Recovery slightly above 1.0 is
ordinary: a restored state can leave the model more confident than the clean run.

## Severed traces: which sublayer carries the recovery

Restoring a window of one sublayer's outputs asks whether that component is what
repairs the residual stream. A sublayer's output at layer *w* is added to the
stream rather than replacing it, so a run of them compounds. One helper serves
both sublayers; `site` picks the output to restore:

<!-- test: gpu slow -->
```python
WINDOW = 10

def severed_trace(site):
    """Recovery from restoring `site(block)` over a window of layers, per window start."""
    with model.trace(prompt):
        clean_site = nnsight.save([site(block).detach() for block in model.model.layers])

    out = torch.zeros(n_layers, n_pos)
    for start in range(n_layers):
        window = range(start, min(start + WINDOW, n_layers))
        with model.trace() as tracer:
            restored = nnsight.save([])
            for seed in SEEDS:
                for pos in range(n_pos):
                    with tracer.invoke(prompt):
                        model.model.embed_tokens.output[:, SUBJECT, :] += noises[seed]
                        for w in window:
                            site(model.model.layers[w])[:, pos, :] = clean_site[w][:, pos, :]
                        restored.append(model.output.logits[0, -1].softmax(-1)[answer].detach())
        for i, p in enumerate(restored):
            seed, pos = divmod(i, n_pos)
            out[start, pos] += recovery(float(p), seed) / len(SEEDS)
    return out

mlp_grid = severed_trace(lambda block: block.mlp.output)
attn_grid = severed_trace(lambda block: block.self_attn.output[0])

for name, g in (("MLP", mlp_grid), ("attention", attn_grid)):
    print(f"{name:>10} " + "".join(f"W{l:<6}" for l in range(0, n_layers, 3)))
    for pos in (2, 3, 4, n_pos - 1):
        print(f"{tokens[pos]!r:>10} " + "".join(f"{g[l, pos]:.3f}  " for l in range(0, n_layers, 3)))

assert mlp_grid[:10, SUBJECT].max() > 0.9 and mlp_grid[12:, SUBJECT].max() < 0.3
assert attn_grid[15:24, -1].max() > 0.8 and attn_grid[:12, -1].max() < 0.2
```

```
       MLP W0     W3     W6     W9     W12    W15    W18    W21    W24    W27    W30    W33
     'iff' 0.561  0.288  0.507  0.263  0.020  0.004  0.002  0.002  0.000  0.000  -0.000  -0.000
      'el' 0.755  0.887  0.711  0.393  0.015  0.037  0.025  0.008  0.000  0.000  0.000  -0.000
  ' Tower' 1.065  1.066  1.118  0.983  0.120  0.124  0.039  0.010  0.001  0.000  0.000  0.000
     ' of' -0.000  -0.000  -0.000  -0.000  -0.000  0.003  0.101  0.424  0.146  0.005  -0.001  -0.001
 attention W0     W3     W6     W9     W12    W15    W18    W21    W24    W27    W30    W33
     'iff' 0.141  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  -0.000  -0.000  0.000
      'el' 0.780  0.001  -0.000  0.000  -0.000  -0.000  -0.000  0.000  0.000  0.000  0.000  -0.000
  ' Tower' 0.187  0.001  -0.000  0.000  0.000  -0.000  -0.000  -0.000  0.000  0.000  -0.000  0.000
     ' of' -0.000  -0.000  -0.000  0.000  0.001  0.754  0.914  0.935  0.521  0.192  0.061  0.003
```

The two grids split the work cleanly. MLPs at the subject over layers 0–13 account
for the early site and contribute almost nothing at the final token; attention at
the final token over layers 15–24 accounts for the late site and contributes almost
nothing at the subject. That is the division ROME reports, and it is the reason
ROME edits MLP weights at the subject rather than anywhere else.

**Window sublayers, not the residual.** At a single position, restoring the
residual over a window of layers is exactly a single-layer restore at the window's
last layer — each write overwrites the previous one, and on Qwen3-8B the two agree
to every printed digit. Windowing the residual only buys anything across several
positions that feed each other; for one position, window the sublayer outputs as
above.

## Reading a causal trace

**Report recovery, not raw probability.** `(restored − corrupt) / (clean − corrupt)`
is comparable across prompts; raw probability is not.

**Average over noise seeds, and check what moves when you change them.** One draw
is one sample of a random perturbation, and `per_seed` from step 3 says which parts
of the picture survive redrawing it:

<!-- test: gpu slow -->
```python
subject_curve = per_seed[:, :, SUBJECT].mean(-1)          # [seed, layer]
last_curve = per_seed[:, :, -1]

print("early-site peak layer per seed:", [int(c.argmax()) for c in subject_curve])
print("late-site  peak layer per seed:", [int(c.argmax()) for c in last_curve])
print("early-site peak recovery:      ",
      " ".join(f"{float(c.max()):.2f}" for c in subject_curve))

# every seed puts the two sites on the correct sides of the model
assert all(int(c.argmax()) < n_layers // 2 for c in subject_curve)
assert all(int(c.argmax()) > n_layers // 2 for c in last_curve)
# but the exact early peak is not resolved by a single seed
assert len({int(c.argmax()) for c in subject_curve}) > 1
```

```
early-site peak layer per seed: [6, 6, 8, 10, 1, 3, 6, 6, 1, 1]
late-site  peak layer per seed: [31, 31, 31, 31, 31, 35, 30, 31, 31, 35]
early-site peak recovery:       0.47 0.70 0.59 0.78 0.70 0.47 0.62 0.52 0.53 0.52
```

Every seed agrees on where the sites are and none of them agrees on the early
peak, which ranges over layers 1–10 and over recoveries of 0.47 to 0.78. Report the
site; do not report a layer index from a single seed as if the protocol resolved
one. The exact indices shift a little between runs — bfloat16 arithmetic is not
reproducible across devices or batch shapes — which is the same point from the
other side.

**Two corners of the grid are architecture, not evidence.** Restoring the final
position at the last block reproduces whichever run the donor came from, and
restoring any non-final position at the last block cannot reach the last position's
logits at all. Both hold for any donor, so neither is a check that the sweep is
wired correctly. The control that can fail is restoring from the *corrupted* run's
own states, which must leave the metric where it was.

**Corruption must be at the embeddings.** Corrupting later layers conflates "the
subject was unreadable" with "the computation was disturbed".

**A trace is a localization, not a mechanism.** It says restoring these states
suffices. To claim a component *computes* the fact, follow with editing (does
changing those weights change the fact?) — see the `model-editing-and-lora`
skill — or with the `activation-patching` skill on a minimal pair.

## Related skills

- `activation-patching` — paired-prompt transplants, head- and MLP-level patching
- `attribution-patching` — a cheap first pass when the grid is too large
- `model-editing-and-lora` — acting on what a trace localizes
- `nnsight` — batching sweeps into single forward passes
