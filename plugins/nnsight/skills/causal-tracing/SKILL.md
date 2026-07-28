---
name: causal-tracing
description: Localize where a model stores and retrieves a fact using the ROME protocol — corrupt the subject tokens with noise, restore individual hidden states from the clean run, and measure how much of the correct answer's probability comes back. Use for factual-recall localization, for producing the classic two-site causal trace (early subject site, late last-token site), for windowed and component-severed traces, and when there is no natural corrupt prompt to pair against. Distinct from activation-patching, which transplants between two real prompts.
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

The output is a (layer × position) map of recovered probability. Compared to the
`activation-patching` skill: patching transplants between two real prompts and is
the right tool when you have a natural minimal pair; causal tracing corrupts with
noise and works on any single prompt.

<!-- test: setup -->
```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)

prompt = "The Eiffel Tower is in the city of"
answer = model.tokenizer.encode(" Paris")[0]
token_ids = model.tokenizer(prompt).input_ids
tokens = [model.tokenizer.decode([i]) for i in token_ids]
print(tokens)
# ['The', ' E', 'iff', 'el', ' Tower', ' is', ' in', ' the', ' city', ' of']

SUBJECT = slice(1, 5)          # ' E','iff','el',' Tower' — the whole subject span
n_layers = len(model.transformer.h)
```

Getting `SUBJECT` right matters: it must cover every token of the subject,
including the pieces a BPE tokenizer splits it into. Print the tokens and index
them; never assume word boundaries.

## Steps 1 and 2: clean states and calibrated noise

Noise is scaled to the embedding distribution — ROME uses 3σ of the embedding
matrix. Generate it **once**, outside the traces, so every restoration run is
corrupted identically:

```python
sigma = model.transformer.wte.weight.std().item()
generator = torch.Generator().manual_seed(0)
noise = (torch.randn(1, SUBJECT.stop - SUBJECT.start, model.config.n_embd,
                     generator=generator) * 3 * sigma).to(model.device)

with model.trace(prompt):
    clean_states = nnsight.save([block.output.detach() for block in model.transformer.h])
    clean_prob = model.output.logits[0, -1].softmax(-1)[answer].detach().save()

with model.trace(prompt):
    model.transformer.wte.output[:, SUBJECT, :] += noise
    corrupt_prob = model.output.logits[0, -1].softmax(-1)[answer].detach().save()

print(f"clean P(answer)     = {float(clean_prob):.4f}")
print(f"corrupted P(answer) = {float(corrupt_prob):.4f}")
```

```
clean P(answer)     = 0.0700
corrupted P(answer) = 0.0008
```

Those two numbers are the scale for everything that follows: restoration is
reported as the fraction of the gap between them that comes back. If corruption
does not collapse the answer, raise the noise or check that `SUBJECT` really
covers the subject — with no gap to recover, the trace is meaningless.

## Step 3: single-state restoration

One clean state at a time, every (layer, position) cell — batched so each layer
costs one forward pass:

```python
grid = []
for layer in range(n_layers):
    with model.trace() as tracer:
        row = nnsight.save([])
        for pos in range(len(token_ids)):
            with tracer.invoke(prompt):
                model.transformer.wte.output[:, SUBJECT, :] += noise
                model.transformer.h[layer].output[:, pos, :] = clean_states[layer][:, pos, :]
                row.append(model.output.logits[0, -1].softmax(-1)[answer].detach())
    grid.append([float(x) for x in row])

print(f"{'token':>10} " + "".join(f"L{l:<5}" for l in range(0, n_layers, 2)))
for pos, token in enumerate(tokens):
    print(f"{token!r:>10} " + "".join(f"{grid[l][pos]:.3f} " for l in range(0, n_layers, 2)))
```

```
     token L0    L2    L4    L6    L8    L10
     'The' 0.001 0.001 0.001 0.001 0.001 0.001
      ' E' 0.002 0.002 0.002 0.001 0.001 0.001
   ' Tower'0.001 0.001 0.001 0.002 0.001 0.001
     ' of' 0.001 0.002 0.003 0.004 0.010 0.044
```

Only the **last position at late layers** recovers anything. That is the honest
result of single-state restoration on a small model — a single hidden state is
rarely enough, because the corrupted signal keeps flowing through every other
path. This is why ROME restores a *window*.

## Windowed restoration — where the two sites appear

Restore a run of consecutive layers (ROME uses ~10 on GPT-2 XL; 5 works on small)
at a fixed position group:

```python
WINDOW = 5

def trace_sites(positions):
    with model.trace() as tracer:
        out = nnsight.save([])
        for layer in range(n_layers):
            with tracer.invoke(prompt):
                model.transformer.wte.output[:, SUBJECT, :] += noise
                for w in range(layer, min(layer + WINDOW, n_layers)):
                    model.transformer.h[w].output[:, positions, :] = clean_states[w][:, positions, :]
                out.append(model.output.logits[0, -1].softmax(-1)[answer].detach())
    return [float(x) for x in out]

subject_site = trace_sites(SUBJECT)
last_site = trace_sites(slice(-1, None))

print("window start:  " + " ".join(f"{l:5d}" for l in range(n_layers)))
print("subject pos:   " + " ".join(f"{v:.3f}" for v in subject_site))
print("last pos:      " + " ".join(f"{v:.3f}" for v in last_site))
```

```
window start:      0     1     2     3     4     5     6     7     8     9    10    11
subject pos:   0.032 0.027 0.033 0.036 0.032 0.027 0.015 0.008 0.005 0.002 0.001 0.001
last pos:      0.003 0.003 0.004 0.005 0.009 0.028 0.041 0.070 0.070 0.070 0.070 0.070
```

This is the classic **two-site** structure, reproduced on GPT-2 small:

- an **early site** at the *subject* tokens in layers 0–5, recovering ~46% of the
  clean probability — where the fact is looked up
- a **late site** at the *last* token from layer 7 on, recovering 100% — where the
  retrieved fact has arrived and determines the output

The early site is the interesting one: it is the claim that a specific range of
layers at the subject position performs factual recall, and it is what ROME
targets when editing a fact.

## Component-severed traces

To ask which sublayer does the work, restore only that sublayer's output:

```python
with model.trace(prompt):
    clean_mlp = nnsight.save([block.mlp.output.detach() for block in model.transformer.h])

with model.trace() as tracer:
    mlp_only = nnsight.save([])
    for layer in range(n_layers):
        with tracer.invoke(prompt):
            model.transformer.wte.output[:, SUBJECT, :] += noise
            for w in range(layer, min(layer + WINDOW, n_layers)):
                model.transformer.h[w].mlp.output[:, SUBJECT, :] = clean_mlp[w][:, SUBJECT, :]
            mlp_only.append(model.output.logits[0, -1].softmax(-1)[answer].detach())

print("mlp-only restore:", " ".join(f"{float(v):.3f}" for v in mlp_only))
```

```
mlp-only restore: 0.006 0.000 0.000 0.001 0.001 0.001 ...
```

Near zero — and that is informative rather than a failure. Restoring a sublayer's
*output* does not repair the residual stream it was added to, so the corrupted
signal still dominates. The meaningful severed design is the reverse: restore the
residual window and **sever** one component (freeze attention, or zero the MLP's
contribution) to see how much of the recovery depends on it. Interpret any
sublayer trace by asking which paths are still carrying corruption.

## Reading a causal trace

**Report recovery, not raw probability.** `(restored − corrupt) / (clean − corrupt)`
is comparable across prompts; raw probability is not.

**Average over noise seeds.** One noise draw is one sample of a random
perturbation. ROME averages ~10. If a peak moves when you change the seed, it
isn't a peak.

**Model size changes the picture.** The early site is sharper in larger models.
Weak or absent structure on a small model is not evidence that the fact is
distributed.

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
