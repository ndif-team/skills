# Causal tracing

Causal tracing (Meng et al., ROME) localizes a fact from one prompt. Add noise to
the subject's token embeddings until the answer collapses, then run the corrupted
prompt again with one clean hidden state restored. A state that brings the
answer back was carrying the fact. With a natural minimal pair, use
[activation-patching](activation-patching.md) instead.

The whole guide runs on a CPU with a 135M-parameter model in float32:

```python
import torch
import nnsight
from nnsight import TransformersModel

torch.set_grad_enabled(False)                       # nothing in this protocol needs a gradient

model = TransformersModel("HuggingFaceTB/SmolLM2-135M-Instruct", task="text-generation",
                          dtype=torch.float32, dispatch=True)
embed = model.model.embed_tokens
n_layers = len(model.model.layers)

prompt = "The Eiffel Tower is in the city of"
answer = model.tokenizer.encode(" Paris", add_special_tokens=False)[0]
tokens = [model.tokenizer.decode([i]) for i in model.tokenizer(prompt).input_ids]
n_pos = len(tokens)
print(tokens)

SUBJECT = slice(1, 4)                               # every piece of the subject
assert tokens[SUBJECT] == [" E", "iffel", " Tower"]

with model.trace(prompt):
    clean_states = nnsight.save([block.output for block in model.model.layers])
    clean_prob = model.output.logits[0, -1].softmax(-1)[answer].save()

print(f"clean P(' Paris') = {clean_prob.item():.4f}")
assert clean_prob > 0.5
```

```
['The', ' E', 'iffel', ' Tower', ' is', ' in', ' the', ' city', ' of']
clean P(' Paris') = 0.9553
```

- **Print the tokens and index the subject from them.** `SUBJECT` covers every
  piece the tokenizer splits the subject into. A tokenizer that prepends BOS
  (Llama-3, Gemma) shifts every index by one.
- **The model must hold the fact confidently.** Recovery divides by clean minus
  corrupt. GPT-2 small gives `' Paris'` 0.07 on this prompt, and no subject cell
  of its grid recovers more than 0.05. If the clean probability is low, change
  the prompt or the model.

## Calibrate the noise by its worst seed

The noise is Gaussian, scaled in units of the embedding matrix's standard
deviation. Sweep the scale before trusting any recovery number:

```python
sigma = embed.weight.std().item()
SEEDS = range(10)

def noise(seed, scale):
    generator = torch.Generator().manual_seed(seed)
    draw = torch.randn(1, SUBJECT.stop - SUBJECT.start, model.config.hidden_size, generator=generator)
    return (draw * scale * sigma).to(model.device, model.dtype)

def corrupted(noises):
    """P(answer) under each noise draw: one invoke per draw, one forward pass."""
    with model.trace() as tracer:
        probs = nnsight.save({})
        for seed, draw in enumerate(noises):
            with tracer.invoke(prompt):
                embed.output[:, SUBJECT, :] += draw
                probs[seed] = model.output.logits[0, -1].softmax(-1)[answer]
    return [probs[seed].item() for seed in range(len(noises))]

for scale in (0.5, 1, 3):
    values = corrupted([noise(seed, scale) for seed in SEEDS])
    print(f"{scale:>3} sigma   worst seed {max(values):.3f}   mean {sum(values) / len(values):.3f}")
```

```
0.5 sigma   worst seed 0.903   mean 0.493
  1 sigma   worst seed 0.154   mean 0.083
  3 sigma   worst seed 0.098   mean 0.045
```

**Judge a scale by its worst seed, not its mean.** At 0.5 sigma the mean says
half the answer is gone while one seed in ten still gives 0.90. That seed's
recovery denominator is near zero, and it dominates any average built on it.

**Do not carry a scale from one model to another.** 3 sigma takes every seed
below 0.10 here. `meta-llama/Llama-3.2-1B` collapses at 1 sigma (worst seed
0.009), and `Qwen/Qwen3-8B` needs 5: at 3 sigma one seed in ten keeps 0.93.

Draw the noise once, outside the traces, so every restoration is corrupted
identically:

```python
noises = [noise(seed, 3) for seed in SEEDS]
corrupt_probs = corrupted(noises)
assert max(corrupt_probs) < 0.15, "the corruption left the answer standing on some seed"

def recovery(restored, seed):
    """0 at the corrupted run, 1 at the clean run."""
    return (restored - corrupt_probs[seed]) / (clean_prob.item() - corrupt_probs[seed])
```

## Restore one state at a time

Every (seed, position) pair is an invoke, so each layer is one forward pass:

```python
per_seed = torch.zeros(len(SEEDS), n_layers, n_pos)
for layer in range(n_layers):
    with model.trace() as tracer:
        restored = nnsight.save({})
        for seed in SEEDS:
            for pos in range(n_pos):
                with tracer.invoke(prompt):
                    embed.output[:, SUBJECT, :] += noises[seed]
                    model.model.layers[layer].output[:, pos, :] = clean_states[layer][:, pos, :]
                    restored[seed, pos] = model.output.logits[0, -1].softmax(-1)[answer]
    for (seed, pos), prob in restored.items():
        per_seed[seed, layer, pos] = recovery(prob.item(), seed)
grid = per_seed.mean(0)                             # (layer, position)

print(f"{'':>10}" + "".join(f"L{layer:<6}" for layer in range(0, n_layers, 3)))
for pos, token in enumerate(tokens):
    print(f"{token!r:>9} " + "".join(f"{grid[layer, pos]:+.2f}  " for layer in range(0, n_layers, 3)))

early_pos = SUBJECT.start + int(grid[:, SUBJECT].max(0).values.argmax())
assert tokens[early_pos] == "iffel"
assert grid[1:21, early_pos].min() > 0.5 and grid[24:, SUBJECT].max() < 0.1   # subject site
assert grid[:20, -1].max() < 0.1 and grid[23:, -1].min() > 0.9                # last-token site
```

```
          L0     L3     L6     L9     L12    L15    L18    L21    L24    L27
    'The' +0.00  +0.00  +0.00  +0.00  +0.00  +0.00  +0.00  +0.00  +0.00  +0.00
     ' E' +0.02  +0.03  +0.02  +0.02  +0.02  +0.02  +0.02  +0.02  +0.00  -0.00
  'iffel' +0.51  +0.77  +0.80  +0.80  +0.71  +0.68  +0.63  +0.41  +0.02  +0.01
 ' Tower' -0.00  -0.03  -0.02  +0.02  +0.04  +0.05  +0.05  +0.01  +0.01  +0.01
    ' is' -0.00  -0.01  -0.00  -0.00  +0.01  +0.01  +0.01  +0.00  +0.00  +0.00
    ' in' -0.00  -0.00  +0.00  +0.01  +0.01  -0.00  +0.00  +0.00  +0.13  +0.10
   ' the' +0.00  -0.00  -0.00  -0.00  +0.00  +0.00  +0.00  +0.00  +0.01  +0.01
  ' city' +0.00  +0.00  -0.00  +0.00  +0.00  +0.00  +0.01  +0.01  +0.00  +0.00
    ' of' +0.00  +0.00  -0.00  -0.00  +0.01  +0.01  +0.02  +0.22  +0.99  +0.99
```

Two sites: an early one at the subject token `'iffel'`, from the first block
through layer 20, and a late one at the last token, from layer 23 on. No other
cell reaches 0.14.

**Report a site, not a layer index.** `per_seed` shows what survives a redraw of
the noise:

```python
peaks = [int(curve.argmax()) for curve in per_seed[:, :, early_pos]]
onsets = [int((curve > 0.9).nonzero()[0]) for curve in per_seed[:, :, -1]]
print("early-site peak layer per seed:", peaks)
print("late site above 0.9 from layer:", onsets)
assert len(set(onsets)) == 1 and len(set(peaks)) > 1
```

```
early-site peak layer per seed: [9, 1, 1, 1, 6, 6, 0, 1, 6, 9]
late site above 0.9 from layer: [23, 23, 23, 23, 23, 23, 23, 23, 23, 23]
```

Every seed puts both sites in the same place. The early site is a plateau, so
the layer of its peak is noise: give the range of layers a site spans, and for
the late site the layer where it starts.

**The last block is architecture, not evidence.** Restoring the last position
there reproduces the clean run, and restoring any other position there cannot
reach the last position's logits, whatever the state. A late site has to start
before the final block: GPT-2 small's last-token row reaches 0.55 at layer 10 of
12 and 1.00 only at layer 11, which is that identity and not a site.

## The control: restore the run's own state

Neither corner of the grid can fail, so neither checks the wiring. Restoring the
corrupted run's own state can: it must leave the answer where it was.

```python
with model.trace(prompt):
    embed.output[:, SUBJECT, :] += noises[0]
    own_state = model.model.layers[6].output[:, early_pos, :].clone().save()

with model.trace(prompt):
    embed.output[:, SUBJECT, :] += noises[0]
    model.model.layers[6].output[:, early_pos, :] = own_state
    self_restored = model.output.logits[0, -1].softmax(-1)[answer].save()

assert abs(recovery(self_restored.item(), 0)) < 1e-3
```

If it moves, the restore itself changes the run (a different noise draw, a write
at the wrong position), and every cell of the grid carries that error.

## Which sublayer: windowed restoration

Restore one sublayer's clean output over a window of layers, in place of the
residual at one layer. A sublayer's output is added to the stream, so a run of
them compounds:

```python
WINDOW = 6
STARTS = range(0, n_layers, 2)

def windowed(site, pos):
    """Mean recovery from restoring site(block) at `pos` over WINDOW layers, per window start."""
    with model.trace(prompt):
        clean_site = nnsight.save([site(block) for block in model.model.layers])
    curve = []
    for start in STARTS:
        with model.trace() as tracer:
            restored = nnsight.save({})
            for seed in SEEDS:
                with tracer.invoke(prompt):
                    embed.output[:, SUBJECT, :] += noises[seed]
                    for layer in range(start, min(start + WINDOW, n_layers)):
                        site(model.model.layers[layer])[:, pos, :] = clean_site[layer][:, pos, :]
                    restored[seed] = model.output.logits[0, -1].softmax(-1)[answer]
        curve.append(sum(recovery(restored[seed].item(), seed) for seed in SEEDS) / len(SEEDS))
    return curve

sublayers = {"mlp": lambda block: block.mlp.output, "attn": lambda block: block.self_attn.output[0]}
curves = {(name, pos): windowed(site, pos)
          for name, site in sublayers.items() for pos in (early_pos, n_pos - 1)}

print(f"{'':<14}" + "".join(f"W{start:<5}" for start in STARTS))
for (name, pos), curve in curves.items():
    print(f"{name:<5}{tokens[pos]!r:<9}" + "".join(f"{value:+.2f} " for value in curve))

assert curves["mlp", early_pos][0] > 0.5 and max(curves["attn", early_pos]) < 0.1
assert max(curves["attn", n_pos - 1]) > 0.9 and max(curves["mlp", n_pos - 1]) < 0.3
```

```
              W0    W2    W4    W6    W8    W10   W12   W14   W16   W18   W20   W22   W24   W26   W28
mlp  'iffel'  +0.68 +0.25 +0.30 +0.18 +0.19 +0.08 +0.04 +0.01 +0.03 +0.01 +0.00 -0.00 -0.00 -0.00 -0.00
mlp  ' of'    +0.00 +0.01 -0.01 -0.01 -0.00 -0.01 -0.00 -0.00 +0.01 +0.06 +0.19 +0.10 -0.04 -0.05 -0.05
attn 'iffel'  +0.01 +0.01 +0.01 +0.01 +0.01 +0.00 +0.00 -0.00 -0.00 +0.01 +0.01 +0.01 +0.00 +0.00 +0.00
attn ' of'    -0.00 -0.00 -0.00 +0.01 +0.02 +0.02 +0.02 +0.04 +0.18 +0.98 +0.85 +0.51 -0.01 +0.04 +0.00
```

The sublayers split the work. MLP outputs restore the answer at the subject
token, most in the window that starts at layer 0, and attention outputs restore
it at the last token, in the windows that start at layers 18 to 22. Attention at
the subject and MLPs at the last token restore at most 0.19. That is the
division ROME reports: early MLPs at the subject, late attention into the last
token.

The converse question, which path a restored state needs, is a severed trace:
restore the residual at one cell and also hold every later `mlp.output` (or
`self_attn.output[0]`) at that position at its value from the corrupted run.

A trace says that restoring these states is enough. To claim a component computes
the fact, follow with an edit to its weights
([model-editing-and-lora](model-editing-and-lora.md)) or with activation patching
on a minimal pair.
