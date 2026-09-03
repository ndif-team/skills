---
name: logit-lens
description: Decode what a model predicts at every layer by applying its final norm and unembedding to intermediate residual streams. Use to see where in depth an answer emerges, to track a specific token's probability across layers, to sanity-check that a model knows a fact before running invasive interventions, or to compare prediction trajectories between prompts. Built on nnsight 0.8; includes per-architecture variants, the tuned-lens caveat, and the failure modes that make a logit lens look informative when it is not.
---

# Logit Lens

A transformer refines its prediction layer by layer in the residual stream. The
logit lens (nostalgebraist, 2020) reads that intermediate state through the
model's *own* final norm and unembedding:

```
prediction_at_layer_L = lm_head(ln_f(residual_L))
```

Typically early layers predict generic frequent tokens and the answer appears
somewhere in the second half. Where it appears — and whether it appears at all —
is the measurement.

<!-- test: setup -->
```python
import torch
import nnsight
from nnsight import TransformersModel

model = TransformersModel("openai-community/gpt2", dispatch=True)
prompt = "The Eiffel Tower is in the city of"
```

## Canonical implementation

```python
with model.trace(prompt):
    top_per_layer = nnsight.save([])
    for block in model.transformer.h:
        resid = block.output                                    # (batch, seq, hidden)
        logits = model.lm_head(model.transformer.ln_f(resid))    # applied out of order
        top_per_layer.append(logits[:, -1, :].argmax(dim=-1))

decoded = [model.tokenizer.decode(token[0]) for token in top_per_layer]
for layer, token in enumerate(decoded):
    print(f"layer {layer:2d}: {token!r}")

assert decoded[:6] == [" the"] * 6
assert decoded[10:] == [" Paris", " Paris"]
```

```
layer  0: ' the'      layer  6: ' East'
layer  1: ' the'      layer  7: ' Ing'
layer  2: ' the'      layer  8: ' Rome'
layer  3: ' the'      layer  9: ' London'
layer  4: ' the'      layer 10: ' Paris'
layer  5: ' the'      layer 11: ' Paris'
```

The whole sweep is **one forward pass**. Calling `model.lm_head(...)` inside the
trace runs its `forward` directly — no hooks, no ordering constraint — so you can
apply the unembedding to a layer-5 activation without re-running anything.

Do not write `block.output[0]`: a block returns a plain tensor on every family
tested, so that indexes batch row 0 — no error, and a lens that decodes the same
token at every layer. (See the `nnsight` skill.)

## Check the wiring before you read anything off it

Applied to the **last** block, the lens is the model's own final computation, so
it must reproduce the model's logits exactly. One line tells you whether the
norm, the head and the block output you picked are the right three:

```python
with model.trace(prompt):
    lens = model.lm_head(model.transformer.ln_f(model.transformer.h[-1].output)).save()
    real = model.output.logits.save()

assert torch.equal(lens, real), (lens - real).abs().max()
```

Run it on any model before plotting. A mismatch means one of three things, none
of which raises on its own:

- **the wrong norm**, or none — `lm_head(resid)` without the final norm decodes
  `' the'` at probability `1.0000` at *every* GPT-2 layer, which looks like a
  confident lens and reads exactly like the "does not transfer" symptom below;
- **the wrong head or block output** — `block.output[0]` on a tensor, or a head
  attribute the model does not have;
- **the model post-processes the logits.** `google/gemma-2-2b` softcaps them
  (`config.final_logit_softcapping = 30.0`), so the check fails by `51.0` and the
  uncapped distribution is far too sharp: max probability `0.9995` against the
  model's `0.9257`, entropy `0.0050` against `0.6010`.

Reapply the cap yourself and every layer is read on the model's own scale. Check
`model.config` rather than a list of model names — Gemma-3 sets
`final_logit_softcapping` to `None`, and GPT-2 has no such attribute at all:

```python
cap = getattr(model.config, "final_logit_softcapping", None)

with model.trace(prompt):
    logits = model.lm_head(model.transformer.ln_f(model.transformer.h[-1].output))
    if cap is not None:
        logits = torch.tanh(logits / cap) * cap
    capped = logits.save()

assert torch.equal(capped, real)      # cap is None here; on gemma-2 this is what makes it hold
```

## Top-k, and the probability trajectory of one token

```python
with model.trace(prompt):
    topk_per_layer = nnsight.save([])
    for block in model.transformer.h:
        logits = model.lm_head(model.transformer.ln_f(block.output))
        topk_per_layer.append(logits[0, -1].topk(5).indices)

for layer, ids in enumerate(topk_per_layer):
    print(f"layer {layer:2d}: {[model.tokenizer.decode(i) for i in ids]}")
```

Tracking one target is usually the more useful measurement — it gives a curve you
can compare across prompts or interventions:

```python
paris = model.tokenizer.encode(" Paris")[0]

with model.trace(prompt):
    trajectory = nnsight.save([])
    for block in model.transformer.h:
        logits = model.lm_head(model.transformer.ln_f(block.output))
        trajectory.append(logits[0, -1].softmax(dim=-1)[paris])

for layer, p in enumerate(trajectory):
    print(f"layer {layer:2d}  P(' Paris') = {p.item():.4f}")
```

## Every position, not just the last

The same decode applied across the sequence gives the classic heatmap — what the
model would predict *at each position* if it stopped at each layer:

```python
with model.trace(prompt):
    grid = nnsight.save([])
    for block in model.transformer.h:
        logits = model.lm_head(model.transformer.ln_f(block.output))
        probs = logits[0].softmax(dim=-1)
        best = probs.max(dim=-1)
        grid.append(torch.stack([best.values, best.indices.float()]).cpu())

stacked = torch.stack(grid)                      # (layers, 2, seq)
tokens = [model.tokenizer.decode([i]) for i in model.tokenizer(prompt).input_ids]
print(f"{'':12}" + "".join(f"L{i:<7}" for i in range(0, 12, 3)))
for pos, token in enumerate(tokens):
    row = "".join(f"{model.tokenizer.decode(int(stacked[i, 1, pos])):<8}" for i in range(0, 12, 3))
    print(f"{token!r:12}{row}")
```

## Comparing two prompts in one pass

Put each prompt in its own invoke — one forward pass, directly comparable curves:

```python
paris = model.tokenizer.encode(" Paris")[0]
rome = model.tokenizer.encode(" Rome")[0]

with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in the city of"):
        eiffel = nnsight.save([])
        for block in model.transformer.h:
            logits = model.lm_head(model.transformer.ln_f(block.output))
            eiffel.append(logits[0, -1].softmax(dim=-1)[paris])

    with tracer.invoke("The Colosseum is in the city of"):
        colosseum = nnsight.save([])
        for block in model.transformer.h:
            logits = model.lm_head(model.transformer.ln_f(block.output))
            colosseum.append(logits[0, -1].softmax(dim=-1)[rome])

for layer, (a, b) in enumerate(zip(eiffel, colosseum)):
    print(f"layer {layer:2d}  P(Paris|Eiffel)={a.item():.3f}   P(Rome|Colosseum)={b.item():.3f}")
```

## Other architectures

The recipe is always *final norm, then unembedding* — only the names change. A
block's `.output` is a plain tensor in every row of this table (nnsight 0.8,
`transformers` 5.15); none of them wants a `[0]`.

| Family | Residual | Final norm | Unembed |
|---|---|---|---|
| GPT-2 | `model.transformer.h[i].output` | `model.transformer.ln_f` | `model.lm_head` |
| Llama / Mistral / Qwen / SmolLM2 | `model.model.layers[i].output` | `model.model.norm` | `model.lm_head` |
| GPT-NeoX / Pythia | `model.gpt_neox.layers[i].output` | `model.gpt_neox.final_layer_norm` | `model.lm_head` |

```python
llama = TransformersModel("HuggingFaceTB/SmolLM2-135M-Instruct", dispatch=True)

with llama.trace("The capital of France is"):
    tops = nnsight.save([])
    for block in llama.model.layers:
        logits = llama.lm_head(llama.model.norm(block.output))
        tops.append(logits[0, -1].argmax(dim=-1))

decoded = [llama.tokenizer.decode(t) for t in tops[-6:]]
print(decoded)

assert decoded == [" the", " the", " the", " the", " the", " Paris"]
```

The answer arrives at the last layer and nowhere earlier — see
[Reading the result honestly](#reading-the-result-honestly). That is the common
case; GPT-2's smooth trajectory is the exception.

Confirm the row for your own checkpoint by running the wiring check on it. It
catches a wrong unembedding name in one line — Pythia's, for instance, is
`model.lm_head`, and `model.embed_out` raises `AttributeError` naming the module
rather than the model:

```python
pythia = TransformersModel("EleutherAI/pythia-70m-deduped", dispatch=True)

with pythia.trace("The capital of France is"):
    lens = pythia.lm_head(
        pythia.gpt_neox.final_layer_norm(pythia.gpt_neox.layers[-1].output)
    ).save()
    real = pythia.output.logits.save()

assert torch.equal(lens, real)
```

## During generation

Wrap the sweep in a bounded iteration loop to watch the trajectory at every
generated token:

```python
with model.generate(prompt, max_new_tokens=3) as tracer:
    per_step = nnsight.save([])
    for step in tracer.iter[:3]:
        logits = model.lm_head(model.transformer.ln_f(model.transformer.h[8].output))
        per_step.append(logits[0, -1].argmax(dim=-1))
    ids = tracer.result.save()

print("layer-8 guess per step:", [model.tokenizer.decode(t) for t in per_step])
print("actually generated:    ", model.tokenizer.decode(ids[0, -3:]))
```

## Reading the result honestly

**The logit lens is a projection, not the model's belief.** It assumes intermediate
residuals live in the same basis the unembedding expects. That holds well for
GPT-2 and poorly for several later models — a flat or nonsensical curve may mean
the lens does not transfer, not that the model knows nothing. Symptoms: every
layer decodes to the same high-frequency token, or probabilities stay near zero
until the final layer and then jump.

Before concluding that, run the wiring check. A lens missing its final norm
produces the same symptom — one high-frequency token at probability `1.0000` on
every layer — and no amount of tuned lens fixes a missing `ln_f`.

If the wiring is right, the remedy is a **tuned lens**: keep the model's frozen
final norm and unembedding, and learn one affine translator per layer that maps
`h_L` into the final layer's basis first
([Belrose et al., 2023](https://arxiv.org/abs/2303.08112)), so the decode is
`lm_head(ln_f(A_L(h_L)))`. It costs a training pass over a corpus and makes
cross-layer comparisons meaningful. Decoding `lm_head(A_L(h_L))` instead — with
the norm dropped — is a different and worse thing: a LayerNorm is not affine, and
its per-token scale varies 33x across the positions of one GPT-2 prompt.

Other things that quietly invalidate a reading:

- **Skipping the final norm.** `lm_head(resid)` without `ln_f` produces
  scale-wrong logits that still look like a distribution — and a *more* confident
  one than the correct lens.
- **Comparing probabilities across layers as if calibrated.** Use them to locate
  transitions, not as calibrated confidences.
- **Reading only the last position.** Where information appears in the sequence is
  often the more interesting axis.
- **Concluding from one prompt.** Layer indices shift across paraphrases; average
  over a set of prompts before claiming "the fact lives at layer 10".

The lens tells you *where* something becomes decodable, which is correlational. To
show a layer is causally responsible, patch it — see the `activation-patching`
skill.

## Related skills

- `nnsight` — the tracing API, module paths, batching
- `activation-patching` — turning "the answer appears at layer 10" into a causal claim
- `attribution-patching` — the same question at scale via gradients
- `nnsight-remote` — running this on a model too large to host locally
