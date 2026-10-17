---
name: logit-lens
description: Decode what a model predicts at every layer by applying its final norm and unembedding to intermediate residual streams. Use to see where in depth an answer emerges, to track a specific token's probability across layers, to sanity-check that a model knows a fact before running invasive interventions, or to compare prediction trajectories between prompts. Includes the one-line wiring check that catches a wrong norm, a wrong unembedding or logit softcapping before you plot, per-architecture variants, the tuned-lens caveat, and the failure modes that make a logit lens look informative when it is not.
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

These tokens come from a float32 run, and several are near-ties: layer 9's top two
logits are 0.14 apart, layer 10's 0.05. Loaded in bfloat16, layer 9 reads
`' Paris'` instead of `' London'`. Track one token's probability or rank
([below](#the-probability-trajectory-of-one-token)) rather than asserting a
layer's exact argmax.

The whole sweep is **one forward pass**. Calling `model.lm_head(...)` inside the
trace runs its `forward` directly with no nnsight interleaving (the module's own
PyTorch hooks still fire), so you can apply the unembedding to a layer-5
activation without re-running anything.

Do not write `block.output[0]`: a block returns a plain tensor on every family
tested, so that indexes batch row 0 — no error, and a lens that decodes the same
token at every layer. (See the `nnsight` skill.)

## Check the wiring before you read anything off it

Applied to the **last** block, the lens is the model's own final computation, so
it must reproduce the model's logits exactly. One line tells you whether the
norm, the head and the block output you picked are the right three:

```python
cap = getattr(model.config, "final_logit_softcapping", None)   # None on GPT-2

with model.trace(prompt):
    lens = model.lm_head(model.transformer.ln_f(model.transformer.h[-1].output))
    if cap is not None:
        lens = torch.tanh(lens / cap) * cap
    lens = lens.save()
    real = model.output.logits.save()

assert torch.equal(lens, real), (lens - real).abs().max()
```

Run it on any model before plotting. A mismatch means one of three things, none
of which raises on its own:

- **the wrong norm**, or none — `lm_head(resid)` without the final norm decodes
  `' the'` at probability `1.0000` at *every* GPT-2 layer: a confident-looking,
  scale-wrong lens that reads exactly like the "does not transfer" symptom below;
- **the wrong head or block output** — a head attribute the model does not have,
  or a block output indexed like a tuple;
- **the model post-processes the logits.** `google/gemma-2-2b` softcaps them
  (`config.final_logit_softcapping = 30.0`); without the cap the check fails by
  `51.0` and the distribution is far too sharp (max probability `0.9995` against
  the model's `0.9257`). The `cap` lines above reapply it, so every layer is read
  on the model's own scale. Check `model.config` rather than a list of model
  names — Gemma-3 sets the cap to `None`.

## The probability trajectory of one token

Tracking one target gives a curve you can compare across prompts or
interventions; the top-5 alongside shows what it competes with:

```python
paris = model.tokenizer.encode(" Paris")[0]

with model.trace(prompt):
    trajectory = nnsight.save([])
    top5 = nnsight.save([])
    for block in model.transformer.h:
        probs = model.lm_head(model.transformer.ln_f(block.output))[0, -1].softmax(dim=-1)
        trajectory.append(probs[paris])
        top5.append(probs.topk(5).indices)

for layer, (p, ids) in enumerate(zip(trajectory, top5)):
    print(f"layer {layer:2d}  P(' Paris') = {p.item():.4f}  {[model.tokenizer.decode(i) for i in ids]}")

assert max(range(12), key=lambda layer: trajectory[layer].item()) == 9
assert trajectory[-1].item() < trajectory[9].item()
```

`P(' Paris')` peaks at `0.248` at layer 9 — while `' London'` is still top-1 —
and `' Paris'` only becomes top-1 at layer 10. The final layer gives `0.0700`,
*less* confident than the peak. Lens probabilities locate transitions; they are
not calibrated confidences.

## Every position, not just the last

The same decode across the sequence gives the classic heatmap — what the model
would predict *at each position* if it stopped at each layer:

```python
with model.trace(prompt):
    grid = nnsight.save([
        model.lm_head(model.transformer.ln_f(block.output))[0].argmax(dim=-1)
        for block in model.transformer.h
    ])                                               # layers x [seq]

tokens = [model.tokenizer.decode([i]) for i in model.tokenizer(prompt).input_ids]
print(f"{'':12}" + "".join(f"L{i:<7}" for i in range(0, 12, 3)))
for pos, token in enumerate(tokens):
    print(f"{token!r:12}" + "".join(f"{model.tokenizer.decode(grid[i][pos]):<8}" for i in range(0, 12, 3)))
```

## Comparing two prompts in one pass

Put each prompt in its own invoke — one forward pass, directly comparable curves:

```python
answers = {"The Eiffel Tower is in the city of": " Paris",
           "The Colosseum is in the city of": " Rome"}

with model.trace() as tracer:
    curves = nnsight.save({})
    for text, answer in answers.items():
        target = model.tokenizer.encode(answer)[0]
        with tracer.invoke(text):
            curves[answer] = [
                model.lm_head(model.transformer.ln_f(block.output))[0, -1].softmax(dim=-1)[target]
                for block in model.transformer.h
            ]

for layer, (a, b) in enumerate(zip(curves[" Paris"], curves[" Rome"])):
    print(f"layer {layer:2d}  P(Paris|Eiffel)={a.item():.3f}   P(Rome|Colosseum)={b.item():.3f}")
```

## Other architectures

The recipe is always *final norm, then unembedding* — only the names change
(verified on `transformers` 5.15):

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
case; GPT-2's smooth trajectory is the exception. Confirm the row for your own
checkpoint by running the wiring check on it.

## During generation

Wrap the sweep in a bounded iteration loop to watch the trajectory at every
generated token:

```python
with model.generate(prompt, max_new_tokens=3, do_sample=False) as tracer:
    per_step = nnsight.save([])
    for step in tracer.iter[:3]:
        logits = model.lm_head(model.transformer.ln_f(model.transformer.h[8].output))
        per_step.append(logits[0, -1].argmax(dim=-1))
    ids = tracer.result.save()

print("layer-8 guess per step:", [model.tokenizer.decode(t) for t in per_step])
print("actually generated:    ", model.tokenizer.decode(ids[0, -3:]))
```

`do_sample=False` is load-bearing. `generate` follows the checkpoint's own
`generation_config`, and most instruct checkpoints set `do_sample=True` there —
so a **final-layer** lens reading that disagrees with the token that came out is
sampling, not a broken lens. Rule out sampling before you go looking for a wrong
norm or a wrong unembedding.

## Reading the result honestly

**The logit lens is a projection, not the model's belief.** It assumes intermediate
residuals live in the same basis the unembedding expects. That holds well for
GPT-2 and poorly for several later models — a flat or nonsensical curve may mean
the lens does not transfer, not that the model knows nothing. Symptoms: every
layer decodes to the same high-frequency token, or probabilities stay near zero
until the final layer and then jump.

Before concluding that, run the wiring check: a missing final norm produces the
same symptom, and no tuned lens fixes a missing `ln_f`.

If the wiring is right, the remedy is a **tuned lens**
([Belrose et al., 2023](https://arxiv.org/abs/2303.08112)): learn one affine
translator per layer and decode `lm_head(ln_f(A_L(h_L)))`, keeping the frozen
final norm. Dropping the norm (`lm_head(A_L(h_L))`) is worse, not equivalent — a
LayerNorm is not affine, and its per-token scale varies 33x across one GPT-2
prompt.

Other things that quietly invalidate a reading:

- **Comparing probabilities across layers as if calibrated.** The final layer
  can be less confident than a middle one (above); use them to locate transitions.
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
- `remote` — running this on a model too large to host locally
