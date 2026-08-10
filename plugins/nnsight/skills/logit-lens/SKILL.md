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

for layer, token in enumerate(top_per_layer):
    print(f"layer {layer:2d}: {model.tokenizer.decode(token[0])!r}")
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

Do not write `block.output[0]`: a GPT-2 block returns a plain tensor, so that
indexes batch row 0. (See the `nnsight` skill.)

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

The recipe is always *final norm, then unembedding* — only the names change.

| Family | Residual | Final norm | Unembed |
|---|---|---|---|
| GPT-2 | `model.transformer.h[i].output` | `model.transformer.ln_f` | `model.lm_head` |
| Llama / Mistral / Qwen | `model.model.layers[i].output[0]` | `model.model.norm` | `model.lm_head` |
| GPT-NeoX / Pythia | `model.gpt_neox.layers[i].output` | `model.gpt_neox.final_layer_norm` | `model.embed_out` |

```python
llama = TransformersModel("HuggingFaceTB/SmolLM2-135M-Instruct", dispatch=True)

with llama.trace("The capital of France is"):
    tops = nnsight.save([])
    for block in llama.model.layers:
        logits = llama.lm_head(llama.model.norm(block.output[0]))
        tops.append(logits[0, -1].argmax(dim=-1))

print([llama.tokenizer.decode(t) for t in tops[-6:]])
```

Note `block.output[0]` there — on this checkpoint the block returns a tuple.
Confirm per model with `scripts/inspect_model.py` in the `nnsight` skill rather
than copying either form.

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

If you hit that, the fix is a **tuned lens**: fit a per-layer affine probe
(`W_L h_L + b_L`) to match the final-layer distribution, then decode through that.
It requires a training pass over a corpus, but it is the standard remedy and makes
cross-layer comparisons meaningful.

Other things that quietly invalidate a reading:

- **Skipping the final norm.** `lm_head(resid)` without `ln_f` produces
  scale-wrong logits that still look like a distribution.
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
