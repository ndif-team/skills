# Steering and ablation

Both are writes to the residual stream. Steering adds a direction to `layer_output`
and lets every later block read it; ablation removes a component's contribution
(`attention_output`, `mlp_output`, a head's slice of `attention_head_outputs`, or a
whole block via `skip_layers`) and measures the move on `next_token_probs`. The
contribution identity is what makes "zero what the MLP adds" one statement on a
sequential, parallel, sandwich-norm or DeltaNet block. nnter pages:
`docs/patterns/steering.md`, `docs/patterns/ablation.md`, `docs/usage/methods.md`.

<!-- test: setup -->
```python
import torch
import torch.nn.functional as F
from nnter import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", device="cpu", dispatch=True, attn_implementation="eager")
prompt = "The Eiffel Tower is in the city of"
ids = model.tokenizer(" Paris", add_special_tokens=False).input_ids     # encode(" Paris")[0] is BOS on Llama-3 / Gemma
assert len(ids) == 1
paris = ids[0]
LAYER = model.num_layers // 2


def kl(clean_probs, other_probs):
    """KL(clean || other) per row, in float32; xlogy keeps exact zeros (Pythia) from turning into NaN."""
    p, q = clean_probs.float(), other_probs.float()
    return (torch.special.xlogy(p, p) - torch.special.xlogy(p, q)).sum(-1)
```

## Steering

### A direction from two invokes

```python
with model.trace() as tracer:
    with tracer.invoke("I love this so much"):
        positive = model.layers[LAYER].layer_output[:, -1].save()          # [1, hidden]
    with tracer.invoke("I hate this so much"):
        negative = model.layers[LAYER].layer_output[:, -1].save()

vector = (positive - negative)[0]
vector = vector / vector.norm()                                             # unit norm: factors compare across directions
```

For prompt *sets*, pass two lists: each invoke's `[:, -1]` is then a batch of last
positions (a list in one invoke is left-padded), and `.mean(0)` averages the set.

### Scale to the stream, then apply as two invokes

```python
steer_prompt = "I went to the bakery and"
with model.trace(steer_prompt):
    scale = model.layers[LAYER].layer_output[0, -1].norm().save()
factor = 0.5 * float(scale)

with model.trace() as tracer:
    with tracer.invoke(steer_prompt):
        baseline = model.next_token_probs.save()
    with tracer.invoke(steer_prompt):
        model.steer(LAYER, vector, factor=factor, token_positions=-1)      # in place on layer_output
        steered = model.next_token_probs.save()

print(f"norm {float(scale):.0f}", model.probs_to_dict(baseline[0], k=3), model.probs_to_dict(steered[0], k=3))
assert not torch.allclose(baseline, steered)
```

```
norm 92 {' bought': 0.095, ' I': 0.089, ' asked': 0.072} {' I': 0.140, ' they': 0.065, ' it': 0.054}
```

The stream's norm grows with depth (55 at block 0, 417 at block 11 on GPT-2; 33 to
684 on SmolLM2), so a factor that is a nudge late is a demolition early. Read the
factor as a fraction of the measured norm and sweep the fraction, with the layer.
The band is per model: a raw `factor=4.0` barely moves GPT-2, while Gemma-3
degenerates at half the stream norm.

### `steer` is one line you could write

```python
with model.trace(steer_prompt):
    model.steer(LAYER, vector, factor=factor, token_positions=-1)
    by_method = model.next_token_probs.save()

with model.trace(steer_prompt):
    out = model.layers[LAYER].layer_output
    out[:, -1] += factor * vector.to(out)
    by_hand = model.next_token_probs.save()

assert torch.equal(by_hand, by_method)                         # the same write
assert torch.allclose(by_hand, steered, atol=1e-5)             # and the batched row agrees to rounding
```

Use the hand form when the edit is not an add: projecting a direction out
(`out[:, -1] -= (out[:, -1] @ vector)[:, None] * vector`, the refusal-direction
form), clamping, or a different vector per position. `steer(layers, vector,
factor, token_positions, batch_index)` moves the vector to the stream's device and
dtype; `layers` may be a list (ascending), `token_positions` an int, list or slice.
`steer` writes `layer_output`; on Gemma-4 that is after the block's `layer_scalar`,
so adding the same vector to a contribution moves the stream `layer_scalar` times
as much.

### One row of a batch

```python
with model.trace([steer_prompt, steer_prompt]):
    model.steer(LAYER, vector, factor=factor, token_positions=-1, batch_index=1)
    both = model.next_token_probs.save()

assert torch.allclose(both[0], baseline[0], atol=1e-5) and torch.allclose(both[1], steered[0], atol=1e-5)
```

### Under `generate`, at every step

```python
N = 5
with model.generate(steer_prompt, max_new_tokens=N, min_new_tokens=N, do_sample=False) as tracer:
    for step in tracer.iter[:N]:
        model.steer(LAYER, vector, factor=factor, token_positions=-1)
    ids = tracer.result.save()

print(repr(model.tokenizer.decode(ids[0])))
```

```
'I went to the bakery and I was very happy with'
```

A bare `steer` in the body fires once, on the prefill; the output still changes
because the prompt was read differently, which makes the mistake look like a modest
success. `token_positions=-1` is right across steps: the prompt's last token on the
prefill, the one token each decode step processes afterwards. Bound the loop and
hold the run to the bound with `min_new_tokens`; an open `tracer.iter[1:]` that the
run does not reach the end of drops everything after it, `tracer.result.save()`
included.

### Both poles

```python
with model.trace(steer_prompt):
    model.steer(LAYER, vector, factor=-factor, token_positions=-1)
    negated = model.next_token_probs.save()

great, terrible = model.tokenizer.encode(" great")[0], model.tokenizer.encode(" terrible")[0]
assert steered[0, great] > baseline[0, great] and negated[0, terrible] > baseline[0, terrible]
```

Compare against a random direction of the same norm at the same layer before
calling the direction a concept; the `model-steering` skill has the sweep, the
collapse signal and the function-vector variant.

## Ablation

### Canonical: two invokes, measured on `next_token_probs`

```python
with model.trace() as tracer:
    with tracer.invoke(prompt):
        clean = model.next_token_probs.save()
    with tracer.invoke(prompt):
        model.layers[LAYER].mlp.mlp_output[:] = 0                # this invoke's rows only
        no_mlp = model.next_token_probs.save()

print(f"P(Paris) clean {clean[0, paris]:.3f}  no mlp {no_mlp[0, paris]:.3f}  KL {float(kl(clean, no_mlp)[0]):.3f}")
assert no_mlp[0, paris] > clean[0, paris]                       # zero ablation raised the target here
```

`mlp_output` is a tensor, so `[:] = 0` writes in place and reaches the model. The
clean row inside the batched trace is within `6e-7` of the prompt run alone, not
bit-equal, so keep the baseline in the same trace. Neither invoke reads the other's
value, so no barrier is needed.

### Every block's two sublayers in one forward

```python
outs = {}                                                        # container outside; entries saved
with model.trace() as tracer:
    for i in range(model.num_layers):
        with tracer.invoke(prompt):
            model.layers[i].self_attn.attention_output[:] = 0
            outs["attn", i] = model.next_token_probs[0, paris].save()
        with tracer.invoke(prompt):
            model.layers[i].mlp.mlp_output[:] = 0
            outs["mlp", i] = model.next_token_probs[0, paris].save()

print("attn", [round(float(outs["attn", i]), 3) for i in range(model.num_layers)])
print("mlp ", [round(float(outs["mlp", i]), 3) for i in range(model.num_layers)])
assert float(outs["attn", 0]) < 0.001 and float(outs["mlp", 0]) < 0.001      # block 0 is load-bearing under zero
```

```
attn [0.0, 0.052, 0.039, 0.057, 0.085, 0.114, 0.048, 0.048, 0.044, 0.033, 0.056, 0.044]
mlp  [0.0, 0.058, 0.035, 0.043, 0.067, 0.1, 0.08, 0.092, 0.07, 0.053, 0.054, 0.128]
```

Block 0 destroying the prediction under zero ablation is the usual
off-distribution shock, not evidence that block 0 "knows" Paris. Compare against
mean ablation before believing a large zero-ablation drop.

### Mean ablation

```python
reference = ["The Colosseum is in the city of", "The Statue of Liberty is in the city of",
             "Big Ben is in the city of", "The Brandenburg Gate is in the city of"]

with model.trace(reference):
    mean_act = model.layers[LAYER].mlp.mlp_output[:, -1].mean(0).save()    # [hidden]; left-padded, so -1 is each last token

with model.trace(prompt):
    model.layers[LAYER].mlp.mlp_output[:, -1] = mean_act
    mean_ablated = model.next_token_probs.save()

print(f"mean-ablated P(Paris) {mean_ablated[0, paris]:.3f}")             # 0.071 against 0.070 clean, 0.080 zeroed
```

### Whole blocks: `skip_layers`

```python
with model.trace(prompt):
    model.skip_layers(LAYER, LAYER + 1)                          # blocks LAYER..LAYER+1 do not run; stream passes through
    skipped = model.next_token_probs.save()

print(f"skipped P(Paris) {skipped[0, paris]:.3f}")
```

A skip must cover every row of a forward. Inside one invoke of a batched trace it
raises, so it goes in a trace of its own (or in every invoke):

<!-- test: expect-error ValueError -->
```python
with model.trace() as tracer:
    with tracer.invoke(prompt):
        clean_row = model.next_token_probs.save()
    with tracer.invoke(prompt):
        model.skip_layers(LAYER, LAYER)                          # ValueError: A batched .skip() has to cover every row
        skipped_row = model.next_token_probs.save()
```

`skip_layers` reads `layers[start].input`, so read `layers[start - 1].layer_output`
before the call if you want the stream it hands on; a second read of
`layers[start].input` is out of order. Negative indices count from the end.

### One head, two ways, identical logits

```python
HEAD = 3
with model.trace() as tracer:
    with tracer.invoke(prompt):
        model.layers[LAYER].self_attn.attention_head_outputs[..., HEAD, :] = 0     # the head's output slice
        via_outputs = model.logits.save()
    with tracer.invoke(prompt):
        model.layers[LAYER].self_attn.attention_probabilities[:, HEAD] = 0        # the head's pattern
        via_pattern = model.logits.save()

assert torch.allclose(via_outputs, via_pattern, atol=1e-5)
```

Both need `attn_implementation="eager"`. Do not slice the projected
`attention_output` into `head_dim`-wide column blocks: after the output projection
the hidden axis no longer decomposes per head, and that edit removes something that
is not the head.

### Every head of a block in one forward

```python
rows = []
with model.trace() as tracer:
    for head in range(model.num_heads):
        with tracer.invoke(prompt):
            model.layers[LAYER].self_attn.attention_head_outputs[..., head, :] = 0
            rows.append(model.next_token_probs[0, paris].save())

print([round(float(p), 3) for p in rows])
assert len(rows) == model.num_heads
```

### A recurrent mixer on a hybrid

On Qwen3.5 three blocks in four carry `linear_attn` (a DeltaNet); on Jamba,
Nemotron-H and Bamba it is a Mamba mixer. Its contribution is `attention_output`
too, so the mixer is one helper decided outside the trace and the ablation is the
same line. Not in the executed set (needs a hybrid checkpoint); verified on the
tiny Qwen3.5 checkpoint nnter's suite pins:

<!-- test: skip -->
```python
model = StandardizedTransformer("Qwen/Qwen3.5-9B", device="cpu", dispatch=True, attn_implementation="eager")

def mixer(layer):
    attn = getattr(layer, "self_attn", None)
    return attn if attn is not None else layer.linear_attn          # `is not None`, never `or`

mixers = [mixer(layer) for layer in model.layers]                   # outside the trace

outs = []
with model.trace() as tracer:
    with tracer.invoke(prompt):
        base = model.next_token_probs.save()
    for i in range(model.num_layers):
        with tracer.invoke(prompt):
            mixers[i].attention_output[:] = 0
            outs.append(model.next_token_probs.save())

for i, probs in enumerate(outs):
    print(f"block {i:2d} mixer KL {float(kl(base, probs)[0]):.4f}")
```

The same loop runs on a non-hybrid, where every `mixer(layer)` is `self_attn`; the
per-family version is in `references/sweeps-and-probing.md`.

### Measuring

`next_token_probs` is `logits[:, -1].softmax(-1)`, derived and read-only (assign
`logits` instead). Read `model.logits` for a logit difference
(`logits[0, -1, paris] - logits[0, -1, rome]`), which is steadier than a probability
when the baseline is near-degenerate, or for a position other than the last. On a
bf16 checkpoint compute the KL in float32 (the `kl` above casts): `next_token_probs`
is then a bf16 softmax, and a bf16 KL between nearly equal rows can come out
slightly negative.

### Gotchas

- `[:] = 0` is in place; `.clone().save()` first if you also want the clean value.
- `[:, -1]` is the last token of every row only under left padding.
- A checkpoint may lack the component: OPT has no `mlp`, a hybrid's linear blocks
  have no `self_attn`. `model.support()` says so per block; reading raises
  `nnter.Unavailable`.
- Head-level ablation needs eager; `attention_output`, `mlp_output` and
  `layer_output` do not.
- Gemma-4: an in-place edit of a KV-sharing source block's `attention_keys` /
  `attention_values` reaches the 11-20 later blocks that borrow them; assign to keep
  it local. Skipping a source block with `skip_layers` fails (`KeyError`).
- Granite line in bf16: a write at one position of a contribution shifts every
  position by up to half the edit's own effect (the scaled copy is divided back);
  run position-specific ablations in float32.
- Zero, mean and resample answer different questions; say which you ran.
