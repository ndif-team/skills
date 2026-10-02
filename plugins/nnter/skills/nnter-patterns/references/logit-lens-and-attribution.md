# Logit lens and direct logit attribution

Both read the residual stream through the unembedding. The lens applies the model's
whole head (`norm`, `lm_head`, softcap) to a stream or a contribution and asks "what
would this predict alone"; direct logit attribution applies `lm_head` alone to each
contribution so the numbers sum. nnter's `project_on_vocab` is the first,
`model.lm_head(x)` inside a trace is the second, and `layers[i].input +
attention_output + mlp_output == layer_output` is what makes the second decompose on
every family. nnter pages: `docs/patterns/logit-lens.md`,
`docs/patterns/contribution-decomposition.md`, `docs/usage/methods.md`.

<!-- test: setup -->
```python
import torch
from nnter import StandardizedTransformer

model = StandardizedTransformer("openai-community/gpt2", device="cpu", dispatch=True, attn_implementation="eager")
prompt = "The Eiffel Tower is in the city of"
ids = model.tokenizer(" Paris", add_special_tokens=False).input_ids     # encode(" Paris")[0] is BOS on Llama-3 / Gemma
assert len(ids) == 1
paris = ids[0]
```

## Logit lens

### The canonical loop and the wiring check

```python
lens = {}
with model.trace(prompt):
    for i, layer in enumerate(model.layers):
        lens[i] = model.project_on_vocab(layer.layer_output)[:, -1].save()   # [batch, vocab]
    logits = model.logits.save()

decoded = [model.tokenizer.decode(lens[i].argmax(-1)[0]) for i in range(model.num_layers)]
assert decoded[:6] == [" the"] * 6 and decoded[10:] == [" Paris", " Paris"]
assert torch.equal(lens[model.num_layers - 1], logits[:, -1])
```

`project_on_vocab` *calls* `model.norm` and `model.lm_head` on your tensor; reading
`model.lm_head.output` is the model's own projection of the normed final stream, a
different thing. The last-block equality holds on every family because
`project_on_vocab` is the family's whole head: the softcap (Gemma-2/4), `logit_scale`
(Cohere), `/ logits_scaling` (Granite), DeepSeek-V4's `hc_head` before the norm.

### Target-probability curve

```python
with model.trace(prompt):
    curve = torch.stack([model.project_on_vocab(layer.layer_output)[0, -1].softmax(-1)[paris]
                         for layer in model.layers]).save()            # [layers]

print([round(float(p), 3) for p in curve])
assert int(curve.argmax()) == 9 and curve[9] > curve[-1]              # the peak comes before the output
```

```
[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.003, 0.026, 0.248, 0.183, 0.07]
```

Argmax says " Paris" arrives at layer 10; the probability says the decision is
made at layer 9 and the last two blocks *lower* it while keeping it on top. Read
the curve, not one layer, and compare probabilities across layers only to locate
transitions: they are not calibrated confidences.

### Top-k per layer and the every-position grid

```python
topk, grid = {}, {}
with model.trace(prompt):
    for i, layer in enumerate(model.layers):                          # one read per block, in forward order
        row = model.project_on_vocab(layer.layer_output)[0]           # [seq, vocab]
        topk[i] = row[-1].topk(5).indices.save()                      # [5]
        grid[i] = row.argmax(-1).save()                               # [seq]: top-1 at every position

print([model.tokenizer.decode(t) for t in topk[9]])
tokens = [model.tokenizer.decode(t) for t in model.tokenizer(prompt).input_ids]
for i in (0, 6, 11):
    print(f"layer {i:2d}:", [model.tokenizer.decode(t) for t in grid[i]])
```

Column `j` of the grid is the prediction *after* token `j` (`tokens[j]`), a guess at
`tokens[j + 1]`. Read each block once: two comprehensions over `model.layers`, one
for the top-k and one for the grid, ask for block 0 again after block 11 and raise
`OutOfOrderError`. Index inside the trace as above; saving every layer's
`[batch, seq, vocab]` is `layers * seq * vocab` floats for nothing.

`model.get_topk_closest_tokens(hidden, k)` is the same projection then softmax and
top-k, returning `{token: probability}` dicts, on a saved tensor outside the trace or
a live one inside. It keys on the decoded string, so byte tokens that decode to the
same `'�'` collapse into one entry; on non-ASCII text take `topk` of the projection
and keep the ids:

```python
with model.trace(prompt):
    resid = model.layers[model.num_layers // 2].layer_output.save()

top = model.get_topk_closest_tokens(resid[0, -1], k=3)                 # one dict: the last position
print(top)
assert len(top) == 1 and len(top[0]) == 3

by_id = model.project_on_vocab(resid[0, -1]).softmax(-1).topk(3)      # (values, indices): no string keying
assert [model.tokenizer.decode(i) for i in by_id.indices] == list(top[0])
```

### Softcapping

`model.logits` is the model's output; `model.lm_head.output` is the raw projection.
On Gemma-2 they differ, because the forward applies `cap * tanh(logits / cap)` after
`lm_head` (`config.final_logit_softcapping`, `30.0` on `google/gemma-2-2b`).
`project_on_vocab` reads that key and applies the cap, so the wiring check passes
there too and every layer is read on the model's scale; an uncapped lens on a
softcapped model is far too sharp at every layer. Check the config, not a model
list: Gemma-3 sets the key to `None`, GPT-2 has no such key.

```python
assert getattr(model.config, "final_logit_softcapping", None) is None    # gpt2: no cap to apply
```

### Parallel streams (DeepSeek-V4)

DeepSeek-V4's `layer_output` is `[batch, seq, streams, hidden]`. The default lens,
`model.project_on_vocab(layer_output)`, goes through `hc_head`, the model's own readout of
the streams, then `norm` and `lm_head`, so the canonical loop runs unchanged, returns
`[batch, seq, vocab]` and passes the wiring check exactly. A per-stream lens is a different
readout the model never makes: `model.lm_head(model.norm(layer_output[:, :, k]))`. A
hand-written `lm_head(norm(layer_output))` returns `[batch, seq, streams, vocab]` without
an error.

### Reading it honestly

The lens assumes intermediate residuals live in the unembedding's basis. GPT-2's
smooth curve is the exception; SmolLM2 decodes punctuation for 27 of 30 layers and
then " Paris" (see the skill body), while answering correctly. Once the wiring check
passes, a late curve is a fact about the lens, not the model. A tuned lens (one
learned affine map per layer before the frozen norm and head) fixes the basis; it
costs a training pass. The `logit-lens` skill has the full treatment, and the lens
is correlational either way — patch the layer to make a causal claim.

## Direct logit attribution

### The stream is the sum of its contributions

```python
parts = {}
with model.trace(prompt):
    base = model.layers[0].input.save()                     # the stream entering block 0
    for i, layer in enumerate(model.layers):                # block i's attention, then its MLP: forward order
        parts["attn", i] = layer.self_attn.attention_output.save()
        parts["mlp", i] = layer.mlp.mlp_output.save()
    final = model.layers[-1].layer_output.save()

total = base + sum(parts.values())
torch.testing.assert_close(total, final, rtol=1e-4, atol=1e-3)
```

Twenty-five float32 terms with entries in the hundreds land within `1.5e-4` of the
block output on real GPT-2, so the tolerance is relative; `atol=1e-5` alone fails
on this checkpoint while passing on tiny random ones.

### Why `layers[0].input` and not `token_embeddings`

```python
with model.trace(prompt):
    emb = model.token_embeddings.save()                     # read before layers[0].input in one trace
    base = model.layers[0].input.save()

assert not torch.equal(emb, base)                          # GPT-2: base = wte + wpe (after the embedding dropout)
```

`token_embeddings` is the embedding module's output, before positional embeddings
or an embedding norm (BLOOM). On Llama the two are equal (checked on SmolLM2);
`layers[0].input` is what enters block 0 on every family, so it is the base.

### Linear DLA: `lm_head(contribution)` sums

```python
dla = {}
with model.trace(prompt):
    dla["base"] = model.lm_head(model.layers[0].input)[0, -1].save()        # [vocab]
    for i, layer in enumerate(model.layers):
        dla["attn", i] = model.lm_head(layer.self_attn.attention_output)[0, -1].save()
        dla["mlp", i] = model.lm_head(layer.mlp.mlp_output)[0, -1].save()
    head_final = model.lm_head(model.layers[-1].layer_output)[0, -1].save()

assert torch.allclose(sum(dla.values()), head_final, atol=1e-3)
for i in range(model.num_layers):
    print(f"block {i:2d}   attn {float(dla['attn', i][paris]):+7.2f}   mlp {float(dla['mlp', i][paris]):+7.2f}")
```

```
block  9   attn  +10.73   mlp   +7.91
block 10   attn   +6.33   mlp  -16.18
block 11   attn  -59.15   mlp  -42.55
```

`lm_head` has no bias on a tied-embedding checkpoint (GPT-2, Llama); on one that
has a bias, it is added once per call, so subtract it `2 * num_layers` times before
comparing. The numbers are on the unnormed scale: the last block's large negative
attributions are what the final norm rescales away.

**Through the final norm.** Freezing the norm's per-position scale at the final
stream's value makes it linear, but what remains is family-specific, and each piece
missed skews every term: a LayerNorm subtracts each term's mean and adds its bias
once (not per term); an RMSNorm multiplies by its weight, which Gemma stores as
`w` and applies as `1 + w`; Granite divides the head's output by
`logits_scaling`, Cohere multiplies by `logit_scale`; a softcap is not linear at
all. Check a hand-built version against `project_on_vocab(final)` before using it.

### The normed lens does not sum

```python
normed = {}
with model.trace(prompt):
    for i, layer in enumerate(model.layers):
        normed["attn", i] = model.project_on_vocab(layer.self_attn.attention_output)[0, -1].save()
        normed["mlp", i] = model.project_on_vocab(layer.mlp.mlp_output)[0, -1].save()
    real = model.logits[0, -1].save()

assert not torch.allclose(sum(normed.values()), real, atol=1e-2)        # the norm was applied per term
```

Use the linear form for attribution *shares* and the normed lens for "what does
this term say by itself"; say which you used. On a softcapped model the logits are
not a sum of anything: `project_on_vocab` applies the cap, the linear form does not.

### Per head

`attention_head_outputs` is each head's output before concatenation and the output
projection, `[batch, seq, heads, head_dim]`. The projection is linear, so head `h`'s
share of `attention_output` is its slice times the projection weight's matching
rows. The projection keeps its native name (`o_proj`, `c_proj`, `dense`, `out_proj` on GPT-Neo
and CodeGen) and its weight layout differs: `torch.nn.Linear` stores `[out, in]`,
GPT-2's `Conv1D` stores `[in, out]`.

```python
LAYER = model.num_layers // 2
attention = model.layers[LAYER].self_attn

projection = None                                   # `or` over envoys truth-tests the module's __len__: TypeError
for name in ("o_proj", "c_proj", "dense", "out_proj"):
    candidate = getattr(attention, name, None)
    if candidate is not None:
        projection = candidate
        break

with model.trace(prompt):
    heads = attention.attention_head_outputs.save()          # [batch, seq, heads, head_dim]
    contribution = attention.attention_output.save()         # [batch, seq, hidden]

W = projection._module.weight
W_in_out = W.t() if isinstance(projection._module, torch.nn.Linear) else W       # [heads * head_dim, hidden]
H, D = heads.shape[2], heads.shape[3]
per_head = torch.einsum("bshd,hdo->bsho", heads, W_in_out.reshape(H, D, -1))      # [batch, seq, heads, hidden]
bias = projection._module.bias
assert torch.allclose(per_head.sum(2) + (bias if bias is not None else 0), contribution, atol=1e-4)

head_dla = per_head[0, -1] @ model.lm_head._module.weight.t()                      # [heads, vocab], linear
print([round(float(x), 2) for x in head_dla[:, paris]])
```

The same numbers come from the model: zero every head but `h` in
`attention_head_outputs` and read `attention_output`; that equals
`per_head[..., h, :]` plus the bias. Under grouped-query attention the head axis is
still `num_heads` (query heads), so the slicing is unchanged; on SmolLM2
`o_proj` is a `Linear [576, 576]` and the check passes exactly. The heads sum to
`attention_output` only where that is the projection's output: on the Granite line
multiply by `residual_multiplier` (0.22-0.28), on post-norm families (Gemma-2/3/4,
OLMo-2/3, EXAONE-4, FlexOlmo, GLM-4, AFMoE) the sum is the *pre-norm* projection
output, and on Laguna an output gate follows the heads.

### Caveats

- **Gemma-4, Doge, ZAYA and DeepSeek-V4 are not a plain sum.** Gemma-4 multiplies
  each block's sum by `layer_scalar` (a term from block `i` reaches the end times
  every scalar from `i` on), Doge and ZAYA rescale the stream per channel, and
  DeepSeek-V4's stream is `[batch, seq, streams, hidden]`, mixed per block. Naive DLA
  there ranks the wrong terms (by ~7,700% on gemma-4-E2B). The weighted sums, verified
  exact: the `nnter` skill's `references/values.md`, "Blocks that are not a plain sum".
  The logit lens itself is unaffected (the final norm removes a scale;
  `project_on_vocab` handles the streams).
- **Post-norm families** (Gemma-2/3/4, OLMo-2/3, EXAONE-4, FlexOlmo, GLM-4, AFMoE).
  `attention_output` is the post-attention norm's output, and the per-head sum is
  the *pre-norm* projection output. Compare against `o_proj.output` there.
- **dtype.** The sums are exact in float32. SmolLM2 loads in bf16 and its 61-term
  linear DLA sum is off by 8 on logits of range 356; load with
  `dtype=torch.float32` for the check, or use a relative tolerance and expect a
  couple of percent.
- **Order.** Block `i`'s `attention_output` then its `mlp_output`, then block `i + 1`.
  Two comprehensions, one over all attentions then one over all MLPs, raise
  `OutOfOrderError`. `token_embeddings` before `layers[0].input`.
- **Three different heads.** `model.lm_head(x)` is a stood-down call on your tensor;
  `model.lm_head.output` is the model's projection of the normed stream;
  `model.logits` adds the softcap.
